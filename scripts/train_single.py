import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch import optim
from tqdm import tqdm
import sys
import os
import argparse
import random
import numpy as np
from collections import defaultdict
import torch.nn.functional as F

# --- Setup Paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Local Imports
from src.model import resnet50, resnet18, resnet10
from src.dataset import DHCP3DGlobalDataset

# Reuse mixup
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_stratified_subject_split(dataset, train_ratio=0.8, seed=42):
    random.seed(seed)
    subject_to_indices = defaultdict(list)
    subject_to_label = {}
    
    for idx, (img_path, label) in enumerate(dataset.samples):
        parts = img_path.split(os.sep)
        subject = next((p for p in parts if p.startswith("sub-")), f"unknown_{idx}").replace("sub-", "")
        subject_to_indices[subject].append(idx)
        subject_to_label[subject] = label
    
    class0_subjects = [s for s, l in subject_to_label.items() if l == 0]
    class1_subjects = [s for s, l in subject_to_label.items() if l == 1]
    
    random.shuffle(class0_subjects)
    random.shuffle(class1_subjects)
    
    n_train_c0 = int(len(class0_subjects) * train_ratio)
    n_train_c1 = int(len(class1_subjects) * train_ratio)
    
    train_subs = class0_subjects[:n_train_c0] + class1_subjects[:n_train_c1]
    val_subs = class0_subjects[n_train_c0:] + class1_subjects[n_train_c1:]
    
    train_indices = []
    for s in train_subs: train_indices.extend(subject_to_indices[s])
    val_indices = []
    for s in val_subs: val_indices.extend(subject_to_indices[s])
    
    return train_indices, val_indices

def validate_tta(model, loader, device):
    """
    Test Time Augmentation: Average prediction of (x) and (flip(x))
    """
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            # Forward Original
            out1 = torch.softmax(model(imgs), dim=1)
            
            # Forward Flip (Sagittal flip)
            imgs_flip = torch.flip(imgs, dims=[2])
            out2 = torch.softmax(model(imgs_flip), dim=1)
            
            # Average prob
            avg_prob = (out1 + out2) / 2.0
            _, pred = avg_prob.max(1)
            
            correct += pred.eq(lbls).sum().item()
            total += lbls.size(0)
    return 100. * correct / total

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="runs/optimized_resnet50")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to pretrained weights")
    parser.add_argument("--data_dir", type=str, default="/mnt/raid5/dhcp", help="Path to raw dataset directory")
    parser.add_argument("--label_csv", type=str, default=None, help="Path to labels CSV file")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16) 
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet10", "resnet18", "resnet50"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--input_size", type=int, default=112, help="Input dim size (cube)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Model: {args.model} | Input: {args.input_size}^3")
    
    # 1. Model
    if args.model == "resnet50":
        model = resnet50(num_classes=2)
    elif args.model == "resnet18":
        model = resnet18(num_classes=2)
    else:
        model = resnet10(num_classes=2)
        
    # Load Weights
    print(f"Loading weights from {args.weights_path}...")
    checkpoint = torch.load(args.weights_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        if "fc" in name: continue
        new_state_dict[name] = v
    
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"  Missing keys: {len(msg.missing_keys)}") 
    model = model.to(device)
    
    # Resolve CSV path
    if args.label_csv is None:
        args.label_csv = os.path.join(parent_dir, "data", "dhcp_preterm_score_labeled.csv")
    
    # 2. Data
    train_full_ds = DHCP3DGlobalDataset(
        root_dir=args.data_dir,
        label_csv=args.label_csv,
        label_col="score_group",
        subject_col="participant_id",
        session_col="session_id",
        target_shape=(args.input_size, args.input_size, args.input_size),
        augment=True
    )
    
    val_full_ds = DHCP3DGlobalDataset(
        root_dir=args.data_dir,
        label_csv=args.label_csv,
        label_col="score_group",
        subject_col="participant_id",
        session_col="session_id",
        target_shape=(args.input_size, args.input_size, args.input_size),
        augment=False
    )
    
    train_idxs, val_idxs = get_stratified_subject_split(train_full_ds)
    train_ds = Subset(train_full_ds, train_idxs)
    val_ds = Subset(val_full_ds, val_idxs)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Class Weights
    fold_train_labels = [train_full_ds.samples[i][1] for i in train_idxs]
    counts = defaultdict(int)
    for l in fold_train_labels: counts[l]+=1
    w = torch.tensor([len(fold_train_labels)/(2*counts[0]), len(fold_train_labels)/(2*counts[1])]).float().to(device)
    
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0
        correct = 0; total = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            
            imgs, la, lb, lam = mixup_data(imgs, lbls)
            out = model(imgs)
            loss = mixup_criterion(criterion, out, la, lb, lam)
            
            loss.backward()
            optimizer.step()
            
            ep_loss += loss.item()
            _, pred = out.max(1)
            correct += (lam*pred.eq(la).sum().float() + (1-lam)*pred.eq(lb).sum().float()).item()
            total += lbls.size(0)
            
            pbar.set_postfix({'loss': loss.item()})
            
        scheduler.step()
        train_acc = 100. * correct / total
        
        # Validation with TTA
        val_acc = validate_tta(model, val_loader, device)
        
        print(f"  Ep {epoch+1} | Loss: {ep_loss/len(train_loader):.4f} | TrAcc: {train_acc:.1f}% | ValAcc(TTA): {val_acc:.1f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"{args.model}_tta_best_{val_acc:.1f}.pth"))

if __name__ == "__main__":
    train()
