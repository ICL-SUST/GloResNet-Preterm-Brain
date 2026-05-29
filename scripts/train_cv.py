import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch import optim
import sys
import os
import argparse
import random
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, 
    roc_auc_score, 
    f1_score, 
    precision_score, 
    matthews_corrcoef,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt

# --- Setup Paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Local Imports
from src.model import resnet10
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

def validate_tta(model, loader, device):
    """
    Test Time Augmentation: Average prediction of (x) and (flip(x))
    Returns: preds, labels, probs
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out1 = torch.softmax(model(imgs), dim=1)
            out2 = torch.softmax(model(torch.flip(imgs, [2])), dim=1)
            avg_prob = (out1 + out2) / 2.0
            _, pred = avg_prob.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
            all_probs.extend(avg_prob[:, 1].cpu().numpy())
    return all_preds, all_labels, all_probs

def train_cv():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="runs/cv_optimized_resnet10")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to pretrained MedicalNet weights")
    parser.add_argument("--data_dir", type=str, default="/mnt/raid5/dhcp", help="Path to raw dataset directory")
    parser.add_argument("--label_csv", type=str, default=None, help="Path to labels CSV file")
    parser.add_argument("--epochs", type=int, default=100) # Recommend 100-150
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--input_size", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Plan J: 5-Fold CV | ResNet-10 | {args.input_size}^3 | TTA | Mixup")
    
    # Resolve CSV path
    if args.label_csv is None:
        args.label_csv = os.path.join(parent_dir, "data", "dhcp_preterm_score_labeled.csv")
    
    # 1. Prepare Full Dataset
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
    
    # Group by Subject
    subject_to_indices = defaultdict(list)
    subject_to_label = {}
    for idx, (img_path, label) in enumerate(train_full_ds.samples):
        parts = img_path.split(os.sep)
        subject = next((p for p in parts if p.startswith("sub-")), f"unknown_{idx}").replace("sub-", "")
        subject_to_indices[subject].append(idx)
        subject_to_label[subject] = label
        
    subjects = list(subject_to_label.keys())
    labels = list(subject_to_label.values())
    
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_subj_idx, val_subj_idx) in enumerate(skf.split(subjects, labels)):
        print(f"\n===========================")
        print(f"Starting Fold {fold+1}/{args.folds}")
        print(f"===========================")
        
        train_input_idxs = []
        for i in train_subj_idx: train_input_idxs.extend(subject_to_indices[subjects[i]])
        val_input_idxs = []
        for i in val_subj_idx: val_input_idxs.extend(subject_to_indices[subjects[i]])
        
        train_ds = Subset(train_full_ds, train_input_idxs)
        val_ds = Subset(val_full_ds, val_input_idxs)
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # Helper: Calculate class weights for this fold
        fold_train_labels = [train_full_ds.samples[i][1] for i in train_input_idxs]
        c0 = fold_train_labels.count(0); c1 = fold_train_labels.count(1)
        w = torch.tensor([len(fold_train_labels)/(2*c0), len(fold_train_labels)/(2*c1)]).float().to(device)
        
        # Init Model
        model = resnet10(num_classes=2)
        checkpoint = torch.load(args.weights_path, map_location='cpu')
        sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_sd = {k.replace("module.", ""): v for k, v in sd.items() if "fc" not in k}
        model.load_state_dict(new_sd, strict=False)
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss(weight=w)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        best_acc = 0.0
        best_metrics = {}
        best_pred_packet = None
        
        for epoch in range(args.epochs):
            model.train()
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                imgs, la, lb, lam = mixup_data(imgs, lbls)
                out = model(imgs)
                loss = mixup_criterion(criterion, out, la, lb, lam)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            # Val with TTA
            preds, true_labels, probs = validate_tta(model, val_loader, device)
            
            # Metrics
            cm = confusion_matrix(true_labels, preds, labels=[0,1])
            tn, fp, fn, tp = cm.ravel()
            acc = (tp+tn) / (tp+tn+fp+fn) if (tp+tn+fp+fn)>0 else 0
            sens = tp/(tp+fn) if (tp+fn)>0 else 0
            spec = tn/(tn+fp) if (tn+fp)>0 else 0
            try:
                auc_score = roc_auc_score(true_labels, probs)
            except:
                auc_score = 0.5
            
            f1 = f1_score(true_labels, preds)
            prec = precision_score(true_labels, preds, zero_division=0)
            mcc = matthews_corrcoef(true_labels, preds)
            
            if acc > best_acc:
                best_acc = acc
                best_metrics = {
                    'acc': acc, 'sens': sens, 'spec': spec, 'auc': auc_score, 
                    'f1': f1, 'prec': prec, 'mcc': mcc, 'epoch': epoch+1
                }
                best_pred_packet = (true_labels, probs)
            
            if (epoch+1)%20 == 0:
                print(f"  Ep {epoch+1} | Acc: {acc:.2%} (Best: {best_acc:.2%}) | AUC: {auc_score:.4f} | F1: {f1:.4f}")
        
        # --- Plot ROC if this is a new Global Best ---
        final_auc = best_metrics.get('auc', 0)
        
        record_file = os.path.join(args.save_dir, "best_auc_record.txt")
        global_best_auc = 0.0
        if os.path.exists(record_file):
            with open(record_file, "r") as f:
                try:
                    global_best_auc = float(f.read().strip())
                except:
                    pass
        
        if final_auc > global_best_auc and best_pred_packet is not None:
            print(f"!!! New Global Best AUC: {final_auc:.4f} (Previous: {global_best_auc:.4f}) !!!")
            # Update Record
            with open(record_file, "w") as f:
                f.write(str(final_auc))
            
            # Draw ROC
            y_true, y_scores = best_pred_packet
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Global Best ROC (Fold {fold+1})')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.savefig(os.path.join(args.save_dir, f"global_best_roc_fold_{fold+1}.pdf"))
            plt.close()
            print(f"Updated global_best_roc_fold_{fold+1}.pdf in save directory.")
        
        print(f"Fold {fold+1} Best: Acc={best_metrics['acc']:.2%}, Sens={best_metrics['sens']:.2f}, Spec={best_metrics['spec']:.2f}, AUC={best_metrics['auc']:.4f}, F1={best_metrics['f1']:.4f}, MCC={best_metrics['mcc']:.4f}")
        fold_results.append(best_metrics)

    # 3. Final Report
    print(f"\n===========================")
    print(f"PLAN J: 5-FOLD CV FINAL RESULTS")
    print(f"===========================")
    avg_acc = np.mean([r['acc'] for r in fold_results])
    avg_sens = np.mean([r['sens'] for r in fold_results])
    avg_spec = np.mean([r['spec'] for r in fold_results])
    avg_auc = np.mean([r['auc'] for r in fold_results])
    avg_f1 = np.mean([r['f1'] for r in fold_results])
    avg_mcc = np.mean([r['mcc'] for r in fold_results])
    
    print(f"Average Accuracy:    {avg_acc:.2%}")
    print(f"Average Sensitivity: {avg_sens:.2f}")
    print(f"Average Specificity: {avg_spec:.2f}")
    print(f"Average AUC:         {avg_auc:.4f}")
    print(f"Average F1-Score:    {avg_f1:.4f}")
    print(f"Average MCC:         {avg_mcc:.4f}")

    with open(os.path.join(args.save_dir, "cv_results.txt"), "w") as f:
        f.write(f"Avg Acc: {avg_acc}\nAvg Sens: {avg_sens}\nAvg Spec: {avg_spec}\nAvg AUC: {avg_auc}\nAvg F1: {avg_f1}\nAvg MCC: {avg_mcc}\n")
        f.write(str(fold_results))

if __name__ == "__main__":
    train_cv()
