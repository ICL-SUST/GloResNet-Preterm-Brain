import csv
import glob
import os
import torch
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from torch.utils.data import Dataset

class DHCP3DGlobalDataset(Dataset):
    """
    Stage 5 Global Dataset:
    - Loads full brain MRI
    - Resizes to fixed target shape (default 128x128x128)
    - No random cropping
    - Optional Augmentation: Random Flip
    """
    def __init__(
        self,
        root_dir: str,
        label_csv: str = None,
        target_shape=(128, 128, 128),
        label_col: str = "label",
        subject_col: str = "subject",
        session_col: str = "session",
        use_restore: bool = True,
        augment: bool = True,
    ) -> None:
        self.root_dir = root_dir
        self.target_shape = np.array(target_shape)
        self.augment_enabled = augment
        
        # If no CSV path provided, default to the one in final_submission/data/
        if label_csv is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            label_csv = os.path.join(project_root, "data", "dhcp_preterm_score_labeled.csv")
            
        print(f"[Dataset] Reading labels from {label_csv}...")
        
        # Load Labels
        self.labels = self._load_labels(label_csv, label_col, subject_col, session_col)
        
        # Find Files
        pattern = "*desc-restore_T2w.nii.gz" if use_restore else "*_T2w.nii.gz"
        # Deterministic sort is critical
        img_paths = sorted(glob.glob(os.path.join(root_dir, "sub-*", "ses-*", "anat", pattern)))
        
        self.samples = []
        missing_count = 0
        
        for img_path in img_paths:
            sub, ses = self._parse_ids(img_path)
            label = self._lookup_label(sub, ses)
            
            if label is not None:
                self.samples.append((img_path, int(label)))
            else:
                missing_count += 1
                
        print(f"[Stage5-Global] Loaded {len(self.samples)} samples. Target Shape: {target_shape}. Skipped {missing_count} unlabelled.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 1. Load Image
        try:
            nii = nib.load(img_path)
            img = nii.get_fdata()
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            raise e
            
        # 2. Resize to Target Shape
        current_shape = img.shape
        zoom_factors = self.target_shape / np.array(current_shape)
        
        # Order=1 (Linear) is fine for MRI and faster than cubic
        img_resized = zoom(img, zoom_factors, order=1)
        
        # 3. Normalization (Z-score)
        mean = np.mean(img_resized)
        std = np.std(img_resized)
        if std > 0:
            img_resized = (img_resized - mean) / std
        
        # 4. Augmentation (Simple Flips)
        if self.augment_enabled:
            # Random Flip along sagittal axis (axis 0)
            if np.random.rand() > 0.5:
                img_resized = np.flip(img_resized, axis=0)
            
        # 5. Add Channel Dim -> [1, D, H, W]
        img_tensor = torch.from_numpy(img_resized.copy()).float().unsqueeze(0)
        
        return img_tensor, label

    def _load_labels(self, csv_path, label_col, sub_col, ses_col):
        label_dict = {}
        if not os.path.exists(csv_path):
            print(f"[Warning] Label CSV file not found at {csv_path}")
            return label_dict
            
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                subj = row.get(sub_col)
                ses = row.get(ses_col) if ses_col in row else None
                lab = row.get(label_col)
                
                if not subj or not lab:
                    continue
                    
                key = (subj.strip(), ses.strip() if ses else None)
                
                # Handle string labels
                if lab == "ge3":
                    label_dict[key] = 1
                elif lab == "le2":
                    label_dict[key] = 0
                else:
                    try:
                        label_dict[key] = int(float(lab))
                    except:
                        pass
        return label_dict

    def _parse_ids(self, path):
        parts = path.split(os.sep)
        sub = next((p for p in parts if p.startswith("sub-")), "").replace("sub-", "")
        ses = next((p for p in parts if p.startswith("ses-")), "").replace("ses-", "")
        return sub, ses

    def _lookup_label(self, sub, ses):
        # 1. Try exact match (subject, session)
        if (sub, ses) in self.labels:
            return self.labels[(sub, ses)]
        # 2. Try subject fallback (subject, None)
        if (sub, None) in self.labels:
            return self.labels[(sub, None)]
        return None
