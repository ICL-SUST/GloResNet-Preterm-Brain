# Infant Brain Injury Prediction from 3D MRI

Repository for the Thesis: **Predicting Infant Brain Injury from 3D MRI Scans using Deep Learning**

---

## Overview

This repository contains the complete, organized codebase for predicting infant brain injury from 3D T2-weighted MRI volumes.
Our pipeline utilizes a lightweight **ResNet-10** backbone pre-trained on 23 medical datasets (**MedicalNet**), combined with **Global Resize** preprocessing, **Mixup** augmentation, and **Test-Time Augmentation (TTA)**.

---

## Project Structure

```
├── data/                             # Dataset metadata and instructions
│   ├── dhcp_preterm_score_labeled.csv # Labeled clinical/demographic metadata (129 subjects)
│   └── README.md                     # Instructions on dHCP dataset setup
├── src/                              # Core library modules
│   ├── __init__.py                   # Package initialization
│   ├── dataset.py                    # 3D MRI Dataset loader with z-score & flips
│   └── model.py                      # 3D ResNet models (ResNet-10/18/50)
├── scripts/                          # Training pipelines
│   ├── train_cv.py                   # Main 5-Fold Cross Validation script
│   └── train_single.py               # Single train-test split script
├── .gitignore                        # Git ignore rules
├── requirements.txt                  # Python package dependencies
├── run_cv.sh                         # Bash runner for 5-Fold Cross Validation
├── run_single.sh                     # Bash runner for Single Model training
└── README.md                         # This file
```

---

## Setup & Installation

### 1. Requirements

Ensure you have Python 3.8+ installed. Install dependencies via pip:

```bash
pip install -r requirements.txt
```

### 2. Dataset Setup

Raw MRI scans are sourced from the **developing Human Connectome Project (dHCP)**.

- Download the 3D T2w MRI volumes (bias-corrected: `*desc-restore_T2w.nii.gz`).
- Arrange them in BIDS folder style: `sub-<ID>/ses-<Session>/anat/sub-<ID>_ses-<Session>_desc-restore_T2w.nii.gz`
- Refer to [data/README.md](file:///data/README.md) for more details.

### 3. Pretrained Weights

This pipeline leverages **MedicalNet** weights. Download the 3D ResNet weights from the [MedicalNet official repo](https://github.com/Tencent/MedicalNet):

- Download `resnet_10_23dataset.pth` (for ResNet-10) and/or `resnet_50_23dataset.pth` (for ResNet-50).
- Place them in a local directory (e.g., `weights/` or `stage4-medicalnet/weights/`).

---

## Running Training

All Python scripts are parametrizable and accept CLI arguments. You can run them via the provided shell runners or customize paths directly:

### 5-Fold Cross Validation

Run:

```bash
bash run_cv.sh
```

Or run python directly:

```bash
python scripts/train_cv.py \
    --weights_path "weights/resnet_10_23dataset.pth" \
    --data_dir "/path/to/dhcp_dataset_root" \
    --epochs 150 \
    --batch_size 32
```

### Single Split Training

To train a single model:

```bash
bash run_single.sh
```

---
