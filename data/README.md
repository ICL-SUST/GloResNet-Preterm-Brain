# Dataset Setup Instructions

To run the training and evaluation pipelines, you need to acquire the MRI scan files and place them in the directory specified by your command-line arguments (or configure the default path).

## 1. Demographic Metadata (`dhcp_preterm_score_labeled.csv`)

The file `data/dhcp_preterm_score_labeled.csv` (already included in this directory) provides clinical and demographic metadata for the 129 subjects in this study.
It includes the following columns:
- `participant_id`: Subject ID (e.g., `CC00576XX16`)
- `gender`: Subject gender (`Male` or `Female`)
- `birth_age`: Gestational age at birth in weeks (e.g., `28.85`)
- `birth_weight`: Birth weight in kg (e.g., `0.85`)
- `singleton`: Singleton or multiple birth status
- `radiology_score`: Clinical score assigned by pediatric neuroradiologists
- `scan_age`: Post-menstrual age (PMA) at the time of MRI scan in weeks (e.g., `35.85`)
- `score_group`: Target label for classification:
  - `le2` (Class 0): Normal / Mild injury (score <= 2)
  - `ge3` (Class 1): Moderate / Severe injury (score >= 3)

---

## 2. 3D MRI Image Files

The raw MRI volumes are from the **developing Human Connectome Project (dHCP)** dataset. 

### Expected File Format
The dataset loader expects 3D MRI scans in **NIfTI** format (`.nii.gz`). Specifically, it queries for T2-weighted structural images:
- Pattern: `*desc-restore_T2w.nii.gz` (bias-field restored T2-weighted scans) or `*_T2w.nii.gz` (standard T2-weighted scans).

### Directory Hierarchy
The loader searches for files recursively in the following BIDS-like folder structure under the dataset root directory:
```
<DATASET_ROOT_DIR>/
├── sub-<participant_id>/
│   └── ses-<session_id>/
│       └── anat/
│           └── sub-<participant_id>_ses-<session_id>_desc-restore_T2w.nii.gz
└── ...
```
For example, if participant is `CC00576XX16` and session is `16`:
`sub-CC00576XX16/ses-16/anat/sub-CC00576XX16_ses-16_desc-restore_T2w.nii.gz`

---

## 3. How to Configure Paths
When executing the scripts, you can specify:
- `--root_dir` (or `--data_dir`): The path to your local `<DATASET_ROOT_DIR>`
- `--label_csv`: The path to `dhcp_preterm_score_labeled.csv` (defaults to `data/dhcp_preterm_score_labeled.csv`)
