#!/bin/bash
# ------------------------------------------------------------------
# 5-Fold Cross Validation Runner: Infant Brain Injury Prediction
# ------------------------------------------------------------------

export PYTHONPATH=$PYTHONPATH:.

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="runs/cv_resnet10_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

# --- Parameters (Change these to match your environment) ---
WEIGHTS="weights/resnet_10_23dataset.pth"     # Path to pretrained MedicalNet weights
DATA_DIR="/mnt/raid5/dhcp"                    # Path to BIDS dataset directory
LABEL_CSV="data/dhcp_preterm_score_labeled.csv" # Path to labels metadata CSV

echo "=================================================================="
echo "Starting 5-Fold Cross Validation Pipeline"
echo "  Weights:   ${WEIGHTS}"
echo "  Dataset:   ${DATA_DIR}"
echo "  Metadata:  ${LABEL_CSV}"
echo "  Output:    ${RUN_DIR}"
echo "=================================================================="

python scripts/train_cv.py \
    --save_dir "${RUN_DIR}" \
    --weights_path "${WEIGHTS}" \
    --data_dir "${DATA_DIR}" \
    --label_csv "${LABEL_CSV}" \
    --epochs 150 \
    --batch_size 32 \
    --folds 5 \
    --input_size 128 \
    2>&1 | tee "${RUN_DIR}/training.log"
