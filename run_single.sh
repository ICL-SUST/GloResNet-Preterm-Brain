#!/bin/bash
# ------------------------------------------------------------------
# Single Model Training Runner: Infant Brain Injury Prediction
# ------------------------------------------------------------------

export PYTHONPATH=$PYTHONPATH:.

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="runs/single_resnet50_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

# --- Parameters (Change these to match your environment) ---
WEIGHTS="weights/resnet_50_23dataset.pth"     # Path to pretrained MedicalNet weights
DATA_DIR="/mnt/raid5/dhcp"                    # Path to BIDS dataset directory
LABEL_CSV="data/dhcp_preterm_score_labeled.csv" # Path to labels metadata CSV

echo "=================================================================="
echo "Starting Single Model Training Pipeline"
echo "  Weights:   ${WEIGHTS}"
echo "  Dataset:   ${DATA_DIR}"
echo "  Metadata:  ${LABEL_CSV}"
echo "  Output:    ${RUN_DIR}"
echo "=================================================================="

python scripts/train_single.py \
    --save_dir "${RUN_DIR}" \
    --weights_path "${WEIGHTS}" \
    --data_dir "${DATA_DIR}" \
    --label_csv "${LABEL_CSV}" \
    --epochs 100 \
    --batch_size 16 \
    --model "resnet50" \
    --input_size 112 \
    2>&1 | tee "${RUN_DIR}/training.log"
