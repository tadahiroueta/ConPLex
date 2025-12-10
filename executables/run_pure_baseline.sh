#!/bin/bash

# Activate virtual environment
source ./venv/bin/activate

echo "Starting Pure Baseline Training (No Improvements)..."

# Pure Baseline Model (SimpleCoembedding)
# - No Early Stopping (patience=100)
# - No Scheduler (scheduler=cosine, effectively constant if not tuned, or default)
# - No Contrastive Learning (to keep it simple and fast as per previous baseline)
# - No Residual/Dropout/Alpha (SimpleCoembedding architecture)

echo "----------------------------------------------------------------"
echo "Running Pure Baseline (SimpleCoembedding) on BIOSNAP..."
conplex-dti train \
    --run-id biosnap_baseline_pure \
    --config config/default_config.yaml \
    --task biosnap \
    --epochs 50 \
    --batch-size 32 \
    --model-architecture SimpleCoembedding \
    --patience 100 \
    --scheduler cosine \
    --lr 0.0001
