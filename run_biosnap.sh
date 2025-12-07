#!/bin/bash

# Activate virtual environment
source ./venv/bin/activate

echo "Starting Contrastive Learning Experiments (DUDe + BIOSNAP)..."
echo "Note: DUDe preprocessing should be cached if you ran the Davis script first."

# 1. Baseline Model (SimpleCoembedding)
echo "----------------------------------------------------------------"
echo "Running Baseline (SimpleCoembedding) on BIOSNAP..."
conplex-dti train \
    --run-id biosnap_baseline_contrastive \
    --config config/default_config.yaml \
    --task biosnap \
    --epochs 15 \
    --batch-size 32 \
    --model-architecture SimpleCoembedding \
    --contrastive True

# 2. Residual Model (ResidualCoembedding)
echo "----------------------------------------------------------------"
echo "Running Residual (ResidualCoembedding) on BIOSNAP..."
conplex-dti train \
    --run-id biosnap_residual_contrastive \
    --config config/default_config.yaml \
    --task biosnap \
    --epochs 15 \
    --batch-size 32 \
    --model-architecture ResidualCoembedding \
    --contrastive True \
    --num-blocks 2

# 3. Deep MLP Model (DeepCoembedding)
echo "----------------------------------------------------------------"
echo "Running Deep MLP (DeepCoembedding) on BIOSNAP..."
conplex-dti train \
    --run-id biosnap_deep_contrastive \
    --config config/default_config.yaml \
    --task biosnap \
    --epochs 15 \
    --batch-size 32 \
    --model-architecture DeepCoembedding \
    --contrastive True \
    --num-layers 3 \
    --dropout 0.1

# 4. Cross-Attention Model (CrossAttentionCoembedding)
echo "----------------------------------------------------------------"
echo "Running Cross-Attention (CrossAttentionCoembedding) on BIOSNAP..."
conplex-dti train \
    --run-id biosnap_crossattn_contrastive \
    --config config/default_config.yaml \
    --task biosnap \
    --epochs 15 \
    --batch-size 32 \
    --model-architecture CrossAttentionCoembedding \
    --contrastive True \
    --num-heads 8 \
    --lr 0.00001

echo "----------------------------------------------------------------"
echo "All BIOSNAP experiments completed!"
