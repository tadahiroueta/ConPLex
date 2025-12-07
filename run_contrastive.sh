#!/bin/bash

# Activate virtual environment
source ./venv/bin/activate

echo "Starting Contrastive Learning Experiments (DUDe + Davis)..."
echo "WARNING: The first run will take ~3 hours to preprocess the DUDe dataset."
echo "Subsequent runs will be faster."

# 1. Baseline Model (SimpleCoembedding)
echo "----------------------------------------------------------------"
echo "Running Baseline (SimpleCoembedding) with Contrastive Learning..."
conplex-dti train \
    --run-id baseline_contrastive \
    --config config/default_config.yaml \
    --task davis \
    --epochs 50 \
    --batch-size 32 \
    --model-architecture SimpleCoembedding \
    --contrastive True

# 2. Residual Model (ResidualCoembedding)
echo "----------------------------------------------------------------"
echo "Running Residual (ResidualCoembedding) with Contrastive Learning..."
conplex-dti train \
    --run-id residual_contrastive \
    --config config/default_config.yaml \
    --task davis \
    --epochs 50 \
    --batch-size 32 \
    --model-architecture ResidualCoembedding \
    --contrastive True \
    --num-blocks 2

# 3. Deep MLP Model (DeepCoembedding)
echo "----------------------------------------------------------------"
echo "Running Deep MLP (DeepCoembedding) with Contrastive Learning..."
conplex-dti train \
    --run-id deep_contrastive \
    --config config/default_config.yaml \
    --task davis \
    --epochs 50 \
    --batch-size 32 \
    --model-architecture DeepCoembedding \
    --contrastive True \
    --num-layers 3 \
    --dropout 0.1

# 4. Cross-Attention Model (CrossAttentionCoembedding)
echo "----------------------------------------------------------------"
echo "Running Cross-Attention (CrossAttentionCoembedding) with Contrastive Learning..."
conplex-dti train \
    --run-id crossattn_contrastive \
    --config config/default_config.yaml \
    --task davis \
    --epochs 50 \
    --batch-size 32 \
    --model-architecture CrossAttentionCoembedding \
    --contrastive True \
    --num-heads 8 \
    --lr 0.00001

echo "----------------------------------------------------------------"
echo "All experiments completed!"
