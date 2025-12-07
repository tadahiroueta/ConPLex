#!/bin/bash

# Run CrossAttentionCoembedding on BIOSNAP
# Using 50 epochs, batch size 32, and default hyperparameters for attention
# We enable early stopping and LR scheduling as they are generally beneficial

conplex-dti train \
    --run-id biosnap_cross_attention_v1 \
    --config ConPLex/config/default_config.yaml \
    --task biosnap \
    --epochs 50 \
    --batch-size 32 \
    --model-architecture CrossAttentionCoembedding \
    --contrastive True \
    --lr 1e-4 \
    --dropout 0.1 \
    --scheduler plateau \
    --patience 10 \
    --num-heads 4 \
    --data-cache-dir ConPLex/datasets
