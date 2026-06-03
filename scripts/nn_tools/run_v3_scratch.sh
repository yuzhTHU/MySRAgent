#!/bin/bash
# Property prediction v3 — From Scratch training + evaluation
# Architecture: per-variable cross-attention heads, deeper MLP, GELU, LayerNorm
# Training: OneCycleLR, Focal Loss, data augmentation, SRBench 40% mix
set -e
cd /data6/huanghao/regression_0423/SR_Agent/MySRAgent

SCRATCH_DIR="logs/nn_tools/train_property_v3/scratch"
mkdir -p "$SCRATCH_DIR"

echo "[$(date)] Launching v3 from-scratch training on cuda:0..."
conda run -n sragent python scripts/nn_tools/train_property.py \
    --mode scratch \
    --save_path "$SCRATCH_DIR" \
    --device cuda:0 \
    --seed 42 \
    --max_steps 10000 \
    --batch_size 64 \
    --eval_every 200 \
    --eval_size 256 \
    --eval_batch_size 64 \
    --patience 25 \
    --sample_num 200 \
    --max_var_num 5 \
    --min_depth 1 \
    --max_depth 5 \
    --d_model 128 \
    --nhead 8 \
    --num_encoder_layers 4 \
    --dim_feedforward 512 \
    --dropout 0.2 \
    --lr 3e-4 \
    --max_per_signature 9999 \
    --srbench_mix_ratio 0.4 \
    --use_focal_loss \
    --label_smoothing 0.05 \
    --noise_std 0.01 \
    --scale_augment \
    --permute_vars \
    2>&1 | tee "$SCRATCH_DIR/console.log"

echo "[$(date)] Evaluating v3 from-scratch model..."
conda run -n sragent python scripts/nn_tools/eval_property.py \
    --checkpoint "$SCRATCH_DIR/best.pth" \
    --device cuda:0 \
    --n_test 500 \
    --sample_num 200 \
    2>&1 | tee "$SCRATCH_DIR/eval_console.log"

echo "[$(date)] === v3 scratch experiment done ==="
