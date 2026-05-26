#!/bin/bash
# From-scratch property model with SymPy labels + SRBench mixing + 4-class encoding
set -e
cd /data6/huanghao/regression_0423/SR_Agent/MySRAgent

SCRATCH_DIR="logs/nn_tools/train_property/scratch"
mkdir -p "$SCRATCH_DIR"

echo "Launching from-scratch training..."
conda run -n sragent python scripts/nn_tools/train_property.py \
    --mode scratch \
    --save_path "$SCRATCH_DIR" \
    --device cuda:0 \
    --seed 42 \
    --max_steps 8000 \
    --batch_size 64 \
    --eval_every 200 \
    --eval_size 128 \
    --eval_batch_size 64 \
    --patience 20 \
    --sample_num 200 \
    --max_var_num 5 \
    --min_depth 1 \
    --max_depth 4 \
    --d_model 128 \
    --nhead 8 \
    --num_encoder_layers 4 \
    --dim_feedforward 512 \
    --lr 3e-4 \
    --max_per_signature 9999 \
    --srbench_mix_ratio 0.2 \
    2>&1 | tee "$SCRATCH_DIR/console.log"

echo "Evaluating from-scratch model..."
conda run -n sragent python scripts/nn_tools/eval_property.py \
    --checkpoint "$SCRATCH_DIR/best.pth" \
    --device cuda:0 \
    --n_test 500 \
    --sample_num 200 \
    2>&1 | tee "$SCRATCH_DIR/eval_console.log"

echo "=== Experiment 1 (scratch) done ==="
