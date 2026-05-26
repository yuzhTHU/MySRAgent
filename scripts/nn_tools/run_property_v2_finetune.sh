#!/bin/bash
# v2: Finetune property model from pretrained encoder + SymPy labels + SRBench mixing
set -e
cd /data6/huanghao/regression_0423/SR_Agent/MySRAgent

PRETRAIN_CKPT="logs/nn_tools/train/pretrain_for_prop/checkpoints/epoch_100000.pth"
FINETUNE_DIR="logs/nn_tools/train_property_v2/finetune"
mkdir -p "$FINETUNE_DIR"

if [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "ERROR: Pretrained checkpoint not found: $PRETRAIN_CKPT"
    exit 1
fi
echo "Using pretrained checkpoint: $PRETRAIN_CKPT"

echo "Launching v2 finetune training..."
conda run -n sragent python scripts/nn_tools/train_property.py \
    --mode finetune \
    --pretrain_checkpoint "$PRETRAIN_CKPT" \
    --freeze_steps 1000 \
    --save_path "$FINETUNE_DIR" \
    --device cuda:1 \
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
    2>&1 | tee "$FINETUNE_DIR/console.log"

echo "Evaluating v2 finetuned model..."
conda run -n sragent python scripts/nn_tools/eval_property.py \
    --checkpoint "$FINETUNE_DIR/best.pth" \
    --device cuda:1 \
    --n_test 500 \
    --sample_num 200 \
    2>&1 | tee "$FINETUNE_DIR/eval_console.log"

echo "=== v2 Experiment 2 (pretrain+finetune) done ==="
