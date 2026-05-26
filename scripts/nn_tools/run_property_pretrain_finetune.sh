#!/bin/bash
# Experiment 2: Pretrain FoundationModel then finetune property heads.
set -e
cd /data6/huanghao/regression_0423/SR_Agent/MySRAgent

PRETRAIN_EXP="pretrain_for_prop"
PRETRAIN_DIR="logs/nn_tools/train/${PRETRAIN_EXP}"
FINETUNE_DIR="logs/nn_tools/train_property/finetune"
mkdir -p "$FINETUNE_DIR"

# ── Step 1: Pretrain FoundationModel (data -> formula) ───────────────────────
echo "Step 1: Pretraining FoundationModel..."
conda run -n sragent python scripts/nn_tools/train.py \
    --exp_name "$PRETRAIN_EXP" \
    --device cuda:1 \
    --seed 42 \
    --batch_size 16 \
    --eval_every 500 \
    --eval_size 128 \
    --patience 20 \
    --sample_num 100 \
    --max_var_num 5 \
    --min_depth 2 \
    --max_depth 5 \
    --d_model 128 \
    --nhead 8 \
    --num_encoder_layers 4 \
    --num_decoder_layers 4 \
    --dim_feedforward 512 \
    --lr 1e-4 \
    --output_pooling attention \
    --data_pooling attention \
    --eq_generator gplearn \
    --data_generator uniform \
    --data_min -10 \
    --data_max 10 \
    2>&1

echo "Step 1 done."
CKPT="$PRETRAIN_DIR/best.pth"
if [ ! -f "$CKPT" ]; then
    CKPT="$PRETRAIN_DIR/checkpoint.pth"
fi
echo "Using checkpoint: $CKPT"

# ── Step 2: Finetune property heads ──────────────────────────────────────────
echo "Step 2: Finetuning property model from pretrained encoder..."
conda run -n sragent python scripts/nn_tools/train_property.py \
    --mode finetune \
    --pretrain_checkpoint "$CKPT" \
    --freeze_steps 1000 \
    --save_path "$FINETUNE_DIR" \
    --device cuda:1 \
    --seed 42 \
    --max_steps 5000 \
    --batch_size 64 \
    --eval_every 200 \
    --eval_size 128 \
    --eval_batch_size 64 \
    --patience 15 \
    --sample_num 100 \
    --max_var_num 5 \
    --min_depth 1 \
    --max_depth 4 \
    --d_model 128 \
    --nhead 8 \
    --num_encoder_layers 4 \
    --dim_feedforward 512 \
    --lr 3e-4 \
    --max_per_signature 9999 \
    2>&1 | tee "$FINETUNE_DIR/console.log"

echo "Evaluating finetuned model..."
conda run -n sragent python scripts/nn_tools/eval_property.py \
    --checkpoint "$FINETUNE_DIR/best.pth" \
    --device cuda:1 \
    --n_test 500 \
    --sample_num 100 \
    2>&1 | tee "$FINETUNE_DIR/eval_console.log"

echo "=== Experiment 2 (pretrain+finetune) done ==="
