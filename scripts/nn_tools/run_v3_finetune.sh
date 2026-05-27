#!/bin/bash
# Property prediction v3 — Pretrain + Finetune with gradual unfreezing
# Uses existing pretrained FoundationModel encoder, then finetunes with
# discriminative LR and gradual layer unfreezing.
set -e
cd /data6/huanghao/regression_0423/SR_Agent/MySRAgent

PRETRAIN_CKPT="logs/nn_tools/train/pretrain_for_prop/checkpoints/epoch_100000.pth"
FINETUNE_DIR="logs/nn_tools/train_property_v3/finetune"
mkdir -p "$FINETUNE_DIR"

if [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "ERROR: Pretrain checkpoint not found: $PRETRAIN_CKPT"
    echo "Please run the FoundationModel pretraining first."
    exit 1
fi
echo "Using pretrain checkpoint: $PRETRAIN_CKPT"

echo "[$(date)] Launching v3 finetune training on cuda:1..."
conda run -n sragent python scripts/nn_tools/train_property.py \
    --mode finetune \
    --pretrain_checkpoint "$PRETRAIN_CKPT" \
    --freeze_steps 2000 \
    --save_path "$FINETUNE_DIR" \
    --device cuda:1 \
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
    2>&1 | tee "$FINETUNE_DIR/console.log"

echo "[$(date)] Evaluating v3 finetuned model..."
conda run -n sragent python scripts/nn_tools/eval_property.py \
    --checkpoint "$FINETUNE_DIR/best.pth" \
    --device cuda:1 \
    --n_test 500 \
    --sample_num 200 \
    2>&1 | tee "$FINETUNE_DIR/eval_console.log"

echo "[$(date)] === v3 finetune experiment done ==="
