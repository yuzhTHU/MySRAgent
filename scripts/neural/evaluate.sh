python ./scripts/neural/evaluate.py --algorithm linear --stimulus Awake --max_rollout_steps 30
python ./scripts/neural/evaluate.py --algorithm linear --stimulus NREM --max_rollout_steps 30
python ./scripts/neural/evaluate.py --algorithm linear --stimulus REM --max_rollout_steps 30

python ./scripts/neural/evaluate.py --algorithm mlp --stimulus Awake --max_rollout_steps 30
python ./scripts/neural/evaluate.py --algorithm mlp --stimulus NREM --max_rollout_steps 30
python ./scripts/neural/evaluate.py --algorithm mlp --stimulus REM --max_rollout_steps 30