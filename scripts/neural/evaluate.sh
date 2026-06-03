python ./scripts/neural/evaluate.py --algorithm linear --stimulus Awake --max_rollout_steps 30
python ./scripts/neural/evaluate.py --algorithm linear --stimulus NREM --max_rollout_steps 30
python ./scripts/neural/evaluate.py --algorithm linear --stimulus REM --max_rollout_steps 30

python ./scripts/neural/evaluate.py --algorithm mlp --stimulus Awake --max_rollout_steps 30
python ./scripts/neural/evaluate.py --algorithm mlp --stimulus NREM --max_rollout_steps 30
python ./scripts/neural/evaluate.py --algorithm mlp --stimulus REM --max_rollout_steps 30


python ./scripts/neural/evaluate.py --algorithm equation --stimulus Awake --max_rollout_steps 30 --equation "dx = 1.0 * x_0 + 0.0 * x_1 + 0.0 * x_2" --hist_steps 3
python ./scripts/neural/evaluate.py --algorithm equation --stimulus NREM --max_rollout_steps 30 --equation "dx = 1.0 * x_0 + 0.0 * x_1 + 0.0 * x_2" --hist_steps 3
python ./scripts/neural/evaluate.py --algorithm equation --stimulus REM --max_rollout_steps 30 --equation "dx = 1.0 * x_0 + 0.0 * x_1 + 0.0 * x_2" --hist_steps 3