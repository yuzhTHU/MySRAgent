#!/bin/bash
set -e
cd /data6/huanghao/regression_0423/SR_Agent/MySRAgent

COMMON="--llm-provider openrouter --llm-model qwen/qwen3.5-flash-02-23 -R 2 -C 1 -L 10 -K 2 --seed 42 --no-skip-existing --debug"
TOOLS="--tools statistics_analysis evaluate_formula submit_formula polynomial_fit code_executor call_sindy call_pysr"
EXP="08_sindy_pysr"

echo "[$(date)] START sindy+pysr group (E, F, G only)"

echo "[$(date)] E: I.30.3_0_0"
conda run -n sragent python bench_sr_agent.py --algorithm my_sr_agent --datasets lsrtransform --problem_names I.30.3_0_0 $TOOLS $COMMON --exp_name "$EXP" 2>&1 | tail -3

echo "[$(date)] F: I.50.26_3_0"
conda run -n sragent python bench_sr_agent.py --algorithm my_sr_agent --datasets lsrtransform --problem_names I.50.26_3_0 $TOOLS $COMMON --exp_name "$EXP" 2>&1 | tail -3

echo "[$(date)] G: III.15.12_0_0"
conda run -n sragent python bench_sr_agent.py --algorithm my_sr_agent --datasets lsrtransform --problem_names III.15.12_0_0 $TOOLS $COMMON --exp_name "$EXP" 2>&1 | tail -3

echo "[$(date)] ALL sindy+pysr DONE"
