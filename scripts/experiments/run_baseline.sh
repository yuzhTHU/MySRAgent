#!/bin/bash
set -e
cd /data6/huanghao/regression_0423/SR_Agent/MySRAgent

COMMON="--llm-provider openrouter --llm-model qwen/qwen3.5-flash-02-23 -R 2 -C 1 -L 10 -K 2 --seed 42 --no-skip-existing --debug"
TOOLS="--tools statistics_analysis evaluate_formula submit_formula polynomial_fit code_executor"
EXP="prop_tool_compare_baseline"

echo "[$(date)] START BASELINE"

echo "[$(date)] A: sin(x1-x2)"
conda run -n sragent python run_sr_agent.py -f "y = sin(x1 - x2)" --x-low -10 --x-high 10 $TOOLS $COMMON --exp_name "${EXP}_sinx1x2" 2>&1 | tail -5

echo "[$(date)] B: II.24.17_0_1"
conda run -n sragent python bench_sr_agent.py --algorithm my_sr_agent --datasets lsrtransform --problem_names II.24.17_0_1 $TOOLS $COMMON --exp_name "$EXP" 2>&1 | tail -5

echo "[$(date)] C: II.13.23_1_0"
conda run -n sragent python bench_sr_agent.py --algorithm my_sr_agent --datasets lsrtransform --problem_names II.13.23_1_0 $TOOLS $COMMON --exp_name "$EXP" 2>&1 | tail -5

echo "[$(date)] D: I.34.14_2_0"
conda run -n sragent python bench_sr_agent.py --algorithm my_sr_agent --datasets lsrtransform --problem_names I.34.14_2_0 $TOOLS $COMMON --exp_name "$EXP" 2>&1 | tail -5

echo "[$(date)] E: I.30.3_0_0"
conda run -n sragent python bench_sr_agent.py --algorithm my_sr_agent --datasets lsrtransform --problem_names I.30.3_0_0 $TOOLS $COMMON --exp_name "$EXP" 2>&1 | tail -5

echo "[$(date)] F: I.50.26_3_0"
conda run -n sragent python bench_sr_agent.py --algorithm my_sr_agent --datasets lsrtransform --problem_names I.50.26_3_0 $TOOLS $COMMON --exp_name "$EXP" 2>&1 | tail -5

echo "[$(date)] G: III.15.12_0_0"
conda run -n sragent python bench_sr_agent.py --algorithm my_sr_agent --datasets lsrtransform --problem_names III.15.12_0_0 $TOOLS $COMMON --exp_name "$EXP" 2>&1 | tail -5

echo "[$(date)] ALL BASELINE DONE"
