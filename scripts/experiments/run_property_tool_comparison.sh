#!/bin/bash
# Property Predictor Tool Comparison Experiments
# Compare search performance WITH and WITHOUT predict_property tool
set -e
cd /data6/huanghao/regression_0423/SR_Agent/MySRAgent

COMMON_ARGS="--llm-provider openrouter --llm-model qwen/qwen3.5-flash-02-23 -R 2 -C 1 -L 10 -K 2 --seed 42 --no-skip-existing --debug"
BASELINE_TOOLS="--tools statistics_analysis evaluate_formula submit_formula polynomial_fit code_executor"
ALL_TOOLS="--tools statistics_analysis evaluate_formula submit_formula polynomial_fit code_executor predict_property"

# ==============================
# Group 1: WITHOUT predict_property (baseline)
# ==============================
EXP_NAME="prop_tool_compare_baseline"

echo "=== BASELINE: sin(x1-x2) ==="
conda run -n sragent python run_sr_agent.py \
    -f "y = sin(x1 - x2)" --x-low -10 --x-high 10 \
    $BASELINE_TOOLS $COMMON_ARGS \
    --exp_name "${EXP_NAME}_sinx1x2"

echo "=== BASELINE: II.24.17_0_1 (hard B) ==="
conda run -n sragent python bench_sr_agent.py \
    --algorithm my_sr_agent --datasets lsrtransform \
    --problem_names II.24.17_0_1 \
    $BASELINE_TOOLS $COMMON_ARGS \
    --exp_name "${EXP_NAME}"

echo "=== BASELINE: II.13.23_1_0 (hard C) ==="
conda run -n sragent python bench_sr_agent.py \
    --algorithm my_sr_agent --datasets lsrtransform \
    --problem_names II.13.23_1_0 \
    $BASELINE_TOOLS $COMMON_ARGS \
    --exp_name "${EXP_NAME}"

echo "=== BASELINE: I.34.14_2_0 (hard D) ==="
conda run -n sragent python bench_sr_agent.py \
    --algorithm my_sr_agent --datasets lsrtransform \
    --problem_names I.34.14_2_0 \
    $BASELINE_TOOLS $COMMON_ARGS \
    --exp_name "${EXP_NAME}"

echo "=== BASELINE: I.30.3_0_0 (trig E) ==="
conda run -n sragent python bench_sr_agent.py \
    --algorithm my_sr_agent --datasets lsrtransform \
    --problem_names I.30.3_0_0 \
    $BASELINE_TOOLS $COMMON_ARGS \
    --exp_name "${EXP_NAME}"

echo "=== BASELINE: I.50.26_3_0 (trig F) ==="
conda run -n sragent python bench_sr_agent.py \
    --algorithm my_sr_agent --datasets lsrtransform \
    --problem_names I.50.26_3_0 \
    $BASELINE_TOOLS $COMMON_ARGS \
    --exp_name "${EXP_NAME}"

echo "=== BASELINE: III.15.12_0_0 (trig G) ==="
conda run -n sragent python bench_sr_agent.py \
    --algorithm my_sr_agent --datasets lsrtransform \
    --problem_names III.15.12_0_0 \
    $BASELINE_TOOLS $COMMON_ARGS \
    --exp_name "${EXP_NAME}"

echo "=== All BASELINE experiments done ==="

# ==============================
# Group 2: WITH predict_property (all tools)
# ==============================
EXP_NAME="prop_tool_compare_with_prop"

echo "=== WITH PROP: sin(x1-x2) ==="
conda run -n sragent python run_sr_agent.py \
    -f "y = sin(x1 - x2)" --x-low -10 --x-high 10 \
    $ALL_TOOLS $COMMON_ARGS \
    --exp_name "${EXP_NAME}_sinx1x2"

echo "=== WITH PROP: II.24.17_0_1 (hard B) ==="
conda run -n sragent python bench_sr_agent.py \
    --algorithm my_sr_agent --datasets lsrtransform \
    --problem_names II.24.17_0_1 \
    $ALL_TOOLS $COMMON_ARGS \
    --exp_name "${EXP_NAME}"

echo "=== WITH PROP: II.13.23_1_0 (hard C) ==="
conda run -n sragent python bench_sr_agent.py \
    --algorithm my_sr_agent --datasets lsrtransform \
    --problem_names II.13.23_1_0 \
    $ALL_TOOLS $COMMON_ARGS \
    --exp_name "${EXP_NAME}"

echo "=== WITH PROP: I.34.14_2_0 (hard D) ==="
conda run -n sragent python bench_sr_agent.py \
    --algorithm my_sr_agent --datasets lsrtransform \
    --problem_names I.34.14_2_0 \
    $ALL_TOOLS $COMMON_ARGS \
    --exp_name "${EXP_NAME}"

echo "=== WITH PROP: I.30.3_0_0 (trig E) ==="
conda run -n sragent python bench_sr_agent.py \
    --algorithm my_sr_agent --datasets lsrtransform \
    --problem_names I.30.3_0_0 \
    $ALL_TOOLS $COMMON_ARGS \
    --exp_name "${EXP_NAME}"

echo "=== WITH PROP: I.50.26_3_0 (trig F) ==="
conda run -n sragent python bench_sr_agent.py \
    --algorithm my_sr_agent --datasets lsrtransform \
    --problem_names I.50.26_3_0 \
    $ALL_TOOLS $COMMON_ARGS \
    --exp_name "${EXP_NAME}"

echo "=== WITH PROP: III.15.12_0_0 (trig G) ==="
conda run -n sragent python bench_sr_agent.py \
    --algorithm my_sr_agent --datasets lsrtransform \
    --problem_names III.15.12_0_0 \
    $ALL_TOOLS $COMMON_ARGS \
    --exp_name "${EXP_NAME}"

echo "=== All WITH PROP experiments done ==="
echo "=== FULL COMPARISON COMPLETE ==="
