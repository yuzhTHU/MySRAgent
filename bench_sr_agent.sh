# conda activate ./venv

python bench_sr_agent.py \
    --problem_names MatSci2 MatSci19 CRK28 BPG1 PO6 \
    --exp_name guanren_tests \
    --no-skip-existing \
    -L 20 -K 2 -R 5 -C 2 \
    --llm-provider openrouter \
    --llm-model qwen/qwen3.5-flash-02-23


python bench_sr_agent.py \
    --problem_names MatSci2 MatSci19 CRK28 BPG1 PO6 \
    --name guanren_tests_gpt-5.5 \
    --no-skip-existing \
    -L 20 -K 2 -R 1 -C 2 \
    --llm-provider openrouter \
    --llm-model openai/gpt-5.5

python bench_sr_agent.py \
    --problem_names MatSci2 MatSci19 CRK28 BPG1 PO6 \
    --name guanren_tests_qwen3.6-plus \
    --no-skip-existing \
    -L 20 -K 2 -R 1 -C 2 \
    --llm-provider openrouter \
    --llm-model qwen/qwen3.6-plus

python bench_sr_agent.py \
    --problem_names MatSci2 MatSci19 CRK28 BPG1 PO6 \
    --name guanren_tests_deepseek-v4-pro \
    --no-skip-existing \
    -L 20 -K 2 -R 1 -C 2 \
    --llm-provider openrouter \
    --llm-model deepseek/deepseek-v4-pro

python bench_sr_agent.py \
    --dataset lsrtransform \
    --exp_name bench_my_sr_agent \
    -R 2 -C 2 -L 5 -K 2 \
    --llm_provider openrouter \
    --llm_model "deepseek/deepseek-v4-flash"

python bench_sr_agent.py \
    --dataset lsrtransform \
    --exp_name bench_my_sr_agent_deepseek-v4-pro \
    -R 2 -C 2 -L 5 -K 2 \
    --llm_provider openrouter \
    --llm_model "deepseek/deepseek-v4-pro"

python bench_sr_agent.py \
    --algorithm pysr \
    --dataset lsrtransform \
    --exp_name bench_pysr
