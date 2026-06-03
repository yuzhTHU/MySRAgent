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
    --exp_name bench_my_sr_agent-v4-flash \
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


python bench_sr_agent.py \
    --dataset lsrtransform \
    --exp_name bench_my_sr_agent-v4-flash_ban_pysr_sindy \
    -R 2 -C 2 -L 5 -K 2 \
    --llm_provider openrouter \
    --llm_model "deepseek/deepseek-v4-flash" \
    --ban_tools "call_pysr" "call_sindy"

python bench_sr_agent.py \
    --dataset lsrtransform \
    --exp_name bench_my_sr_agent-v4-flash_ban_predict_property \
    -R 2 -C 2 -L 5 -K 2 \
    --llm_provider openrouter \
    --llm_model "deepseek/deepseek-v4-flash" \
    --ban_tools "predict_property"

python bench_sr_agent.py \
    --algorithm sr_scientist \
    --datasets lsrtransform \
    --exp_name bench_sr_scientist-v4-flash \
    --llm_provider openrouter \
    --llm_model "deepseek/deepseek-v4-flash" \
    --sr_scientist_sandbox_urls http://127.0.0.1:8080/run_code \
    --sr_scientist_num_turns 2 \
    --sr_scientist_max_assistant_turns 20 \
    --sr_scientist_top_k 2

python bench_sr_agent.py \
    --dataset lsrtransform \
    --exp_name bench_my_sr_agent_gpt-4o-mini \
    -R 2 -C 2 -L 5 -K 2 \
    --llm_provider openrouter \
    --llm_model "openai/gpt-4o-mini"


python bench_sr_agent.py \
    --algorithm codex \
    --exp-name codex_gpt55 \
    --codex-cmd "npx --yes @openai/codex@latest" \
    --codex-echo-events \
    --skip-successful \
    --ban-tools "ask_human" "workspace_shell"
