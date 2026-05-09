conda activate ./venv

python bench_sr_agent.py \
    --problem_names MatSci2 MatSci19 CRK28 BPG1 PO6 \
    --exp_name guanren_tests \
    --no-skip-existing \
    -L 20 -K 2 -R 5 -C 2 \
    --llm-provider openrouter \
    --llm-model qwen/qwen3.5-flash-02-23
