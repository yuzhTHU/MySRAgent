conda activate ./venv

python run_sr_agent.py \
    -f "y = sin(x1 - x2)" \
    --x-low -10 \
    --x-high 10 \
    --llm-provider openrouter \
    --llm-model qwen/qwen3.5-flash-02-23
