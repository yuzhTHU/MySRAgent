conda activate ./venv

python run_sr_agent.py \
    -f "y = sin(x1 - x2)" \
    --x-low -10 \
    --x-high 10 \
    --llm-provider openrouter \
    --llm-model qwen/qwen3.5-flash-02-23 \
    -R 1 -C 1 -L 3 -K 1


python run_sr_agent.py \
    -f "y = sin(x1 - x2)" \
    --x-low -10 \
    --x-high 10 \
    --llm-provider openrouter \
    --llm-model qwen/qwen3.5-flash-02-23 \
    -R 2 -C 1 -L 10 -K 2


# 对比分析模型性能，参数 R / L 对实验结果的影响
python run_sr_agent.py \
    -f "y = sin(x1 - x2)" \
    --x-low -10 \
    --x-high 10 \
    --llm-provider openrouter \
    --llm-model anthropic/claude-opus-4.7 \
    -R 1 -C 1 -L 3 -K 1 \
    --seed 42 \
    --exp_name "claude_opus_R1C1L3K1"


python run_sr_agent.py \
    -f "y = sin(x1 - x2)" \
    --x-low -10 \
    --x-high 10 \
    --llm-provider openrouter \
    --llm-model qwen/qwen3.5-flash-02-23 \
    -R 2 -C 1 -L 10 -K 2 \
    --seed 42 \
    --exp_name "qwen_flash_R2C1L10K2"