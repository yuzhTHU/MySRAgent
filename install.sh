# 将 SRAgent 代码下载到本地
# git clone git@github.com:yuzhTHU/MySRAgent.git ./SRAgent && cd SRAgent

# 创建环境
conda create -p ./venv python=3.12 -y && conda activate ./venv

# nd2py 库尚不稳定，建议以可编辑方式单独安装
git clone git@github.com:yuzhTHU/nd2py.git ./src/nd2py_package
pip install -e ./src/nd2py_package

# 安装其它依赖
pip install -e ".[dev]"

# （可选）下载 LLMSR-Bench 数据到本地
# git clone git@hf.co:datasets/nnheui/llm-srbench ./data/llm-srbench-data

# （可选）下载其它代码以供参考
# git clone git@github.com:GAIR-NLP/SR-Scientist.git ./src/sr_scientist
# git clone git@github.com:deep-symbolic-mathematics/llm-srbench.git ./data/llm-srbench-code
