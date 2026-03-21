import numpy as np
import sys
import os
from sr_agent.utils import init_logger

init_logger('sr_agent', exp_name='Test')

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sr_agent import SRAgent

if __name__ == '__main__':
    # 创建一个简单的符号回归问题：y = sin(phi0 - phi1)
    np.random.seed(42)
    x = np.random.rand(100, 2)
    y = np.sin(x[:, 0] - x[:, 1])
    X = {'phi0': x[:, 0], 'phi1': x[:, 1]}

    # 初始化 Agent
    # 使用 siliconflow 作为 LLM 提供商，可以使用其免费的模型
    agent = SRAgent(
        llm_provider='siliconflow',
        llm_model='Qwen3-8B',
        max_iteration=20,
        verbose=True,
    )

    # 执行符号回归任务
    result = agent.fit(
        X=X,
        y=y,
        problem_description="Find the relationship between l = f(phi0, phi1)",
    )

    print("\n" + "=" * 50)
    print("Symbolic Regression Result")
    print("=" * 50)
    print(f"Best Formula: {result['best_formula']}")
    print(f"Best Score (MSE): {result['best_score']}")
    print(f"Total Iterations: {result['iterations']}")
    print("=" * 50)
