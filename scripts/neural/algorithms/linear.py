# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""【样例代码】
不考虑节点间交互的一阶线性模型：
    (d/dt) x_i(t) = a0 * x_i(t) + a1 * x_i(t-1) + ... + aN * x_i(t-N)
其中 N 是历史步数, 由 --hist_steps 参数控制
所有节点和时间步共享同一组参数 {a0, a1, ..., aN}
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path('../../..').absolute()
sys.path.insert(0, str(ROOT / "src" / "experimental"))

import numpy as np
from sklearn.linear_model import Ridge


def update_parser(parser):
    """ 当前算法添加的命令行参数 """
    parser.add_argument("--linear_alpha", type=float, default=1.0, help="Ridge alpha.")
    parser.set_defaults(hist_steps=1) # 模型默认只使用前一步数据预测下一步
    return parser


def train_model(args, data, node_info, time_info):
    """ 给定训练数据, 训练模型并返回
    Args:
        data: np.ndarray, shape (n_times, n_nodes), 训练数据矩阵
        node_info: pd.DataFrame, 包含节点特征的 DataFrame
            (index): 节点编号
            (acs, names, ids): 节点所在脑区 (名称, 描述, 编号)
            (coords_2d_x, coords_2d_y): 节点所在空间位置
        time_info: pd.DataFrame, 包含时间特征的 DataFrame
            (index): 时间步编号 (注意: 虽然能保证单调递增，但是可能不连续！)
            (stimulus): 时间步对应的刺激状态 (Awake, REM, NREM)
            (time): 时间步对应的实际时间 (单位秒, 采样率 4Hz, 因此每个时间步间隔 0.25 秒)
    """
    hist_steps = args.hist_steps
    x = []
    dx_dt = []
    for t in range(hist_steps - 1, len(time_info)-1):
        # 检查时间是否连续
        if np.all(np.diff(time_info.index[t-hist_steps+1:t+2]) == 1):
            x.append(data[t-hist_steps+1:t+1].T)
            dx_dt.append(data[t+1] - data[t])
    x = np.concatenate(x, axis=0)
    dx_dt = np.concatenate(dx_dt, axis=0)
    model = Ridge(alpha=args.linear_alpha)
    model.fit(x, dx_dt)
    return {"model": model, "hist_steps": hist_steps, "train_rows": int(len(dx_dt))}


def get_predict_func(args, result_dict):
    """ 给定训练好的模型, 返回一个预测函数 predict_func, 根据历史数据预测未来数据 """
    model = result_dict["model"]

    def predict_func(data, node_info, time_info):
        """ 根据历史数据预测未来数据
        Args:
            同 train_model 函数中的参数
        Returns:
            pred: np.ndarray, shape (n_nodes), 预测的下一步数据矩阵
        """
        x = np.asarray(data[-args.hist_steps:, :], dtype=np.float64)
        dx_dt = model.predict(x.T).squeeze()
        return x[-1] + dx_dt

    return predict_func


def format_result(args, result_dict) -> str:
    """ 将训练结果格式化为字符串, 以便打印给用户看 """
    model = result_dict["model"]
    formula = "dx = "
    for i, coef in enumerate(model.coef_):
        formula += f"{coef:.6f} * x(t-{result_dict['hist_steps']-i}) + "
    formula = formula.rstrip(" + ")
    return f"Formula: {formula}"
