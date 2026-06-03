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

import logging
import numpy as np
import nd2py as nd
from collections import defaultdict

_logger = logging.getLogger(f"sr_agent.{__name__}")


def update_parser(parser):
    """ 当前算法添加的命令行参数 """
    parser.add_argument("--linear_alpha", type=float, default=1.0, help="Ridge alpha.")
    parser.add_argument("--equation", type=str, required=True, help="Equation string, e.g., dx = 1.0 * x_0 + 0.0 * x_1")
    parser.add_argument("--fit_params", type=bool, default=True, help="Whether to fit parameters in --equation from data. If False, will directly use the equation without fitting.")
    parser.set_defaults(hist_steps=2) # 与 equation 中的 x_0, x_1 数量保持一致
    return parser


def _build_dataset(data, time_info, hist_steps):
    """ 将原始数据构造为 data_dict
        x_0 -> 当前时间步
        x_1 -> 前一个时间步
        ...
        dx -> 下一时间步与当前时间步的差值
    """
    data_dict = defaultdict(list)
    for t in range(hist_steps - 1, len(time_info) - 1):
        # 检查时间是否连续
        if np.all(np.diff(time_info.index[t - hist_steps + 1 : t + 2]) == 1):
            for n in range(hist_steps):
                data_dict[f'x_{n}'].append(data[t - n])
            data_dict['dx'].append(data[t + 1] - data[t])
    for key, value in data_dict.items():
        data_dict[key] = np.asarray(value, dtype=np.float32)
    return data_dict


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
    data_dict = _build_dataset(data, time_info, hist_steps)
    edge_list = None
    num_nodes = None
    if '=' in args.equation:
        eq_str = args.equation.split('=', 1)[1].strip()
    else:
        eq_str = args.equation.strip()
    variables = {f'x_{n}': nd.Variable(f'x_{n}', nettype='node') for n in range(hist_steps)}
    eq = nd.parse(eq_str, variables=variables)

    for var in eq.iter_preorder():
        if isinstance(var, nd.Variable) and var.name not in variables:
            raise ValueError(
                f"Equation contains undefined variable: {var.name}. "
                f"Since --hist_steps={hist_steps}, only x_0, x_1, ..., x_{hist_steps-1} are allowed."
            )

    if args.fit_params:
        bfgs = nd.BFGSFit(eq, edge_list=edge_list, num_nodes=num_nodes, method='BFGS')
        bfgs.fit(data_dict, data_dict['dx'])
        eq = bfgs.expression

    _logger.note(f"Fitted equation: dx = {eq.to_str(number_format='.4f')}")
    return {"eq": eq, "edge_list": edge_list, "num_nodes": num_nodes}


def get_predict_func(args, result_dict):
    """ 给定训练好的模型, 返回一个预测函数 predict_func, 根据历史数据预测未来数据 """
    eq = result_dict["eq"]
    edge_list = result_dict["edge_list"]
    num_nodes = result_dict["num_nodes"]

    def predict_func(data, node_info, time_info):
        """ 根据历史数据预测未来数据
        Args:
            同 train_model 函数中的参数
        Returns:
            pred: np.ndarray, shape (n_nodes), 预测的下一步数据矩阵
        """
        if data.ndim == 2: # (History, Nodes)
            data_dict = {f'x_{n}': data[args.hist_steps - 1 - n, :] for n in range(args.hist_steps)}
        elif data.ndim == 3: # (Batch, History, Nodes)
            data_dict = {f'x_{n}': data[:, args.hist_steps - 1 - n, :] for n in range(args.hist_steps)}
        else:
            raise ValueError(f"Input data must have shape (History, Nodes) or (Batch, History, Nodes), but got {data.shape}")
        dx = eq.eval(data_dict, edge_list=edge_list, num_nodes=num_nodes)
        return data_dict['x_0'] + dx

    predict_func.batched = True
    return predict_func


def format_result(args, result_dict) -> str:
    """ 将训练结果格式化为字符串, 以便打印给用户看 """
    eq = result_dict["eq"]
    return f"Formula: dx = {eq.to_str(number_format='.6f')}"
