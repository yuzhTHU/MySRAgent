# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""评估给定的白盒动力学方程的性能

【支持的变量】
x_0, x_1, ..., x_{N-1}: 历史数据
    x_0 表示当前时刻的数据, x_1 表示前一个时刻的数据, 以此类推
    N 由 --hist_steps 参数控制

【支持的算符】
常见非线性算符: pow2, pow3, sin, cos, tan, exp, log, tanh, sigmoid, ...
网络动力学算符: sour, targ, aggr
    sour: 将节点级变量映射为连边级变量, 若 edge_list[k] = (i, j), 则 sour(x_0)[k] = x_0[i]
    targ: 将节点级变量映射为连边级变量, 若 edge_list[k] = (i, j), 则 targ(x_0)[k] = x_0[j]
    aggr: 将连边级变量映射为节点级变量, aggr(sour(x_0))[j] = sum_{k such that edge_list[k][1] = j} sour(x_0)[k]
例如: dx_i = x_i + 0.5 * sum_{j such that edge_list[k] = (j, i)} sin(x_j - x_i) 可以表示为 dx = x_0 + 0.5 * aggr(sin(sour(x_0) - targ(x_0)))
其中映射关系大致为 x_j -> sour(x), x_i -> targ(x), sum_j -> aggr

【支持的功能】
- 指定 fit_params 时，算法会根据给定的方程式结构拟合参数（例如上面例子中的 1.0 和 0.5）以最小化预测值与真实值之间的误差；不指定 fit_params 时，则直接使用给定的方程式进行预测，不进行参数拟合。
- 根据指定的 network_builder (需要在 build_network 中实现对应算法) 构造 edge_list 和 num_nodes 以供方程式中的 sour, targ, aggr 算符使用。
    如果不指定 network_builder, 则无法使用 sour, targ, aggr 等网络动力学算符, 只能拟合类似 dx = 1.0 * x_0 + 0.5 * x_1 这样不包含节点间相关性的方程式。
- 如果你认为连边是有权重的，例如：dx_i = x_i + 0.5 * sum_j w_{k such that edge_list[k] = (i, j)} * sin(x_j - x_i)
    你可以在 network_builder.build_network 中算出这个 w (长度为 len(edge_list) 的一维数组), 然后返回
        {'edge_list': edge_list, 'num_nodes': num_nodes, 'w': nd2py.Number(w, nettype='edge')}
    此时 eq = nd.parse(eq_str, variables=variables) 会将这个 w 记录在 eq 中, 随后的 eq.eval(data_dict) 会自动将 w[k] 与 sour(x_0)[k] 进行逐元素乘法。
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path('../../..').absolute()
sys.path.insert(0, str(ROOT / "src" / "experimental"))

import logging
import numpy as np
import nd2py as nd
from pathlib import Path
from collections import defaultdict
from .build_network import list_algorithms, get_algorithm

_logger = logging.getLogger(f"sr_agent.{__name__}")


def update_parser(parser):
    """ 当前算法添加的命令行参数 """
    parser.add_argument("--equation", type=str, required=True, help="Equation string, e.g., dx = 1.0 * x_0 + 0.0 * x_1 or dx = 1.0 * x_0 + 0.5 * aggr(sin(sour(x_0) - targ(x_0)))")
    parser.add_argument("--fit_params", action='store_true', default=True, help="Whether to fit parameters in --equation from data. If False, will directly use the equation without fitting.")
    parser.add_argument("--network_builder", type=str, default=None, choices=list_algorithms(), help="Optional name of a network builder function defined in build_network, used to construct edge_list and num_nodes for equation evaluation.")
    parser.set_defaults(hist_steps=2) # 与 equation 中的 x_0, x_1 数量保持一致
    return parser


def _build_dataset(args, data, node_info, time_info):
    """ 将原始数据构造为 data_dict
        x_0 -> 当前时间步
        x_1 -> 前一个时间步
        ...
        dx -> 下一时间步与当前时间步的差值
    """
    data_dict = defaultdict(list)
    for t in range(args.hist_steps - 1, len(time_info) - 1):
        # 检查时间是否连续
        if np.all(np.diff(time_info.index[t - args.hist_steps + 1 : t + 2]) == 1):
            for n in range(args.hist_steps):
                data_dict[f'x_{n}'].append(data[t - n])
            data_dict['dx'].append(data[t + 1] - data[t])
    for key, value in data_dict.items():
        data_dict[key] = np.asarray(value, dtype=np.float32)
    
    if args.network_builder is not None:
        network_builder = get_algorithm(args.network_builder)
        data_dict |= network_builder.build_network(args, data, node_info, time_info)

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
    data_dict = _build_dataset(args, data, node_info, time_info)
    edge_list = data_dict.get('edge_list', None)
    num_nodes = data_dict.get('num_nodes', None)
    if '=' in args.equation:
        eq_str = args.equation.split('=', 1)[1].strip()
    else:
        eq_str = args.equation.strip()
    variables = {f'x_{n}': nd.Variable(f'x_{n}', nettype='node') for n in range(args.hist_steps)}
    for key in data_dict:
        if key not in variables:
            variables[key] = data_dict[key]
    eq = nd.parse(eq_str, variables=variables)

    for var in eq.iter_preorder():
        if isinstance(var, nd.Variable) and var.name not in variables:
            raise ValueError(
                f"Equation contains undefined variable: {var.name}. "
                f"Since --hist_steps={args.hist_steps}, only x_0, x_1, ..., x_{args.hist_steps-1} are allowed."
            )

    if args.fit_params:
        bfgs = nd.BFGSFit(eq, edge_list=edge_list, num_nodes=num_nodes, method='BFGS')
        bfgs.fit(data_dict, data_dict['dx'])
        eq = bfgs.expression

    _logger.note(f"Fitted equation: dx = {eq.to_str(number_format='.4f')}")
    result_dict = {"eq": eq, "edge_list": edge_list, "num_nodes": num_nodes}
    if edge_list is not None:
        save_path = Path(args.save_path)
        np.save(save_path / "edge_list.npy", np.column_stack(edge_list))
        result_dict['edge_list_path'] = str(save_path / "edge_list.npy")
    return result_dict


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
        if data.ndim == 2:  # (History, Nodes)
            data_dict = {f'x_{n}': data[args.hist_steps - 1 - n, :] for n in range(args.hist_steps)}
        elif data.ndim == 3:  # (Batch, History, Nodes)
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
    log = f"Formula: dx = {eq.to_str(number_format='.6f')}"
    if 'edge_list_path' in result_dict:
        log += f"\nEdge list saved to: {result_dict['edge_list_path']}"
    return log
