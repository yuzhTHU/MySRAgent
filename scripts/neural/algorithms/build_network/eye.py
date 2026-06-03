# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""【样例代码】
构造一个最简单的网络, 每个节点只与自己有一条边
"""


def build_network(data, node_info, time_info):
    """构造一个最简单的网络, 每个节点只与自己有一条边
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
    Returns:
        edge_list: list of (source_node, target_node) tuples, 有向边列表
        num_nodes: int, 节点数量
    """
    num_nodes = len(node_info)
    edge_list = [(i, i) for i in range(num_nodes)]
    return {'edge_list': edge_list, 'num_nodes': num_nodes}
