# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""【样例代码】
黑盒 MLP 动力学模型：
    x(t+1) = x(t) + f_MLP(x(t), x(t-1), ..., x(t-N))
其中 N 由 --hist_steps 参数控制。
MLP 直接建模所有采样节点的整体状态，输出所有节点的下一步变化量。
"""
from __future__ import annotations

import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset


def update_parser(parser):
    """当前算法添加的命令行参数"""
    parser.add_argument("--mlp_hidden_dims", type=str, default="256,256", help="Hidden dims, e.g. 256,256.")
    parser.add_argument("--mlp_epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--mlp_batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--mlp_lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--mlp_weight_decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--mlp_device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device.")
    parser.set_defaults(hist_steps=10)
    return parser


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(last_dim, hidden_dim), nn.GELU()]
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _build_dataset(data, time_info, hist_steps):
    x = []
    dx = []
    for t in range(hist_steps - 1, len(time_info) - 1):
        # 检查时间是否连续
        if np.all(np.diff(time_info.index[t - hist_steps + 1 : t + 2]) == 1):
            x.append(data[t - hist_steps + 1 : t + 1].reshape(-1))
            dx.append(data[t + 1] - data[t])
    x = np.asarray(x, dtype=np.float32)
    dx = np.asarray(dx, dtype=np.float32)
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(dx))


def train_model(args, data, node_info, time_info):
    """给定训练数据, 训练模型并返回
    Args:
        data: np.ndarray, shape (n_times, n_nodes), 训练数据矩阵
        node_info: pd.DataFrame, 包含节点特征的 DataFrame
        time_info: pd.DataFrame, 包含时间特征的 DataFrame
    """
    hist_steps = args.hist_steps
    n_nodes = data.shape[1]
    hidden_dims = [int(x) for x in args.mlp_hidden_dims.split(",") if x.strip()]
    device = torch.device(args.mlp_device)

    dataset = _build_dataset(data, time_info, hist_steps)
    loader = DataLoader(dataset, batch_size=args.mlp_batch_size, shuffle=True)

    model = MLP(hist_steps * n_nodes, n_nodes, hidden_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.mlp_lr, weight_decay=args.mlp_weight_decay)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in tqdm(range(args.mlp_epochs), desc="Training MLP", unit="epoch", disable=not args.verbose):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    return {
        "model": model.cpu(),
        "loss": loss.item(),
        "hist_steps": hist_steps,
        "n_nodes": n_nodes,
        "hidden_dims": hidden_dims,
        "train_rows": len(dataset),
    }


def get_predict_func(args, result_dict):
    """给定训练好的模型, 返回一个预测函数 predict_func, 根据历史数据预测未来数据"""
    model = result_dict["model"].to(args.mlp_device)
    hist_steps = result_dict["hist_steps"]
    n_nodes = result_dict["n_nodes"]
    model.eval()

    def predict_func(data, node_info, time_info):
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, -hist_steps:, :]
            squeeze = True
        else:
            arr = arr[:, -hist_steps:, :]
            squeeze = False
        B = arr.shape[0]
        x_tensor = torch.from_numpy(arr.reshape(B, hist_steps * n_nodes)).to(args.mlp_device)
        with torch.no_grad():
            dx = model(x_tensor).cpu().numpy()
        out = arr[:, -1, :] + dx
        return out[0] if squeeze else out

    predict_func.batched = True
    return predict_func


def format_result(args, result_dict) -> str:
    """将训练结果格式化为字符串, 以便打印给用户看"""
    hidden_dims_str = ",".join(str(dim) for dim in result_dict["hidden_dims"])
    return (
        f"MLP with hidden dims: {hidden_dims_str}\n"
        f"Parameters: {sum(p.numel() for p in result_dict['model'].parameters()):,}\n"
        f"Trained on {result_dict['train_rows']} samples\n"
        f"Final loss: {result_dict['loss']:.6f}"
    )
