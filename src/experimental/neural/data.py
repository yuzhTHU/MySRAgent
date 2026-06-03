# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal


def load_data(args): # 已经过人类审核，不许修改这个函数的接口和行为
    data_path = Path(args.data_dir) / "data.npy"
    node_path = Path(args.data_dir) / "node_info.csv"
    time_path = Path(args.data_dir) / "time_info.csv"
    if missing := [str(path) for path in (data_path, node_path, time_path) if not path.exists()]:
        raise FileNotFoundError(f"Missing required files: {missing}")

    data = np.load(data_path, mmap_mode="r")
    if args.normalize:
        # 每个神经元的活动进行 z-score 标准化
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    node_info = pd.read_csv(node_path, index_col=0)
    time_info = pd.read_csv(time_path, index_col=0)
    time_info['time'] = time_info.index.astype(np.float64) / args.sampling_hz
    if data.ndim != 2:
        raise ValueError(f"Expected data.npy to be 2-D, got shape={data.shape}.")
    if data.shape[0] != len(time_info):
        raise ValueError(f"Time mismatch: data rows={data.shape[0]}, time rows={len(time_info)}.")
    if data.shape[1] != len(node_info):
        raise ValueError(f"Node mismatch: data cols={data.shape[1]}, node rows={len(node_info)}.")
    return data, node_info, time_info


def sample_indices( # 已经过人类审核，不许修改这个函数的接口和行为
    node_info: pd.DataFrame,
    time_info: pd.DataFrame,
    n_nodes: int,
    seed: int,
    stimulus: Literal["all", "Awake", "REM", "NREM"],
) -> tuple[np.ndarray, np.ndarray]:
    """用 seed 采样 n_nodes 个 node_idx, 按 stimulus 选择 time_idx, 返回它们的索引数组。"""
    if not 1 <= n_nodes <= len(node_info):
        raise ValueError(
            f"n_nodes must satisfy 1 <= n_nodes <= {len(node_info)}, got {n_nodes}."
        )
    rng = np.random.default_rng(seed)
    node_idx = np.sort(rng.choice(len(node_info), size=n_nodes, replace=False))
    if stimulus == "all":
        time_idx = np.arange(len(time_info) - 1, dtype=np.int64)
    else:
        labels = time_info["stimuli"].astype(str).to_numpy()
        mask = (labels[:-1] == stimulus) & (labels[1:] == stimulus)
        time_idx = np.arange(len(time_info) - 1, dtype=np.int64)[mask]
    if len(time_idx) < 2:
        raise ValueError(f"Need at least two eligible transition times for state={state!r}.")
    return node_idx, time_idx


def node_features(node_info: pd.DataFrame, node_idx: np.ndarray) -> pd.DataFrame:
    node = node_info.iloc[node_idx].copy()
    node["node_idx"] = node_idx
    node["node_id"] = pd.to_numeric(node["ids"], errors="coerce").fillna(-1.0).astype(float)
    node["coord_x"] = pd.to_numeric(node["coords_2d_x"], errors="coerce").astype(float)
    node["coord_y"] = pd.to_numeric(node["coords_2d_y"], errors="coerce").astype(float)
    return node[["node_idx", "node_id", "coord_x", "coord_y", "acs", "names"]]
