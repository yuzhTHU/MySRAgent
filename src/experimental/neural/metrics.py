# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
import numpy as np
import pandas as pd
from tqdm import tqdm


def pearsonr_flat(true: np.ndarray, pred: np.ndarray) -> float:
    true = np.asarray(true, dtype=np.float64).ravel()
    pred = np.asarray(pred, dtype=np.float64).ravel()
    if true.size < 2:
        return float("nan")
    true = true - true.mean()
    pred = pred - pred.mean()
    denom = np.sqrt(np.sum(true**2) * np.sum(pred**2))
    return float("nan") if denom == 0 else float(np.sum(true * pred) / denom)


def r2_score_flat(true: np.ndarray, pred: np.ndarray) -> float:
    true = np.asarray(true, dtype=np.float64).ravel()
    pred = np.asarray(pred, dtype=np.float64).ravel()
    denom = np.sum((true - true.mean()) ** 2)
    return float("nan") if denom == 0 else float(1.0 - np.sum((true - pred) ** 2) / denom)


def valid_start_positions(time_info: pd.DataFrame, hist_steps: int, pred_step: int) -> List[int]:
    steps = time_info.index.to_numpy()
    starts = []
    for pos in range(hist_steps - 1, len(time_info) - pred_step):
        window = steps[pos - hist_steps + 1 : pos + pred_step + 1]
        if np.all(np.diff(window) == 1):
            starts.append(pos)
    return starts


def is_contiguous(time_info, start, final):
    """ 检查 time_info[start:final] 的时间窗口 (不含 final) 是否连续 """
    if start < 0 or final >= len(time_info):
        return False
    else:
        window = time_info.index.to_numpy()[start:final]
        return np.all(np.diff(window) == 1)


def rollout_metrics(args, predict_func, data, node_info, time_info) -> pd.DataFrame:
    positions = valid_start_positions(time_info, args.hist_steps, pred_step=0)
    simulations = [np.asarray(data[pos + 1 - args.hist_steps : pos + 1], dtype=np.float64) for pos in positions]
    rows = [{'step': 0, 'seconds': 0.0, 'pearson': 1.0, 'r2': 1.0, 'n_time': len(positions), 'n_node': data.shape[1]}]
    for step in tqdm(range(1, args.max_rollout_steps + 1), desc="Rollout", unit="step", disable=not args.verbose):
        next_positions = []
        next_simulations = []
        preds, trues = [], []
        for i, (pos, sim) in enumerate(zip(positions, simulations)):
            if is_contiguous(time_info, pos + step - args.hist_steps, pos + step + 1):
                local_time_info = time_info.iloc[pos + step - args.hist_steps:pos + step]
                pred = predict_func(sim, node_info, local_time_info)
                preds.append(pred)
                trues.append(data[pos + step])
                next_positions.append(pos)
                next_simulations.append(np.concatenate([sim[1:], pred[None, :]], axis=0))
        positions = next_positions
        simulations = next_simulations
        rows.append({
            "step": step,
            "seconds": step / args.sampling_hz,
            "pearson": pearsonr_flat(trues, preds) if preds else np.nan,
            "r2": r2_score_flat(trues, preds) if preds else np.nan,
            "n_time": len(positions),
            "n_node": data.shape[1],
        })
    return pd.DataFrame(rows)


def first_crossing(metrics: pd.DataFrame, column: str, threshold: float) -> dict:
    metrics = metrics.sort_values("step") # 确保按 step 升序排列
    pos = metrics[column].ge(threshold).cummin().sum() - 1
    row = (
        metrics
        .iloc[[pos]] # 注意这里是 [[pos]], 保留 DataFrame, 否则变成 Series 后类型会变
        .rename(columns={column: "value"})
        [['step', 'seconds', 'value', 'n_time', 'n_node']]
        .to_dict("records")[0]
    )
    return row


def evaluate_func(args, predict_func, data, node_info, time_info):
    simulation = rollout_metrics(args, predict_func, data, node_info, time_info)
    result = {
        "pearson>0.8": first_crossing(simulation, "pearson", 0.8),
        "pearson>0.5": first_crossing(simulation, "pearson", 0.5),
        # "pearson>0.0": first_crossing(simulation, "pearson", 0.0),
        # "r2>0.8": first_crossing(simulation, "r2", 0.8),
        "r2>0.5": first_crossing(simulation, "r2", 0.5),
        "r2>0.0": first_crossing(simulation, "r2", 0.0),
    }
    return result, simulation
