# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import numpy as np

__all__ = [
    "R2_score",
    "RMSE_score",
    "sMAPE_score",
    "MAPE_score",
    "MAE_score",
]


def R2_score(true, pred):
    return 1 - np.mean((true - pred) ** 2) / np.var(true)


def RMSE_score(true, pred):
    return np.sqrt(np.mean((true - pred) ** 2))


def sMAPE_score(true, pred):
    return 2 * np.mean(np.abs(true - pred) / (np.abs(true) + np.abs(pred) + 1e-6))


def MAPE_score(true, pred):
    return np.mean(np.abs(true - pred) / (np.abs(true) + 1e-6))


def MAE_score(true, pred):
    return np.mean(np.abs(true - pred))
