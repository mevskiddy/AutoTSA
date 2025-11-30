import numpy as np
from typing import Callable


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) + 1e-8
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100)


def get_metric(name: str) -> Callable:
    name = name.lower()
    if name == "mae":
        return mae
    if name == "rmse":
        return rmse
    if name == "mape":
        return mape
    if name == "smape":
        return smape
    raise ValueError(f"Unknown metric: {name}")
