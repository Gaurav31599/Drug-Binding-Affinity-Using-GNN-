"""Regression metrics for continuous affinity prediction.

This is a *regression* task (predicting a continuous affinity score), so the
classification metrics some pipelines log by default — accuracy, precision,
recall, F1, ROC-AUC — do not apply and are deliberately omitted. We report
RMSE, MAE, Pearson r and R^2 instead. Each metric is implemented against a
known definition so the unit tests can pin exact values.
"""

from __future__ import annotations

import numpy as np


def _as_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    return yt, yp


def rmse(y_true, y_pred) -> float:
    yt, yp = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true, y_pred) -> float:
    yt, yp = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def pearson_r(y_true, y_pred) -> float:
    """Pearson correlation coefficient. Returns 0.0 if a series is constant."""
    yt, yp = _as_arrays(y_true, y_pred)
    if np.std(yt) == 0 or np.std(yp) == 0:
        return 0.0
    return float(np.corrcoef(yt, yp)[0, 1])


def r2(y_true, y_pred) -> float:
    """Coefficient of determination (1 - SS_res / SS_tot)."""
    yt, yp = _as_arrays(y_true, y_pred)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """All four metrics as a dict, ready for MLflow ``log_metrics``."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "pearson_r": pearson_r(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }
