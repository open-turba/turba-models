from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-12, np.nan, np.abs(y_true))
    values = np.abs((y_true - y_pred) / denom)
    return float(np.nanmean(values) * 100.0)


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom < 1e-12, np.nan, denom)
    values = 2.0 * np.abs(y_pred - y_true) / denom
    return float(np.nanmean(values) * 100.0)


def regression_report(y_true, y_pred, target_names=None) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    if target_names is None:
        target_names = [f"target_{i}" for i in range(y_true.shape[1])]

    rows = []
    for i, name in enumerate(target_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        rows.append({
            "target": name,
            "r2": r2_score(yt, yp),
            "mae": mean_absolute_error(yt, yp),
            "medae": median_absolute_error(yt, yp),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mape": _mape(yt, yp),
            "smape": _smape(yt, yp),
        })

    report = pd.DataFrame(rows)
    overall = {
        "target": "overall_mean",
        "r2": report["r2"].mean(),
        "mae": report["mae"].mean(),
        "medae": report["medae"].mean(),
        "rmse": report["rmse"].mean(),
        "mape": report["mape"].mean(),
        "smape": report["smape"].mean(),
    }
    return pd.concat([report, pd.DataFrame([overall])], ignore_index=True)
