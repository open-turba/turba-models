from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLUMNS = ["recommended_n", "recommended_p2o5", "recommended_k2o"]
REQUIRED_FEATURE_COLUMNS = [
    "longitude",
    "latitude",
    "soil_ph",
    "organic_matter_pct",
    "available_p2o5",
    "available_k2o",
]


def predict_recommendation(model, X: pd.DataFrame, as_frame: bool = True):
    missing = [c for c in REQUIRED_FEATURE_COLUMNS if c not in X.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X_use = X[REQUIRED_FEATURE_COLUMNS].copy()
    preds = model.predict(X_use)
    if not as_frame:
        return preds

    arr = np.asarray(preds)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return pd.DataFrame(arr, columns=TARGET_COLUMNS[: arr.shape[1]], index=X.index)
