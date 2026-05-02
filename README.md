<h1 align="center">turba-models</h1>

<p align="center">
  <strong>Model package for loading pretrained soil fertility and fertilizer recommendation models for Morocco.</strong>
</p>

<p align="center">
  <!-- PyPI version -->
  <a href="https://pypi.org/project/turba-models/">
    <img src="https://img.shields.io/pypi/v/turba-models.svg?label=PyPI" alt="PyPI" />
  </a>

  <!-- Security -->
  <a href="https://socket.dev/pypi/package/turba-models">
    <img src="https://badge.socket.dev/pypi/package/turba-models/?artifact_id=tar-gz#1764083045680" alt="Socket" />
  </a>

  <!-- Downloads -->
  <a href="https://pepy.tech/project/turba-models">
    <img src="https://static.pepy.tech/badge/turba-models" alt="Downloads" />
  </a>

  <!-- License -->
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" />
  </a>
</p>


`turba-models` is the pretrained model package of the turba ecosystem. It provides a simple way to inspect the published model artifacts, load a model for a supported crop, and generate direct NPK recommendations.

## Scope of this first release

This release publishes one direct recommendation model per available crop. The models were selected from the following candidates using a fixed deterministic 80/20 split and benchmarked with the same evaluation protocol:

- Extra Trees
- LightGBM
- CatBoost
- Random Forest
- XGBoost
- Linear Regression
- Ridge
- Elastic Net
- AdaBoost

The published models use only the following input features:

- `longitude`
- `latitude`
- `soil_ph`
- `organic_matter_pct`
- `available_p2o5`
- `available_k2o`

Outputs are:

- `recommended_n`
- `recommended_p2o5`
- `recommended_k2o`

## Installation

```bash
pip install turba-models
```

For compatibility with the packaged artifacts, use an environment with:

- `scikit-learn >= 1.6, < 1.7`
- `lightgbm >= 4, < 5`
- `xgboost >= 2, < 3`

## Quick start

```python
import pandas as pd
import turba_models as tm

print(tm.list_models())

model = tm.load_model("Wheat (Rainfed)")

X = pd.DataFrame([
    {
        "longitude": -6.85,
        "latitude": 33.97,
        "soil_ph": 7.1,
        "organic_matter_pct": 1.2,
        "available_p2o5": 45.0,
        "available_k2o": 180.0,
    }
])

predictions = tm.predict_recommendation(model, X)
print(predictions)
```

## Public API

```python
from turba_models import (
    list_models,
    load_model,
    predict_recommendation,
    regression_report,
)
```

### `list_models()`
Returns the published model entries and their metadata.

### `load_model(model_name)`
Loads a packaged `.joblib` model. The function accepts either the crop name or the published model name.

### `predict_recommendation(model, X)`
Runs inference and returns a DataFrame with:

- `recommended_n`
- `recommended_p2o5`
- `recommended_k2o`

### `regression_report(y_true, y_pred, target_names=None)`
Returns a DataFrame with:

- `r2`
- `mae`
- `medae`
- `rmse`
- `mape`
- `smape`

## Published models in this package

- Barley (Rainfed) — LightGBM
- Maize (Grain) — XGBoost
- Maize (Silage) — XGBoost
- Wheat (Irrigated) — LightGBM
- Wheat (Rainfed) — LightGBM

## Reproducibility

- packaged `.joblib` model files under `src/turba_models/models/`
- the training notebook used to produce the current results under `notebooks/`
- exported benchmark summaries under `reports/`