import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.multioutput import MultiOutputRegressor

from turba_models.inference import predict_recommendation


def test_predict_recommendation_returns_expected_columns():
    X = pd.DataFrame([
        {
            "longitude": -6.8,
            "latitude": 33.9,
            "soil_ph": 7.0,
            "organic_matter_pct": 1.5,
            "available_p2o5": 40.0,
            "available_k2o": 180.0,
        }
    ])
    y = [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]]
    model = MultiOutputRegressor(DummyRegressor(strategy="mean")).fit(pd.concat([X, X], ignore_index=True), y)
    out = predict_recommendation(model, X)
    assert list(out.columns) == ["recommended_n", "recommended_p2o5", "recommended_k2o"]
    assert out.shape == (1, 3)
