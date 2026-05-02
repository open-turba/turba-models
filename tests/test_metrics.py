import numpy as np

from turba_models.metrics import regression_report


def test_regression_report_shape_and_columns():
    y_true = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])
    y_pred = y_true.copy()
    report = regression_report(y_true, y_pred, ["n", "p", "k"])
    assert list(report.columns) == ["target", "r2", "mae", "medae", "rmse", "mape", "smape"]
    assert len(report) == 4
    assert report.iloc[-1]["target"] == "overall_mean"
