"""Metric functions pinned against hand-computed values."""

import numpy as np
import pytest

from affinity_gnn import evaluate


def test_perfect_prediction():
    y = [1.0, 2.0, 3.0, 4.0]
    assert evaluate.rmse(y, y) == pytest.approx(0.0)
    assert evaluate.mae(y, y) == pytest.approx(0.0)
    assert evaluate.r2(y, y) == pytest.approx(1.0)
    assert evaluate.pearson_r(y, y) == pytest.approx(1.0)


def test_rmse_mae_known_values():
    y_true = [0.0, 0.0, 0.0]
    y_pred = [3.0, 0.0, 0.0]      # errors: 3, 0, 0
    # MAE = 3/3 = 1 ; RMSE = sqrt(9/3) = sqrt(3)
    assert evaluate.mae(y_true, y_pred) == pytest.approx(1.0)
    assert evaluate.rmse(y_true, y_pred) == pytest.approx(np.sqrt(3.0))


def test_r2_mean_predictor_is_zero():
    y_true = [1.0, 2.0, 3.0, 4.0]
    mean_pred = [2.5, 2.5, 2.5, 2.5]
    assert evaluate.r2(y_true, mean_pred) == pytest.approx(0.0)


def test_pearson_perfect_anticorrelation():
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [4.0, 3.0, 2.0, 1.0]
    assert evaluate.pearson_r(y_true, y_pred) == pytest.approx(-1.0)


def test_pearson_constant_series_returns_zero():
    assert evaluate.pearson_r([1, 2, 3], [5, 5, 5]) == 0.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate.rmse([1.0, 2.0], [1.0, 2.0, 3.0])


def test_regression_metrics_keys():
    m = evaluate.regression_metrics([1, 2, 3], [1, 2, 4])
    assert set(m) == {"rmse", "mae", "pearson_r", "r2"}
