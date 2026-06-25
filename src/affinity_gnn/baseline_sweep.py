"""Classical regression sweep — the baseline the GNN must beat.

Eight estimators trained on identical Morgan-fingerprint + AAC features under
the same scaffold split and the same CV protocol, so the comparison is fair.
This replaces the original circular single-Random-Forest setup (which "predicted"
a label derived from its only feature) with a genuine multi-model benchmark.

Each estimator is returned with a small param dict describing the
hyperparameters we actually set, so the caller (``train.py``) can log them to
MLflow without re-deriving them here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:  # pragma: no cover
    _HAS_XGB = False

from .evaluate import regression_metrics


@dataclass
class ModelSpec:
    name: str
    estimator: Any
    params: dict[str, Any] = field(default_factory=dict)


def build_model_specs(random_seed: int = 42) -> list[ModelSpec]:
    """The eight-model regression roster (XGBoost included when available)."""
    specs = [
        ModelSpec("LinearRegression", LinearRegression(), {}),
        ModelSpec("Ridge", Ridge(alpha=1.0, random_state=random_seed),
                  {"alpha": 1.0}),
        ModelSpec("Lasso", Lasso(alpha=0.001, random_state=random_seed,
                                 max_iter=10000), {"alpha": 0.001}),
        ModelSpec("ElasticNet",
                  ElasticNet(alpha=0.001, l1_ratio=0.5,
                             random_state=random_seed, max_iter=10000),
                  {"alpha": 0.001, "l1_ratio": 0.5}),
        ModelSpec("RandomForest",
                  RandomForestRegressor(n_estimators=300, n_jobs=-1,
                                        random_state=random_seed),
                  {"n_estimators": 300}),
        ModelSpec("GradientBoosting",
                  GradientBoostingRegressor(n_estimators=300,
                                            random_state=random_seed),
                  {"n_estimators": 300}),
        ModelSpec("SVR", SVR(C=1.0, kernel="rbf"), {"C": 1.0, "kernel": "rbf"}),
        ModelSpec("KNN", KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
                  {"n_neighbors": 5}),
    ]
    if _HAS_XGB:
        specs.insert(
            6,
            ModelSpec(
                "XGBoost",
                XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                             subsample=0.8, n_jobs=-1, random_state=random_seed),
                {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05},
            ),
        )
    return specs


def cross_val_metrics(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    random_seed: int = 42,
) -> dict[str, float]:
    """Out-of-fold CV metrics: every point is scored from a model that did not
    see it in training. Gives an honest, leakage-free in-sample estimate."""
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    oof_pred = cross_val_predict(estimator, X, y, cv=cv, n_jobs=None)
    return regression_metrics(y, oof_pred)


def fit_and_test_metrics(estimator, X_train, y_train, X_test, y_test) -> dict[str, float]:
    """Fit on the full training split, evaluate on the held-out scaffold test set."""
    estimator.fit(X_train, y_train)
    return regression_metrics(y_test, estimator.predict(X_test))
