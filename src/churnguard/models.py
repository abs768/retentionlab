"""Model definitions: an interpretable baseline and a nonlinear challenger."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config
from .features import build_linear_preprocessor, build_tree_preprocessor, infer_feature_types

logger = logging.getLogger(__name__)

BASELINE_NAME: str = "logistic_regression"
CHALLENGER_NAME: str = "hist_gradient_boosting"


def build_baseline_model(
    numeric_features: list[str], categorical_features: list[str]
) -> Pipeline:
    """Regularised logistic regression on scaled + one-hot encoded features."""
    return Pipeline(
        steps=[
            ("preprocess", build_linear_preprocessor(numeric_features, categorical_features)),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=config.RANDOM_SEED,
                ),
            ),
        ]
    )


def build_challenger_model(
    numeric_features: list[str], categorical_features: list[str]
) -> Pipeline:
    """Histogram gradient boosting on ordinal-encoded features.

    Hyper-parameters are fixed at conservative, well-known defaults rather than
    searched: the project is time-boxed and the marginal gain from tuning is
    small relative to the risk of overfitting the validation split.
    """
    n_numeric = len(numeric_features)
    n_categorical = len(categorical_features)
    categorical_mask = [False] * n_numeric + [True] * n_categorical
    return Pipeline(
        steps=[
            ("preprocess", build_tree_preprocessor(numeric_features, categorical_features)),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=300,
                    max_leaf_nodes=15,
                    min_samples_leaf=40,
                    l2_regularization=1.0,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=25,
                    categorical_features=categorical_mask,
                    random_state=config.RANDOM_SEED,
                ),
            ),
        ]
    )


def build_candidate_models(X_train: pd.DataFrame) -> dict[str, Pipeline]:
    """Construct the unfitted candidate models keyed by name."""
    numeric_features, categorical_features = infer_feature_types(X_train)
    return {
        BASELINE_NAME: build_baseline_model(numeric_features, categorical_features),
        CHALLENGER_NAME: build_challenger_model(numeric_features, categorical_features),
    }


def build_calibrated_model(
    base_model: Pipeline,
    method: str,
    cv: int = config.CALIBRATION_CV_FOLDS,
) -> "CalibratedClassifierCV":
    """Wrap an *unfitted* pipeline in cross-validated probability calibration.

    Cross-validated calibration (rather than calibrating a prefit model on the
    validation set) keeps the validation split usable as an honest comparison
    set for calibrated vs uncalibrated probabilities.
    """
    if method not in config.CALIBRATION_METHODS:
        raise ValueError(f"Unsupported calibration method: {method}")
    return CalibratedClassifierCV(
        estimator=clone(base_model),
        method=method,
        cv=cv,
        n_jobs=None,
    )
