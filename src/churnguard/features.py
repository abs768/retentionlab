"""Preprocessing pipelines built with ColumnTransformer.

Two variants are provided because the two model families want different
encodings:

* ``build_linear_preprocessor`` -> median imputation + standardisation +
  one-hot encoding, which LogisticRegression needs.
* ``build_tree_preprocessor`` -> ordinal encoding with an explicit unknown
  code, which histogram gradient boosting consumes natively.

Both are *unfitted* transformers; the pipeline fits them on training data only.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config

logger = logging.getLogger(__name__)

UNKNOWN_ORDINAL_CODE: int = -1


def infer_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split feature columns into (numeric, categorical) by dtype.

    The identifier and target are expected to be absent already; they are
    filtered defensively so a mistake upstream cannot leak them into a model.
    """
    excluded = {config.ID_COLUMN, config.TARGET_COLUMN, config.TARGET_LABEL_COLUMN}
    columns = [c for c in X.columns if c not in excluded]
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in columns if c not in numeric]
    logger.info(
        "Inferred %d numeric and %d categorical features",
        len(numeric),
        len(categorical),
    )
    return numeric, categorical


def build_linear_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Median-impute + scale numerics, most-frequent-impute + one-hot categoricals."""
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def build_tree_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Pass numerics through (NaN-tolerant) and ordinal-encode categoricals.

    Unknown categories seen at inference map to ``UNKNOWN_ORDINAL_CODE`` rather
    than raising, so the model degrades gracefully on unseen levels.
    """
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=UNKNOWN_ORDINAL_CODE,
                    encoded_missing_value=np.nan,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
