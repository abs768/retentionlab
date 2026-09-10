"""Data loading, cleaning, target encoding, and leakage-safe splitting."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

logger = logging.getLogger(__name__)


class DataSplits(NamedTuple):
    """Feature/target frames for the three-way stratified split."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def load_raw_data(path: Path | str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load the immutable raw CSV exactly as distributed (all columns as-is)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Download it from "
            f"{config.DATA_SOURCE_URL} and save it there."
        )
    df = pd.read_csv(path)
    logger.info("Loaded raw data: %d rows, %d columns", df.shape[0], df.shape[1])
    return df


def coerce_numeric_columns(
    df: pd.DataFrame,
    columns: tuple[str, ...] = config.NUMERIC_COERCION_COLUMNS,
) -> pd.DataFrame:
    """Coerce text-encoded numeric columns to float, turning blanks into NaN.

    ``TotalCharges`` ships as an object column containing empty strings for
    customers with ``tenure == 0``; ``errors="coerce"`` makes those NaN so the
    imputer (fitted on training data only) can handle them.
    """
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(
                out[column].astype(str).str.strip(), errors="coerce"
            )
    return out


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map the ``Churn`` Yes/No column to an integer ``churn`` column (1/0)."""
    out = df.copy()
    mapping = {config.POSITIVE_LABEL: 1, config.NEGATIVE_LABEL: 0}
    unexpected = set(out[config.TARGET_COLUMN].unique()) - set(mapping)
    if unexpected:
        raise ValueError(f"Unexpected target values: {sorted(unexpected)}")
    out[config.TARGET_LABEL_COLUMN] = out[config.TARGET_COLUMN].map(mapping).astype(int)
    return out


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply row-independent cleaning: numeric coercion plus target encoding.

    These steps are deliberately stateless (no fitted statistics), so applying
    them before splitting cannot leak information between splits.
    """
    return encode_target(coerce_numeric_columns(df))


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with the identifier and both target columns removed from X."""
    drop_columns = [
        column
        for column in (
            config.ID_COLUMN,
            config.TARGET_COLUMN,
            config.TARGET_LABEL_COLUMN,
        )
        if column in df.columns
    ]
    X = df.drop(columns=drop_columns)
    y = df[config.TARGET_LABEL_COLUMN].astype(int)
    return X, y


def make_splits(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    validation_size: float = config.VALIDATION_SIZE,
    random_state: int = config.RANDOM_SEED,
) -> DataSplits:
    """Stratified 70/15/15 train/validation/test split with a fixed seed.

    The split is performed in two stages: test is carved out first and never
    touched again, then validation is carved out of the remainder.
    """
    X, y = split_features_target(df)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # Re-scale the validation fraction so it stays 15% of the *full* dataset.
    val_fraction_of_temp = validation_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_fraction_of_temp,
        stratify=y_temp,
        random_state=random_state,
    )
    logger.info(
        "Split sizes -> train=%d, validation=%d, test=%d",
        len(X_train),
        len(X_val),
        len(X_test),
    )
    return DataSplits(X_train, X_val, X_test, y_train, y_val, y_test)
