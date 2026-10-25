"""Tests for loading, cleaning, target encoding, and splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from retentionlab import config, data


def test_raw_data_has_expected_schema_and_row_count(raw_df: pd.DataFrame) -> None:
    assert list(raw_df.columns) == list(config.EXPECTED_COLUMNS)
    assert raw_df.shape[0] == config.EXPECTED_ROW_COUNT


def test_load_raw_data_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        data.load_raw_data(tmp_path / "does_not_exist.csv")


def test_total_charges_is_coerced_to_float_with_nan_for_blanks() -> None:
    frame = pd.DataFrame({"TotalCharges": ["29.85", " ", "1889.5", ""]})
    coerced = data.coerce_numeric_columns(frame)
    assert coerced["TotalCharges"].dtype.kind == "f"
    assert coerced["TotalCharges"].isna().sum() == 2
    assert coerced["TotalCharges"].iloc[0] == pytest.approx(29.85)


def test_real_total_charges_blanks_are_only_zero_tenure_customers(
    prepared_df: pd.DataFrame,
) -> None:
    missing = prepared_df[prepared_df["TotalCharges"].isna()]
    assert len(missing) > 0
    assert (missing["tenure"] == 0).all()


def test_target_is_mapped_to_binary_integers(prepared_df: pd.DataFrame) -> None:
    target = prepared_df[config.TARGET_LABEL_COLUMN]
    assert set(target.unique()) == {0, 1}
    assert target.dtype.kind in "iu"
    expected_positive = int((prepared_df[config.TARGET_COLUMN] == "Yes").sum())
    assert int(target.sum()) == expected_positive


def test_encode_target_rejects_unexpected_labels() -> None:
    frame = pd.DataFrame({config.TARGET_COLUMN: ["Yes", "No", "Maybe"]})
    with pytest.raises(ValueError, match="Unexpected target values"):
        data.encode_target(frame)


def test_features_exclude_customer_id_and_target(prepared_df: pd.DataFrame) -> None:
    X, y = data.split_features_target(prepared_df)
    assert config.ID_COLUMN not in X.columns
    assert config.TARGET_COLUMN not in X.columns
    assert config.TARGET_LABEL_COLUMN not in X.columns
    assert len(X) == len(y)


def test_splits_are_disjoint_and_cover_the_dataset(
    splits: data.DataSplits, prepared_df: pd.DataFrame
) -> None:
    train, val, test = set(splits.X_train.index), set(splits.X_val.index), set(splits.X_test.index)
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()
    assert len(train | val | test) == len(prepared_df)


def test_split_proportions_and_stratification(
    splits: data.DataSplits, prepared_df: pd.DataFrame
) -> None:
    n = len(prepared_df)
    assert len(splits.X_test) / n == pytest.approx(config.TEST_SIZE, abs=0.01)
    assert len(splits.X_val) / n == pytest.approx(config.VALIDATION_SIZE, abs=0.01)
    overall = prepared_df[config.TARGET_LABEL_COLUMN].mean()
    for y in (splits.y_train, splits.y_val, splits.y_test):
        assert y.mean() == pytest.approx(overall, abs=0.01)


def test_splitting_is_reproducible_and_seed_sensitive(prepared_df: pd.DataFrame) -> None:
    first = data.make_splits(prepared_df)
    second = data.make_splits(prepared_df)
    assert np.array_equal(first.X_test.index.to_numpy(), second.X_test.index.to_numpy())

    other = data.make_splits(prepared_df, random_state=config.RANDOM_SEED + 1)
    assert not np.array_equal(first.X_test.index.to_numpy(), other.X_test.index.to_numpy())
