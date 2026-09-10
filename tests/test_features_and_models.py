"""Tests for preprocessing and model behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from churnguard import config, data, evaluation, features, models


def test_infer_feature_types_partitions_columns_and_excludes_identifiers(
    splits: data.DataSplits,
) -> None:
    numeric, categorical = features.infer_feature_types(splits.X_train)
    assert set(numeric) & set(categorical) == set()
    assert len(numeric) + len(categorical) == splits.X_train.shape[1]
    assert config.ID_COLUMN not in numeric + categorical
    assert "TotalCharges" in numeric and "Contract" in categorical


def test_linear_preprocessor_fits_on_train_only_and_removes_nan(
    splits: data.DataSplits,
) -> None:
    numeric, categorical = features.infer_feature_types(splits.X_train)
    preprocessor = features.build_linear_preprocessor(numeric, categorical)
    train_matrix = preprocessor.fit_transform(splits.X_train)
    test_matrix = preprocessor.transform(splits.X_test)

    assert train_matrix.shape[0] == len(splits.X_train)
    assert train_matrix.shape[1] == test_matrix.shape[1]
    dense = train_matrix.toarray() if hasattr(train_matrix, "toarray") else np.asarray(train_matrix)
    assert not np.isnan(dense).any()
    # The imputer's learned median must come from training rows only.
    fitted_median = preprocessor.named_transformers_["numeric"].named_steps["impute"]
    expected = splits.X_train[numeric].median().to_numpy()
    assert np.allclose(fitted_median.statistics_, expected, equal_nan=True)


def test_tree_preprocessor_maps_unseen_categories_to_unknown_code(
    splits: data.DataSplits,
) -> None:
    numeric, categorical = features.infer_feature_types(splits.X_train)
    preprocessor = features.build_tree_preprocessor(numeric, categorical)
    preprocessor.fit(splits.X_train)

    unseen = splits.X_test.head(1).copy()
    unseen.loc[:, "Contract"] = "Quarterly-Novel-Level"
    transformed = np.asarray(preprocessor.transform(unseen))
    assert (transformed == features.UNKNOWN_ORDINAL_CODE).any()


@pytest.mark.parametrize("model_name", [models.BASELINE_NAME, models.CHALLENGER_NAME])
def test_models_produce_probabilities_in_unit_interval(
    splits: data.DataSplits, model_name: str
) -> None:
    model = models.build_candidate_models(splits.X_train)[model_name]
    model.fit(splits.X_train, splits.y_train)
    probabilities = model.predict_proba(splits.X_val)[:, 1]
    assert probabilities.shape == (len(splits.X_val),)
    assert probabilities.min() >= 0.0 and probabilities.max() <= 1.0
    # A useful model must beat the trivial constant-probability ranker.
    assert evaluation.compute_metrics(splits.y_val.to_numpy(), probabilities)["roc_auc"] > 0.7


def test_compute_metrics_on_a_perfect_ranker() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.01, 0.02, 0.98, 0.99])
    metrics = evaluation.compute_metrics(y_true, y_prob)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["brier_score"] < 0.01


def test_compute_metrics_rejects_out_of_range_probabilities() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        evaluation.compute_metrics(np.array([0, 1]), np.array([0.5, 1.5]))
