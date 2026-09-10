"""Tests for the inferential section and pipeline-level guarantees."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnguard import config, inference, models, pipeline


def test_chi_square_reports_statistic_p_value_and_effect_size(
    prepared_df: pd.DataFrame,
) -> None:
    summary, details = inference.run_chi_square_suite(prepared_df)
    assert list(summary["feature"]) == list(config.CHI_SQUARE_FEATURES)
    assert (summary["chi2_statistic"] > 0).all()
    assert ((summary["p_value"] >= 0) & (summary["p_value"] <= 1)).all()
    assert ((summary["cramers_v"] >= 0) & (summary["cramers_v"] <= 1)).all()
    assert set(details["Contract"]["churn_rate_by_level"]) == {
        "Month-to-month",
        "One year",
        "Two year",
    }


def test_cramers_v_is_zero_for_independent_variables() -> None:
    independent = np.array([[50, 50], [50, 50]])
    assert inference.cramers_v(independent, chi2=0.0) == pytest.approx(0.0)


def test_model_selection_uses_pr_auc_not_accuracy() -> None:
    metrics = pd.DataFrame(
        [
            {"model": "high_accuracy", "pr_auc": 0.55, "brier_score": 0.14, "accuracy": 0.90},
            {"model": "high_pr_auc", "pr_auc": 0.65, "brier_score": 0.15, "accuracy": 0.70},
        ]
    )
    assert pipeline.select_model(metrics) == "high_pr_auc"


def test_fitted_model_never_saw_the_customer_id_column(splits) -> None:
    model = models.build_candidate_models(splits.X_train)[models.BASELINE_NAME]
    model.fit(splits.X_train, splits.y_train)
    seen = set(model.named_steps["preprocess"].feature_names_in_)
    assert config.ID_COLUMN not in seen
    assert config.TARGET_COLUMN not in seen
    assert config.TARGET_LABEL_COLUMN not in seen
