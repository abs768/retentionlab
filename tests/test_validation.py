"""Tests for the data-quality validation report."""

from __future__ import annotations

import pandas as pd

from churnguard import validation


def test_real_dataset_passes_validation_with_total_charges_warning(
    raw_df: pd.DataFrame,
) -> None:
    report = validation.validate_raw_dataframe(raw_df)
    assert report["passed"] is True
    assert report["checks"]["duplicate_customer_ids"]["count"] == 0
    assert report["checks"]["target"]["observed_values"] == ["No", "Yes"]
    assert 0.0 < report["checks"]["target"]["positive_prevalence"] < 1.0
    coercion = report["checks"]["numeric_coercion"]["TotalCharges"]
    assert coercion["values_failing_numeric_parse"] > 0
    assert any("TotalCharges" in w for w in report["warnings"])


def test_validation_detects_duplicate_ids_and_missing_columns(
    raw_df: pd.DataFrame,
) -> None:
    corrupted = pd.concat([raw_df, raw_df.head(1)], ignore_index=True)
    corrupted = corrupted.drop(columns=["tenure"])
    report = validation.validate_raw_dataframe(corrupted)
    assert report["passed"] is False
    assert report["checks"]["duplicate_customer_ids"]["count"] == 1
    assert report["checks"]["schema"]["missing_columns"] == ["tenure"]


def test_report_is_json_serialisable(raw_df: pd.DataFrame, tmp_path) -> None:
    report = validation.validate_raw_dataframe(raw_df)
    path = validation.save_validation_report(report, tmp_path / "report.json")
    assert path.exists() and path.read_text().strip().startswith("{")
