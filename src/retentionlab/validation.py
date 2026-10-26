"""Programmatic data-quality validation with a machine-readable report."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def validate_raw_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Run schema, uniqueness, missingness, and target checks on the raw frame.

    Returns a JSON-serialisable report. ``passed`` is True only when every
    hard check (schema, row count, duplicate IDs, target values) holds; soft
    findings such as coercible blanks are recorded as ``warnings``.
    """
    checks: dict[str, Any] = {}
    warnings: list[str] = []

    observed = list(df.columns)
    expected = list(config.EXPECTED_COLUMNS)
    missing_columns = [c for c in expected if c not in observed]
    unexpected_columns = [c for c in observed if c not in expected]
    checks["schema"] = {
        "expected_columns": expected,
        "observed_columns": observed,
        "missing_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "passed": not missing_columns and not unexpected_columns,
    }

    checks["row_count"] = {
        "observed": int(df.shape[0]),
        "expected": config.EXPECTED_ROW_COUNT,
        "passed": int(df.shape[0]) == config.EXPECTED_ROW_COUNT,
    }
    checks["column_count"] = {"observed": int(df.shape[1])}

    duplicate_ids = int(df[config.ID_COLUMN].duplicated().sum())
    checks["duplicate_customer_ids"] = {
        "count": duplicate_ids,
        "passed": duplicate_ids == 0,
    }
    checks["duplicate_rows"] = {"count": int(df.duplicated().sum())}

    observed_targets = sorted(df[config.TARGET_COLUMN].dropna().unique().tolist())
    allowed_targets = sorted([config.POSITIVE_LABEL, config.NEGATIVE_LABEL])
    prevalence = float((df[config.TARGET_COLUMN] == config.POSITIVE_LABEL).mean())
    checks["target"] = {
        "observed_values": observed_targets,
        "allowed_values": allowed_targets,
        "positive_prevalence": prevalence,
        "positive_count": int((df[config.TARGET_COLUMN] == config.POSITIVE_LABEL).sum()),
        "passed": observed_targets == allowed_targets,
    }

    raw_missing = {
        column: int(count)
        for column, count in df.isna().sum().items()
        if int(count) > 0
    }
    checks["missing_values_raw"] = {"columns": raw_missing}

    coercion: dict[str, Any] = {}
    for column in config.NUMERIC_COERCION_COLUMNS:
        as_numeric = pd.to_numeric(df[column].astype(str).str.strip(), errors="coerce")
        n_failed = int(as_numeric.isna().sum() - df[column].isna().sum())
        coercion[column] = {
            "raw_dtype": str(df[column].dtype),
            "values_failing_numeric_parse": n_failed,
        }
        if n_failed:
            warnings.append(
                f"{column}: {n_failed} value(s) could not be parsed as numeric "
                "and were coerced to NaN (imputed inside the training pipeline)."
            )
    checks["numeric_coercion"] = coercion

    hard_checks = ("schema", "row_count", "duplicate_customer_ids", "target")
    passed = all(bool(checks[name]["passed"]) for name in hard_checks)

    report = {
        "dataset": str(config.RAW_DATA_PATH.name),
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "checks": checks,
        "warnings": warnings,
        "passed": passed,
    }
    logger.info("Validation passed=%s with %d warning(s)", passed, len(warnings))
    return report


def save_validation_report(report: dict[str, Any], path: Path) -> Path:
    """Write the validation report to disk as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
    return path
