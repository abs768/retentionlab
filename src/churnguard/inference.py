"""Small pre-specified inferential section: chi-square tests of independence.

These tests describe association between a categorical attribute and observed
churn in this observational sample. They are not evidence of causation, and the
comparisons are fixed in ``config.CHI_SQUARE_FEATURES`` before looking at
results rather than selected post hoc.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from . import config

logger = logging.getLogger(__name__)


def cramers_v(contingency: np.ndarray, chi2: float) -> float:
    """Cramer's V effect size for an r x c contingency table."""
    n = contingency.sum()
    min_dim = min(contingency.shape) - 1
    if n == 0 or min_dim == 0:
        return float("nan")
    return float(np.sqrt((chi2 / n) / min_dim))


def chi_square_test(df: pd.DataFrame, feature: str) -> dict[str, object]:
    """Chi-square test of independence between ``feature`` and churn."""
    table = pd.crosstab(df[feature], df[config.TARGET_COLUMN])
    chi2, p_value, dof, _ = chi2_contingency(table.to_numpy())
    churn_rates = (
        df.groupby(feature)[config.TARGET_LABEL_COLUMN].mean().round(4).to_dict()
    )
    return {
        "feature": feature,
        "contingency_table": table.to_dict(),
        "churn_rate_by_level": churn_rates,
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "cramers_v": cramers_v(table.to_numpy(), float(chi2)),
        "n_observations": int(table.to_numpy().sum()),
    }


def run_chi_square_suite(
    df: pd.DataFrame, features: tuple[str, ...] = config.CHI_SQUARE_FEATURES
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Run all pre-specified tests, returning a summary frame and full details."""
    details = {feature: chi_square_test(df, feature) for feature in features}
    summary = pd.DataFrame(
        [
            {
                "feature": d["feature"],
                "n_observations": d["n_observations"],
                "chi2_statistic": d["chi2_statistic"],
                "degrees_of_freedom": d["degrees_of_freedom"],
                "p_value": d["p_value"],
                "cramers_v": d["cramers_v"],
            }
            for d in details.values()
        ]
    )
    logger.info("Ran %d pre-specified chi-square tests", len(summary))
    return summary, details
