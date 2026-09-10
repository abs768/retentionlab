"""Retention outreach policy analysis.

These are *retrospective prioritisation* metrics computed on held-out data:
they describe how well the model ranks customers who did churn, not the effect
of contacting anyone. No causal interpretation is implied anywhere here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402
from .evaluation import PLOT_STYLE  # noqa: E402

logger = logging.getLogger(__name__)


def evaluate_policy(
    y_true: np.ndarray, y_prob: np.ndarray, capacity_fraction: float
) -> dict[str, float]:
    """Score a "contact the top ``capacity_fraction`` by predicted risk" policy.

    Lift is the ratio of the capture rate achieved by risk-ranked targeting to
    the capture rate a random sample of the same size would achieve in
    expectation (which equals ``capacity_fraction``).
    """
    if not 0.0 < capacity_fraction <= 1.0:
        raise ValueError("capacity_fraction must be in (0, 1].")
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    n_customers = y_true.size
    n_contacted = int(np.ceil(capacity_fraction * n_customers))
    # Descending risk; ties broken deterministically by original position.
    order = np.lexsort((np.arange(n_customers), -y_prob))
    contacted = y_true[order[:n_contacted]]

    total_churners = int(y_true.sum())
    captured = int(contacted.sum())
    realised_fraction = n_contacted / n_customers
    capture_rate = captured / total_churners if total_churners else float("nan")

    return {
        "capacity_fraction": float(capacity_fraction),
        "n_customers": int(n_customers),
        "n_contacted": n_contacted,
        "contacted_fraction": float(realised_fraction),
        "total_churners": total_churners,
        "churners_captured": captured,
        "churner_capture_rate": float(capture_rate),
        "precision_in_contacted": float(captured / n_contacted),
        "expected_random_capture": float(realised_fraction),
        "expected_random_churners": float(realised_fraction * total_churners),
        "lift_vs_random": float(capture_rate / realised_fraction),
    }


def evaluate_policies(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    capacity_fractions: tuple[float, ...] = config.POLICY_CAPACITY_FRACTIONS,
) -> pd.DataFrame:
    """Evaluate several outreach capacities and return one row per policy."""
    rows = [evaluate_policy(y_true, y_prob, f) for f in capacity_fractions]
    return pd.DataFrame(rows)


def cumulative_gains(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """Cumulative share of churners captured as outreach capacity increases."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    order = np.lexsort((np.arange(y_true.size), -y_prob))
    ranked = y_true[order]
    total = ranked.sum()
    contacted_fraction = np.arange(1, ranked.size + 1) / ranked.size
    captured_fraction = np.cumsum(ranked) / total
    return pd.DataFrame(
        {
            "contacted_fraction": contacted_fraction,
            "captured_churner_fraction": captured_fraction,
        }
    )


def plot_cumulative_gains(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    path: Path,
    capacity_fractions: tuple[float, ...] = config.POLICY_CAPACITY_FRACTIONS,
) -> Path:
    """Cumulative-gains chart with the evaluated capacities marked."""
    gains = cumulative_gains(y_true, y_prob)
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots()
        ax.plot(
            gains["contacted_fraction"],
            gains["captured_churner_fraction"],
            color="#1f77b4",
            label="Model-ranked outreach",
        )
        ax.plot([0, 1], [0, 1], "--", color="grey", label="Random outreach")
        for fraction in capacity_fractions:
            result = evaluate_policy(y_true, y_prob, fraction)
            ax.scatter(
                result["contacted_fraction"],
                result["churner_capture_rate"],
                color="#d62728",
                zorder=5,
            )
            ax.annotate(
                f"top {fraction:.0%}: {result['churner_capture_rate']:.0%}",
                (result["contacted_fraction"], result["churner_capture_rate"]),
                textcoords="offset points",
                xytext=(8, -10),
                fontsize=8,
            )
        ax.set_xlabel("Fraction of customers contacted")
        ax.set_ylabel("Fraction of actual churners captured")
        ax.set_title("Cumulative gains (test set)")
        ax.legend(loc="lower right")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    logger.info("Wrote figure %s", path.name)
    return path
