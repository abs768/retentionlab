"""Prospective A/B experiment design for a retention intervention.

The dataset contains no randomised treatment assignment, so nothing here
estimates a treatment effect. These functions answer a forward-looking design
question only: *how large would a randomised trial need to be* to detect a
churn reduction of an assumed size, if one exists.

Power calculations use the normal approximation for a two-sided test comparing
two independent proportions, via ``statsmodels`` with Cohen's h as the effect
size.
"""

from __future__ import annotations

import logging
import math

import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from . import config

logger = logging.getLogger(__name__)

_POWER_SOLVER = NormalIndPower()


def required_sample_size_per_arm(
    baseline_rate: float,
    absolute_reduction: float | None = None,
    treatment_rate: float | None = None,
    alpha: float = config.EXPERIMENT_ALPHA,
    power: float = 0.80,
) -> int:
    """Customers required *per arm* for a two-sided two-proportion test.

    Supply either ``absolute_reduction`` (percentage points of churn removed)
    or ``treatment_rate`` directly. The result is rounded up to a whole
    customer.
    """
    if (absolute_reduction is None) == (treatment_rate is None):
        raise ValueError("Provide exactly one of absolute_reduction or treatment_rate.")
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError("baseline_rate must be in (0, 1).")
    if treatment_rate is None:
        treatment_rate = baseline_rate - float(absolute_reduction)
    if not 0.0 < treatment_rate < 1.0:
        raise ValueError("Implied treatment_rate must be in (0, 1).")
    if math.isclose(treatment_rate, baseline_rate):
        raise ValueError("Baseline and treatment rates must differ.")

    effect_size = proportion_effectsize(baseline_rate, treatment_rate)
    n_per_arm = _POWER_SOLVER.solve_power(
        effect_size=abs(effect_size),
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative="two-sided",
    )
    return int(math.ceil(float(n_per_arm)))


def minimum_detectable_effect(
    baseline_rate: float,
    n_per_arm: int,
    alpha: float = config.EXPERIMENT_ALPHA,
    power: float = 0.80,
) -> dict[str, float]:
    """Smallest absolute churn reduction detectable with ``n_per_arm`` per arm."""
    if n_per_arm < 2:
        raise ValueError("n_per_arm must be at least 2.")
    effect_size = float(
        _POWER_SOLVER.solve_power(
            effect_size=None,
            nobs1=n_per_arm,
            alpha=alpha,
            power=power,
            ratio=1.0,
            alternative="two-sided",
        )
    )
    # Invert Cohen's h: h = 2*asin(sqrt(p_control)) - 2*asin(sqrt(p_treatment)).
    phi_control = math.asin(math.sqrt(baseline_rate))
    treatment_rate = math.sin(max(phi_control - effect_size / 2.0, 0.0)) ** 2
    return {
        "baseline_rate": float(baseline_rate),
        "n_per_arm": int(n_per_arm),
        "total_n": int(2 * n_per_arm),
        "alpha": float(alpha),
        "power": float(power),
        "cohens_h": effect_size,
        "mde_treatment_rate": float(treatment_rate),
        "mde_absolute_reduction": float(baseline_rate - treatment_rate),
        "mde_relative_reduction": float(
            (baseline_rate - treatment_rate) / baseline_rate
        ),
    }


def build_sample_size_table(
    baseline_rate: float,
    absolute_reductions: tuple[float, ...] = config.EXPERIMENT_ABSOLUTE_REDUCTIONS,
    powers: tuple[float, ...] = config.EXPERIMENT_POWERS,
    alpha: float = config.EXPERIMENT_ALPHA,
) -> pd.DataFrame:
    """Required-sample-size grid across assumed effect sizes and power levels."""
    rows = []
    for power in powers:
        for reduction in absolute_reductions:
            n_per_arm = required_sample_size_per_arm(
                baseline_rate=baseline_rate,
                absolute_reduction=reduction,
                alpha=alpha,
                power=power,
            )
            rows.append(
                {
                    "analysis": "required_sample_size",
                    "baseline_churn_rate": baseline_rate,
                    "assumed_absolute_reduction": reduction,
                    "assumed_treatment_churn_rate": baseline_rate - reduction,
                    "assumed_relative_reduction": reduction / baseline_rate,
                    "alpha": alpha,
                    "power": power,
                    "n_per_arm": n_per_arm,
                    "total_n": 2 * n_per_arm,
                }
            )
    return pd.DataFrame(rows)


def build_mde_table(
    baseline_rate: float,
    sample_sizes: tuple[int, ...] = config.MDE_SAMPLE_SIZES,
    powers: tuple[float, ...] = config.EXPERIMENT_POWERS,
    alpha: float = config.EXPERIMENT_ALPHA,
) -> pd.DataFrame:
    """Minimum-detectable-effect grid across realistic per-arm sample sizes."""
    rows = []
    for power in powers:
        for n_per_arm in sample_sizes:
            result = minimum_detectable_effect(
                baseline_rate=baseline_rate, n_per_arm=n_per_arm, alpha=alpha, power=power
            )
            rows.append(
                {
                    "analysis": "minimum_detectable_effect",
                    "baseline_churn_rate": result["baseline_rate"],
                    "alpha": result["alpha"],
                    "power": result["power"],
                    "n_per_arm": result["n_per_arm"],
                    "total_n": result["total_n"],
                    "mde_absolute_reduction": result["mde_absolute_reduction"],
                    "mde_treatment_churn_rate": result["mde_treatment_rate"],
                    "mde_relative_reduction": result["mde_relative_reduction"],
                }
            )
    return pd.DataFrame(rows)


def build_experiment_design_table(
    baseline_rate: float,
    absolute_reductions: tuple[float, ...] = config.EXPERIMENT_ABSOLUTE_REDUCTIONS,
    sample_sizes: tuple[int, ...] = config.MDE_SAMPLE_SIZES,
    powers: tuple[float, ...] = config.EXPERIMENT_POWERS,
    alpha: float = config.EXPERIMENT_ALPHA,
) -> pd.DataFrame:
    """Concatenate the sample-size and MDE grids into one tidy design table."""
    size_table = build_sample_size_table(baseline_rate, absolute_reductions, powers, alpha)
    mde_table = build_mde_table(baseline_rate, sample_sizes, powers, alpha)
    return pd.concat([size_table, mde_table], ignore_index=True, sort=False)
