"""Tests for prospective experiment-design calculations."""

from __future__ import annotations

import pytest

from retentionlab import experimentation as ex


def test_smaller_effects_require_larger_samples() -> None:
    sizes = [
        ex.required_sample_size_per_arm(0.30, absolute_reduction=d, power=0.80)
        for d in (0.05, 0.03, 0.02)
    ]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


def test_higher_power_requires_larger_samples() -> None:
    at_80 = ex.required_sample_size_per_arm(0.30, absolute_reduction=0.03, power=0.80)
    at_90 = ex.required_sample_size_per_arm(0.30, absolute_reduction=0.03, power=0.90)
    assert at_90 > at_80


def test_treatment_rate_and_absolute_reduction_agree() -> None:
    by_reduction = ex.required_sample_size_per_arm(0.40, absolute_reduction=0.05)
    by_rate = ex.required_sample_size_per_arm(0.40, treatment_rate=0.35)
    assert by_reduction == by_rate


def test_sample_size_requires_exactly_one_effect_specification() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        ex.required_sample_size_per_arm(0.30)
    with pytest.raises(ValueError, match="exactly one"):
        ex.required_sample_size_per_arm(0.30, absolute_reduction=0.02, treatment_rate=0.28)


def test_mde_shrinks_as_sample_size_grows() -> None:
    effects = [
        ex.minimum_detectable_effect(0.35, n)["mde_absolute_reduction"]
        for n in (1_000, 5_000, 25_000)
    ]
    assert effects == sorted(effects, reverse=True)
    assert all(0.0 < e < 0.35 for e in effects)


def test_mde_and_sample_size_are_mutually_consistent() -> None:
    baseline, n_per_arm = 0.35, 5_000
    mde = ex.minimum_detectable_effect(baseline, n_per_arm, power=0.80)
    required = ex.required_sample_size_per_arm(
        baseline, absolute_reduction=mde["mde_absolute_reduction"], power=0.80
    )
    assert required == pytest.approx(n_per_arm, rel=0.01)


def test_design_table_has_both_analyses_and_positive_sample_sizes() -> None:
    table = ex.build_experiment_design_table(0.35)
    assert set(table["analysis"]) == {"required_sample_size", "minimum_detectable_effect"}
    assert (table["n_per_arm"] > 0).all()
    assert (table["total_n"] == 2 * table["n_per_arm"]).all()
