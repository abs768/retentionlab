"""Tests for retention-policy prioritisation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from retentionlab import policy


def test_perfect_ranking_captures_every_churner_within_capacity() -> None:
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.99, 0.98, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03])
    result = policy.evaluate_policy(y_true, y_prob, 0.20)
    assert result["n_contacted"] == 2
    assert result["churners_captured"] == 2
    assert result["churner_capture_rate"] == pytest.approx(1.0)
    assert result["precision_in_contacted"] == pytest.approx(1.0)
    assert result["lift_vs_random"] == pytest.approx(5.0)


def test_uninformative_ranking_has_lift_of_one() -> None:
    # Churners spread evenly through a constant-score ranking: capture equals capacity.
    y_true = np.tile(np.array([1, 0, 0, 0, 0]), 20)
    y_prob = np.full(y_true.size, 0.5)
    result = policy.evaluate_policy(y_true, y_prob, 0.20)
    assert result["churner_capture_rate"] == pytest.approx(0.20)
    assert result["lift_vs_random"] == pytest.approx(1.0)


def test_lift_equals_capture_rate_over_contacted_fraction() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.binomial(1, 0.3, size=500)
    y_prob = np.clip(0.2 + 0.5 * y_true + rng.normal(0, 0.2, size=500), 0, 1)
    for fraction in (0.10, 0.20, 0.30):
        result = policy.evaluate_policy(y_true, y_prob, fraction)
        assert result["lift_vs_random"] == pytest.approx(
            result["churner_capture_rate"] / result["contacted_fraction"]
        )
        assert result["churners_captured"] <= result["n_contacted"]
        assert result["churners_captured"] <= result["total_churners"]


def test_capture_rate_is_non_decreasing_in_capacity() -> None:
    rng = np.random.default_rng(1)
    y_true = rng.binomial(1, 0.25, size=400)
    y_prob = np.clip(0.2 + 0.4 * y_true + rng.normal(0, 0.25, size=400), 0, 1)
    table = policy.evaluate_policies(y_true, y_prob, (0.10, 0.20, 0.30))
    rates = table["churner_capture_rate"].to_numpy()
    assert np.all(np.diff(rates) >= 0)
    assert list(table["capacity_fraction"]) == [0.10, 0.20, 0.30]


def test_invalid_capacity_fraction_raises() -> None:
    with pytest.raises(ValueError, match="capacity_fraction"):
        policy.evaluate_policy(np.array([0, 1]), np.array([0.1, 0.9]), 0.0)


def test_cumulative_gains_ends_at_full_capture() -> None:
    rng = np.random.default_rng(2)
    y_true = rng.binomial(1, 0.3, size=200)
    y_prob = rng.random(200)
    gains = policy.cumulative_gains(y_true, y_prob)
    assert len(gains) == 200
    assert gains["captured_churner_fraction"].iloc[-1] == pytest.approx(1.0)
    assert gains["contacted_fraction"].iloc[-1] == pytest.approx(1.0)
