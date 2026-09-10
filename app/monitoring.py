"""Fixed training bins for PSI; categorical total variation and missingness shifts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import CATEGORICAL_DRIFT_FEATURES, NUMERIC_DRIFT_FEATURES

EPSILON = 1e-6
MISSINGNESS_WARNING = 0.05


def bin_proportions(values: pd.Series, interior_edges: list[float]) -> list[float]:
    clean = values.dropna().to_numpy(dtype=float)
    counts, _ = np.histogram(clean, bins=[-np.inf, *interior_edges, np.inf])
    return (counts / max(len(clean), 1)).tolist()


def population_stability_index(expected, actual, epsilon: float = EPSILON) -> float:
    expected, actual = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    if expected.shape != actual.shape or expected.ndim != 1 or expected.size == 0:
        raise ValueError("PSI requires matching nonempty bin vectors")
    if (
        not np.isfinite(expected).all()
        or not np.isfinite(actual).all()
        or (expected < 0).any()
        or (actual < 0).any()
        or epsilon <= 0
    ):
        raise ValueError("Invalid PSI proportions")
    expected = np.maximum(expected, epsilon)
    actual = np.maximum(actual, epsilon)
    expected /= expected.sum()
    actual /= actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def shift_status(value: float) -> str:
    # Operational heuristics, not statistical significance cutoffs.
    return (
        "substantial_apparent_shift"
        if value >= 0.25
        else ("investigate" if value >= 0.10 else "little_apparent_shift")
    )


def build_profile(training: pd.DataFrame) -> dict:
    numeric = {}
    for name in NUMERIC_DRIFT_FEATURES:
        values = training[name]
        clean = values.dropna()
        interior = sorted(set(float(x) for x in clean.quantile(np.arange(0.1, 1, 0.1))))
        numeric[name] = {
            "count": int(values.count()),
            "missing_fraction": float(values.isna().mean()),
            "mean": float(clean.mean()),
            "std": float(clean.std(ddof=0)),
            "quantiles": {
                str(q): float(clean.quantile(q)) for q in (0, 0.05, 0.25, 0.5, 0.75, 0.95, 1)
            },
            "interior_bin_edges": interior,
            "bin_proportions": bin_proportions(values, interior),
        }
    categorical = {}
    for name in CATEGORICAL_DRIFT_FEATURES:
        values = training[name]
        categorical[name] = {
            "count": int(values.count()),
            "missing_fraction": float(values.isna().mean()),
            "proportions": values.dropna().value_counts(normalize=True).sort_index().to_dict(),
        }
    return {
        "profile_version": 1,
        "source_split": "train",
        "row_count": len(training),
        "random_seed": 42,
        "numeric": numeric,
        "categorical": categorical,
        "bin_convention": "[-infinity, interior edges, +infinity]; left-inclusive bins",
        "numeric_proportions_denominator": "nonmissing rows; missingness reported separately",
    }


def load_reference(path: Path) -> dict:
    profile = json.loads(path.read_text())
    if profile["source_split"] != "train" or profile["row_count"] <= 0:
        raise ValueError("Invalid reference provenance")
    if set(profile["numeric"]) != set(NUMERIC_DRIFT_FEATURES):
        raise ValueError("Invalid numeric reference contract")
    if set(profile["categorical"]) != set(CATEGORICAL_DRIFT_FEATURES):
        raise ValueError("Invalid categorical reference contract")
    for feature in profile["numeric"].values():
        edges = np.asarray(feature["interior_bin_edges"], dtype=float)
        props = np.asarray(feature["bin_proportions"], dtype=float)
        if not np.isfinite(edges).all() or not (np.diff(edges) > 0).all():
            raise ValueError("Invalid reference bins")
        if len(props) != len(edges) + 1:
            raise ValueError("Invalid reference proportions")
    for group in (profile["numeric"], profile["categorical"]):
        for feature in group.values():
            props = np.asarray(
                feature.get("bin_proportions", list(feature.get("proportions", {}).values())),
                dtype=float,
            )
            if not np.isfinite(props).all() or (props < 0).any() or not np.isclose(props.sum(), 1):
                raise ValueError("Invalid reference proportions")
            if not 0 <= feature["missing_fraction"] <= 1:
                raise ValueError("Invalid reference missingness")
    return profile


def check_drift(frame: pd.DataFrame, profile: dict) -> dict:
    numeric, categorical, missingness, warnings = {}, {}, {}, []
    statuses = []
    if len(frame) < 100:
        warnings.append("Small batch: drift estimates are unstable; inspect a larger window")
    for name, reference in profile["numeric"].items():
        values = frame[name]
        proportions = bin_proportions(values, reference["interior_bin_edges"])
        psi = (
            population_stability_index(reference["bin_proportions"], proportions)
            if values.notna().any()
            else None
        )
        status = shift_status(psi) if psi is not None else "insufficient_data"
        numeric[name] = {
            "psi": psi,
            "status": status,
            "actual_bin_proportions": proportions,
            "interior_bin_edges": reference["interior_bin_edges"],
        }
        statuses.append(status)
        if status != "little_apparent_shift":
            warnings.append(f"{name}: {status}")
    for name, reference in profile["categorical"].items():
        actual = frame[name].dropna().value_counts(normalize=True).to_dict()
        categories = set(reference["proportions"]) | set(actual)
        tvd = 0.5 * sum(
            abs(actual.get(c, 0) - reference["proportions"].get(c, 0)) for c in categories
        )
        status = shift_status(tvd)
        categorical[name] = {"total_variation_distance": tvd, "status": status}
        statuses.append(status)
        if status != "little_apparent_shift":
            warnings.append(f"{name}: {status}")
    for name, reference in {**profile["numeric"], **profile["categorical"]}.items():
        actual = float(frame[name].isna().mean())
        change = actual - reference["missing_fraction"]
        missingness[name] = {
            "reference_fraction": reference["missing_fraction"],
            "actual_fraction": actual,
            "change": change,
        }
        if abs(change) >= MISSINGNESS_WARNING:
            warnings.append(f"{name}: missingness changed by at least 5 percentage points")
    overall = (
        "substantial_apparent_shift"
        if "substantial_apparent_shift" in statuses
        else "investigate"
        if warnings
        else "little_apparent_shift"
    )
    return {
        "row_count": len(frame),
        "numeric_drift": numeric,
        "categorical_drift": categorical,
        "missingness_changes": missingness,
        "warnings": warnings,
        "overall_status": overall,
    }
