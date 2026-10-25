"""RetentionLab: customer retention decision and experimentation engine.

A time-boxed, end-to-end retention project on the IBM Telco churn dataset:
validated data, two contrasting models, calibrated probabilities, an outreach
policy under fixed capacity, and a power analysis for the randomized
experiment the observational data cannot substitute for.
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = [
    "association",
    "config",
    "data",
    "evaluation",
    "experimentation",
    "features",
    "models",
    "pipeline",
    "policy",
    "reporting",
    "validation",
]
