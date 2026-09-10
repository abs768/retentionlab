"""Metric computation and diagnostic plots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config  # noqa: E402

logger = logging.getLogger(__name__)

PLOT_STYLE: dict[str, Any] = {
    "figure.figsize": (6.0, 4.5),
    "figure.dpi": 130,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
}


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = config.CLASSIFICATION_THRESHOLD,
) -> dict[str, float]:
    """Ranking, probability-quality, and thresholded classification metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.min() < 0.0 or y_prob.max() > 1.0:
        raise ValueError("Predicted probabilities must lie in [0, 1].")
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_observations": int(y_true.size),
        "positive_rate": float(y_true.mean()),
        "threshold": float(threshold),
    }


def _finalise(fig: plt.Figure, ax: plt.Axes, title: str, path: Path) -> Path:
    ax.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info("Wrote figure %s", path.name)
    return path


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, path: Path) -> Path:
    """ROC curve with the random-classifier diagonal for reference."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color="#1f77b4", label=f"Model (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="grey", label="Random")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right")
        return _finalise(fig, ax, "ROC curve (test set)", path)


def plot_precision_recall_curve(
    y_true: np.ndarray, y_prob: np.ndarray, path: Path
) -> Path:
    """Precision-recall curve with the prevalence baseline."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    prevalence = float(np.asarray(y_true).mean())
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots()
        ax.plot(recall, precision, color="#1f77b4", label=f"Model (PR-AUC = {ap:.3f})")
        ax.axhline(
            prevalence, ls="--", color="grey", label=f"Prevalence = {prevalence:.3f}"
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="upper right")
        return _finalise(fig, ax, "Precision-recall curve (test set)", path)


def plot_calibration_curve(
    curves: dict[str, np.ndarray], path: Path, n_bins: int = 10
) -> Path:
    """Reliability diagram for one or more sets of predicted probabilities.

    ``curves`` maps a label to a ``(y_true, y_prob)`` pair.
    """
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfectly calibrated")
        for (label, (y_true, y_prob)), colour in zip(
            curves.items(), ["#1f77b4", "#d62728", "#2ca02c"]
        ):
            observed, predicted = calibration_curve(
                y_true, y_prob, n_bins=n_bins, strategy="quantile"
            )
            ax.plot(predicted, observed, "o-", color=colour, label=label)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed churn frequency")
        ax.legend(loc="upper left")
        return _finalise(fig, ax, "Calibration curve (test set)", path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    path: Path,
    threshold: float = config.CLASSIFICATION_THRESHOLD,
) -> Path:
    """Confusion matrix at the decision threshold."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        ax.imshow(matrix, cmap="Blues")
        ax.grid(False)
        labels = ["No churn", "Churn"]
        ax.set_xticks([0, 1], labels=labels)
        ax.set_yticks([0, 1], labels=labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        threshold_colour = matrix.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:,}",
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > threshold_colour else "black",
                )
        return _finalise(
            fig, ax, f"Confusion matrix (test set, threshold = {threshold:g})", path
        )
