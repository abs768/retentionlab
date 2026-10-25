"""End-to-end pipeline: load -> validate -> split -> train -> select -> report.

Ordering is deliberate and enforced by construction:

1. Model selection uses **validation** performance only.
2. Calibration is chosen on **validation** performance only.
3. The **test** split is scored exactly once, at the end, by
   :func:`evaluate_on_test`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from . import (
    association,
    config,
    data,
    evaluation,
    experimentation,
    models,
    policy,
    reporting,
    validation,
)

logger = logging.getLogger(__name__)

UNCALIBRATED_LABEL: str = "uncalibrated"


@dataclass
class PipelineResult:
    """Everything the pipeline produced, for the runner script to summarise."""

    validation_report: dict[str, Any]
    split_sizes: dict[str, int]
    validation_metrics: pd.DataFrame
    selected_model_name: str
    calibration_comparison: pd.DataFrame
    calibration_choice: str
    calibration_improved_brier: bool
    test_metrics: dict[str, float]
    policy_table: pd.DataFrame
    experiment_table: pd.DataFrame
    chi_square_summary: pd.DataFrame
    artifacts: dict[str, str] = field(default_factory=dict)


def _write_table(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.info("Wrote table %s (%d rows)", path.name, len(frame))
    return path


def _write_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    logger.info("Wrote %s", path.name)
    return path


def train_and_compare_models(
    splits: data.DataSplits,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit both candidates on train and score them on validation."""
    fitted: dict[str, Any] = {}
    rows = []
    for name, model in models.build_candidate_models(splits.X_train).items():
        logger.info("Training %s on %d rows", name, len(splits.X_train))
        model.fit(splits.X_train, splits.y_train)
        val_prob = model.predict_proba(splits.X_val)[:, 1]
        metrics = evaluation.compute_metrics(splits.y_val.to_numpy(), val_prob)
        rows.append({"model": name, "split": "validation", **metrics})
        fitted[name] = model
    return fitted, pd.DataFrame(rows)


def select_model(validation_metrics: pd.DataFrame) -> str:
    """Pick the model with the best validation PR-AUC, breaking ties on Brier.

    PR-AUC is the primary criterion because the business use case is ranking a
    minority positive class under a fixed outreach budget; accuracy is reported
    but deliberately not used for selection.
    """
    ranked = validation_metrics.sort_values(
        by=["pr_auc", "brier_score"], ascending=[False, True]
    )
    selected = str(ranked.iloc[0]["model"])
    logger.info("Selected model on validation PR-AUC: %s", selected)
    return selected


def choose_calibration(
    base_model: Any,
    splits: data.DataSplits,
) -> tuple[Any, str, pd.DataFrame, bool]:
    """Compare uncalibrated vs calibrated probabilities on the validation split.

    Returns the chosen fitted estimator, its label, the comparison table, and
    whether calibration actually improved the validation Brier score.
    """
    rows = []
    candidates: dict[str, Any] = {}

    uncalibrated = clone(base_model).fit(splits.X_train, splits.y_train)
    candidates[UNCALIBRATED_LABEL] = uncalibrated
    rows.append(
        {
            "variant": UNCALIBRATED_LABEL,
            **evaluation.compute_metrics(
                splits.y_val.to_numpy(), uncalibrated.predict_proba(splits.X_val)[:, 1]
            ),
        }
    )

    for method in config.CALIBRATION_METHODS:
        logger.info("Fitting %s calibration (%d-fold)", method, config.CALIBRATION_CV_FOLDS)
        calibrated = models.build_calibrated_model(base_model, method=method)
        calibrated.fit(splits.X_train, splits.y_train)
        candidates[method] = calibrated
        rows.append(
            {
                "variant": method,
                **evaluation.compute_metrics(
                    splits.y_val.to_numpy(),
                    calibrated.predict_proba(splits.X_val)[:, 1],
                ),
            }
        )

    comparison = pd.DataFrame(rows)
    best = comparison.sort_values("brier_score", ascending=True).iloc[0]
    choice = str(best["variant"])
    baseline_brier = float(
        comparison.loc[comparison["variant"] == UNCALIBRATED_LABEL, "brier_score"].iloc[0]
    )
    improved = bool(float(best["brier_score"]) < baseline_brier)
    logger.info(
        "Calibration choice=%s (validation Brier %.5f vs uncalibrated %.5f)",
        choice,
        float(best["brier_score"]),
        baseline_brier,
    )
    return candidates[choice], choice, comparison, improved


def evaluate_on_test(
    model: Any, splits: data.DataSplits
) -> tuple[dict[str, float], np.ndarray]:
    """Score the final model on the held-out test split. Called exactly once."""
    test_prob = model.predict_proba(splits.X_test)[:, 1]
    metrics = evaluation.compute_metrics(splits.y_test.to_numpy(), test_prob)
    logger.info(
        "Test performance: ROC-AUC=%.4f PR-AUC=%.4f Brier=%.4f",
        metrics["roc_auc"],
        metrics["pr_auc"],
        metrics["brier_score"],
    )
    return metrics, test_prob


def run_pipeline(
    raw_path: Path = config.RAW_DATA_PATH,
    save_artifacts: bool = True,
) -> PipelineResult:
    """Run every stage of the project and persist the reporting artifacts."""
    raw = data.load_raw_data(raw_path)
    validation_report = validation.validate_raw_dataframe(raw)
    if not validation_report["passed"]:
        raise ValueError(
            "Data validation failed; see the validation report for details."
        )

    prepared = data.prepare_dataset(raw)
    splits = data.make_splits(prepared)
    split_sizes = {
        "train": len(splits.X_train),
        "validation": len(splits.X_val),
        "test": len(splits.X_test),
    }

    fitted_models, validation_metrics = train_and_compare_models(splits)
    selected_name = select_model(validation_metrics)
    base_model = models.build_candidate_models(splits.X_train)[selected_name]

    final_model, calibration_choice, calibration_comparison, improved = choose_calibration(
        base_model, splits
    )
    test_metrics, test_prob = evaluate_on_test(final_model, splits)

    policy_table = policy.evaluate_policies(splits.y_test.to_numpy(), test_prob)
    # The RCT baseline is the observed churn rate among the customers a
    # top-20%-capacity policy would actually contact -- the eligible high-risk
    # population an intervention would target.
    high_risk_baseline = float(
        policy_table.loc[
            policy_table["capacity_fraction"] == 0.20, "precision_in_contacted"
        ].iloc[0]
    )
    experiment_table = experimentation.build_experiment_design_table(high_risk_baseline)
    chi_square_summary, chi_square_details = association.run_chi_square_suite(prepared)

    artifacts: dict[str, str] = {}
    if save_artifacts:
        artifacts.update(
            _persist(
                splits=splits,
                prepared=prepared,
                validation_report=validation_report,
                validation_metrics=validation_metrics,
                calibration_comparison=calibration_comparison,
                test_metrics=test_metrics,
                test_prob=test_prob,
                policy_table=policy_table,
                experiment_table=experiment_table,
                chi_square_summary=chi_square_summary,
                chi_square_details=chi_square_details,
                final_model=final_model,
                selected_name=selected_name,
                calibration_choice=calibration_choice,
                fitted_models=fitted_models,
                high_risk_baseline=high_risk_baseline,
            )
        )

    return PipelineResult(
        validation_report=validation_report,
        split_sizes=split_sizes,
        validation_metrics=validation_metrics,
        selected_model_name=selected_name,
        calibration_comparison=calibration_comparison,
        calibration_choice=calibration_choice,
        calibration_improved_brier=improved,
        test_metrics=test_metrics,
        policy_table=policy_table,
        experiment_table=experiment_table,
        chi_square_summary=chi_square_summary,
        artifacts=artifacts,
    )


def _persist(**kw: Any) -> dict[str, str]:
    """Write every table, figure, model, and summary artifact to disk."""
    splits: data.DataSplits = kw["splits"]
    test_prob: np.ndarray = kw["test_prob"]
    y_test = splits.y_test.to_numpy()
    paths: dict[str, Path] = {}

    # Processed splits (features + target, identifier deliberately excluded).
    for name, X, y in (
        ("train", splits.X_train, splits.y_train),
        ("validation", splits.X_val, splits.y_val),
        ("test", splits.X_test, splits.y_test),
    ):
        frame = X.copy()
        frame[config.TARGET_LABEL_COLUMN] = y.to_numpy()
        path = config.PROCESSED_DATA_DIR / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        paths[f"processed_{name}"] = path

    paths["validation_report"] = validation.save_validation_report(
        kw["validation_report"], config.REPORTS_DIR / "validation_report.json"
    )
    paths["validation_metrics"] = _write_table(
        kw["validation_metrics"], config.TABLES_DIR / "validation_model_comparison.csv"
    )
    paths["calibration_comparison"] = _write_table(
        kw["calibration_comparison"], config.TABLES_DIR / "calibration_comparison.csv"
    )
    paths["test_metrics"] = _write_table(
        pd.DataFrame([kw["test_metrics"]]), config.TABLES_DIR / "test_metrics.csv"
    )
    paths["test_metrics_json"] = _write_json(
        kw["test_metrics"], config.TABLES_DIR / "test_metrics.json"
    )
    paths["policy_table"] = _write_table(
        kw["policy_table"], config.TABLES_DIR / "retention_policy.csv"
    )
    paths["experiment_table"] = _write_table(
        kw["experiment_table"], config.TABLES_DIR / "experiment_design.csv"
    )
    paths["chi_square"] = _write_table(
        kw["chi_square_summary"], config.TABLES_DIR / "chi_square_tests.csv"
    )
    paths["chi_square_details"] = _write_json(
        kw["chi_square_details"], config.TABLES_DIR / "chi_square_details.json"
    )

    # Figures.
    paths["roc_curve"] = evaluation.plot_roc_curve(
        y_test, test_prob, config.FIGURES_DIR / "roc_curve.png"
    )
    paths["pr_curve"] = evaluation.plot_precision_recall_curve(
        y_test, test_prob, config.FIGURES_DIR / "precision_recall_curve.png"
    )
    paths["confusion_matrix"] = evaluation.plot_confusion_matrix(
        y_test, test_prob, config.FIGURES_DIR / "confusion_matrix.png"
    )
    uncalibrated_test_prob = kw["fitted_models"][kw["selected_name"]].predict_proba(
        splits.X_test
    )[:, 1]
    paths["calibration_curve"] = evaluation.plot_calibration_curve(
        {
            f"Final ({kw['calibration_choice']})": (y_test, test_prob),
            "Uncalibrated": (y_test, uncalibrated_test_prob),
        },
        config.FIGURES_DIR / "calibration_curve.png",
    )
    paths["cumulative_gains"] = policy.plot_cumulative_gains(
        y_test, test_prob, config.FIGURES_DIR / "cumulative_gains.png"
    )

    # Model artifact.
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / "final_model.joblib"
    joblib.dump(kw["final_model"], model_path)
    paths["final_model"] = model_path

    # Register the two narrative artifacts before the summary is built so the
    # manifest it contains is complete.
    resume_facts_path = config.REPORTS_DIR / "resume_facts.md"
    run_summary_path = config.REPORTS_DIR / "run_summary.json"
    paths["resume_facts"] = resume_facts_path
    paths["run_summary"] = run_summary_path

    # Run summary tying every headline number back to a generated file.
    summary = {
        "random_seed": config.RANDOM_SEED,
        "n_rows": int(kw["prepared"].shape[0]),
        "n_raw_columns": int(kw["prepared"].shape[1]) - 1,  # excludes derived churn
        "n_model_features": int(splits.X_train.shape[1]),
        "churn_prevalence": float(kw["prepared"][config.TARGET_LABEL_COLUMN].mean()),
        "split_sizes": {
            "train": len(splits.X_train),
            "validation": len(splits.X_val),
            "test": len(splits.X_test),
        },
        "selected_model": kw["selected_name"],
        "calibration_choice": kw["calibration_choice"],
        "test_metrics": kw["test_metrics"],
        "policy": kw["policy_table"].to_dict(orient="records"),
        "rct_baseline_churn_rate_high_risk": kw["high_risk_baseline"],
        "artifacts": {k: str(v.relative_to(config.PROJECT_ROOT)) for k, v in paths.items()},
    }
    _write_json(summary, run_summary_path)
    reporting.write_resume_facts(
        summary=summary,
        validation_metrics=kw["validation_metrics"],
        calibration_comparison=kw["calibration_comparison"],
        experiment_table=kw["experiment_table"],
        chi_square_summary=kw["chi_square_summary"],
        validation_report=kw["validation_report"],
        path=resume_facts_path,
    )

    return {k: str(v.relative_to(config.PROJECT_ROOT)) for k, v in paths.items()}
