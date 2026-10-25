# AGENTS.md — RetentionLab

Instructions for any coding agent (Codex, Claude Code, Copilot) working in this repository.
Read this whole file before the first edit of a session.

## Project
RetentionLab ranks telecom customers by churn risk so a retention team with limited capacity
knows who to contact first, then serves that ranking through a FastAPI service.

- `src/churnguard/` — the analytical pipeline: validation, splits, models, calibration,
  evaluation, retention policy, experiment sizing, chi-square tests. Entry point:
  `pipeline.run_pipeline()`, called by `scripts/run_pipeline.py`.
- `app/` — the serving layer: typed schemas, model loading, scoring, ranking, drift
  monitoring, Prometheus metrics, landing page. Entry point: `app.main:app`.
- `scripts/` — build, verify, benchmark and documentation scripts.
- `tests/` — pytest suite for both halves.
- `diagrams/uml_final.mmd` — the class/module design. `diagrams/uml.mmd` is a copy of it.

## Current phase: Phase 6 — documentation and portfolio polish
- Allowed: `README.md`, `AGENTS.md`, `reflection.md`, `ai_interactions.md`, `diagrams/`, `docs/`.
- Not allowed: any change to `src/`, `app/`, `scripts/`, `tests/`, `models/`, `artifacts/` or `reports/`.
- Every number written in the README must come from a file in `reports/` or `artifacts/`.
  Run `python scripts/verify_readme_claims.py` after any README edit; it must report all claims verified.

## Rules that always apply
- **The model is frozen.** Never retrain, re-serialize or replace `models/final_model.joblib`.
  Its SHA256 is checked at startup and by `scripts/verify_model_artifact.py`.
- **The test set is scored once.** Model and calibration choices come from the validation split only.
- **No leakage.** `customerID` and the target never reach a model. Every transformer is fitted
  inside a `Pipeline` on the training split.
- **No causal claims.** The data has no treatment variable. Lift means "ranks churners well",
  never "saves customers". Experiment effect sizes are assumptions for sizing.
- **One adapter.** Every endpoint converts customers with `app.inference.customers_to_frame`.
  Don't build feature frames anywhere else.
- **No request-time training or loading.** The model loads once, in the FastAPI lifespan.
- **Never log customer payloads** or use IDs or raw values as metric labels.
- **New dependencies need a reason.** Ask before adding one; pin it in both `pyproject.toml`
  and `requirements.txt`. scikit-learn stays at 1.6.1 because the artifact was serialized with it.
- If the class/module design changes, update `diagrams/uml_final.mmd` in the same change
  and copy it to `diagrams/uml.mmd`.

## How to verify
Use the project venv on Python 3.10:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/download_data.py
python scripts/verify_model_artifact.py
python -m pytest
ruff check .
python scripts/verify_readme_claims.py
```

- A change is not done until all five pass. Report the actual output, not "should pass".
- If a test fails, explain why before changing code. Never edit a test just to make it pass.
- If you could not run a command, say so plainly instead of implying it passed.

## Style
- Python 3.10, type hints, `from __future__ import annotations` where the module already uses it.
- Ruff rules E, F, W, I, B; line length 100.
- One-line docstrings on public functions. Keep it simple; explain any change that adds complexity first.
