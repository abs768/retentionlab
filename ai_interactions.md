# AI Interactions Log

How RetentionLab was built with an AI coding agent, phase by phase. The agent was **Codex**;
the final documentation pass (Phase 6) was done with **Claude**. The commit for each phase is
listed so every entry can be checked against the code.

---

## The workflow I used in every phase

I treated myself as the human in the loop between my idea and the agent's output:

1. **Describe the idea.** I wrote a prompt explaining what the phase should achieve and why,
   in business terms first (for example, "a retention team can only call 20% of customers").
2. **Get the design before the code.** Codex proposed the classes, modules, attributes and
   methods first. I read that design and checked it against the problem before anything was built.
3. **Approve or redirect.** If the design made sense, I passed it on to the next step. If it
   didn't, I explained what was wrong and had Codex revise it.
4. **Build in small, modular pieces.** Codex wrote the Python files one module at a time,
   each with one responsibility, so a module could be read and tested on its own.
5. **Verify, then repeat.** Before moving on I ran the tests, read the generated reports,
   and asked the same question as in step 2: does this make sense? Only then did the next phase start.

The rules I wanted the agent to follow are kept in [`AGENTS.md`](AGENTS.md).

---

## Phase 1 — Analytical pipeline (`fa3fc9d`)

**What I asked for:** an end-to-end churn analysis that treats churn as a ranking problem under
a fixed outreach budget, with calibrated probabilities and a way to size a future experiment.

**What Codex produced:**
- The module design in [`diagrams/uml_draft.mmd`](diagrams/uml_draft.mmd): `config`, `validation`,
  `data` (with `DataSplits`), `features`, `models`, `evaluation`, `policy`, `experimentation`,
  `inference`, `reporting`, and `pipeline` (with `PipelineResult`) orchestrating them.
- The implementation of every module, `scripts/run_pipeline.py`, and 37 tests.
- `scripts/verify_readme_claims.py`, which checks each README number against the saved reports.

**What I checked before moving on:**
- That `customerID` and the target can never reach a model, and every transformer is fitted on
  the training split only.
- That the model and calibration are chosen on the validation split, and the test split is
  scored exactly once, after both choices are final.
- That selection uses PR-AUC rather than accuracy, since churners are a 26.5% minority and the
  business use is ranking.
- That the README separates ranking quality from causal impact, because the data has no
  treatment variable.
- Results at this checkpoint: 37 tests passed, Ruff clean, 107 of 107 README numbers verified,
  and three consecutive runs byte-identical (`reports/project_audit.md`).

## Phase 2 — ML serving and monitoring (`edf28e0`)

**What I asked for:** serve the existing model as an API, without retraining it, and add drift
monitoring and service metrics.

**What Codex produced:**
- The `app/` package: `schemas`, `model_service` (`ModelService`), `inference` (the single
  input adapter), `ranking`, `monitoring`, `metrics` (`ServiceMetrics`), `config` and `main`.
- Model metadata, a training-only drift reference, a Dockerfile, a CI workflow, verification
  and benchmark scripts, and 35 new tests (72 in total).

**What I checked before moving on:**
- That the analytical code and the model file were unchanged. A Python 3.10 rerun reproduced the
  run summary and every customer ranking; probability differences were at most `3.33e-16`
  (`reports/analytical_preservation.json`).
- That the model loads once at startup, verifies its SHA256 first, and fails closed (503) if
  anything is wrong.
- That customer payloads are never logged and never used as metric labels
  (`reports/security_portability_audit.json`).
- What could not be verified was written down instead of claimed: the Docker daemon was
  unavailable, so the image was not built locally, and remote CI had not run yet
  (`reports/engineering_audit.md`).

## Phase 3 — Public release preparation (`d6b2a08`)

**What I asked for:** get the repository ready to be public and deployable on Vercel.

**What Codex produced:** a first landing page (`app/landing.py`), `.vercelignore`, a deployment
verification script (`scripts/verify_deployment.py`) and generated landing-page facts.

**What I checked:** that the raw dataset was removed from the tracked files, because IBM's
license covers the code pattern and does not clearly grant redistribution of the data; it is now
downloaded and checksum-verified by `scripts/download_data.py` instead.

## Phase 4 — Landing page (`3af9e8f`)

**What I asked for:** a page a recruiter can understand in a minute, with live demos.

**What Codex produced:** the full landing page, its stylesheet and script, and tests for it.

**What I checked:** that every figure on the page is generated from the saved reports rather
than typed by hand, and that the demos call the real `/predict` and `/rank` endpoints.

## Phase 5 — Deployment verification (`5f981bd`, `2b983bc`)

**What I asked for:** confirm the public deployment works, then link the repository.

**What was verified:** all 14 public routes and page assets returned correct responses over HTTPS,
the live service loaded the checksum-verified model, and live `/predict` matched the local result
for the same input (`reports/deployment.json`).

## Phase 6 — Documentation and portfolio polish (this branch)

**What I asked for:** clean the repository up in the same structure as my CodePath AI110 projects.

**What Claude did:**
- Re-ran the full verification on Python 3.10: 89 tests passed, the model checksum matched, and
  184 of 184 README numbers were verified.
- Found four places where the README contradicted itself because it had been written in stages
  (for example, saying both "no deployment attempted" and "deployed and verified"), and fixed them.
- Wrote `diagrams/uml_draft.mmd` and `diagrams/uml_final.mmd` from the actual code, restructured
  the README, and added this file, `AGENTS.md` and `reflection.md`.

**What I checked:** that every number in the new README still passes
`scripts/verify_readme_claims.py`, and that no code, test, model or report file was changed.
