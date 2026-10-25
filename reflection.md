# RetentionLab Project Reflection

## 1. System Design

**Core user actions**

- A retention analyst gets every customer ranked by calibrated churn risk and sees how many
  churners a 10%, 20% or 30% outreach budget would reach.
- A planner learns how large a randomized retention test must be before spending money on one.
- An application sends customers to the API and gets back probabilities, a ranked outreach list,
  or a drift report on a recent batch.

**1a. Initial design**

I started from the idea, described it to Codex, and had it propose the class and module design
before writing any code. The Phase 1 design ([`diagrams/uml_draft.mmd`](diagrams/uml_draft.mmd))
splits the analysis into modules that each do one job:

- **`config`** holds every constant (seed, split sizes, capacities, experiment grid) so no number
  is hidden inside a function.
- **`validation`** checks the raw file before anything else runs.
- **`data`** loads, cleans and splits the data, returning a `DataSplits` tuple so the train,
  validation and test sets always travel together.
- **`features` and `models`** build the preprocessors and the two candidate models, always inside
  a `Pipeline` so preprocessing can only be fitted on training rows.
- **`evaluation`, `policy`, `experimentation` and `association`** (called `inference` in
  Phase 1) each answer one business
  question: how good is the ranking, how far does a budget go, how big must a test be, and which
  features are associated with churn.
- **`pipeline`** is the only module that knows the order of the steps. It returns one
  `PipelineResult` with everything produced.

I reviewed this design before approving it. What I checked was whether each module could be
tested on its own and whether the test set could reach any decision. Keeping `pipeline` as the
single orchestrator made the second question easy to answer: the holdout test set is used in
exactly one function, `evaluate_on_test`, after both selection steps.

**1b. Design changes**

Comparing [`uml_draft.mmd`](diagrams/uml_draft.mmd) with [`uml_final.mmd`](diagrams/uml_final.mmd)
shows how the design grew. The analytical modules' logic did not change after Phase 1, apart from `reporting` gaining the
serving facts it writes out; the serving layer was added around them:

- **`ModelService`** loads the frozen model once and verifies it before use.
- **Pydantic schemas** (`Customer`, `BatchRequest`, `RankRequest`, `Prediction`,
  `RankingResult`) define the API contract.
- **`app.inference.customers_to_frame`** is the one feature adapter every endpoint uses, so
  features always reach the model in the same order and types. This is what prevents
  training–serving skew.
- **`monitoring`** builds its drift reference by calling the original `data.make_splits` and
  using only the training rows, so the serving layer reuses the analysis instead of
  re-implementing it.

The decision I care most about here is that the model was **not** retrained for serving. The
API uses the exact file the analysis produced, verified by checksum, so every metric in the
README describes the model that is actually served.

In Phase 6 I renamed the package from its working name `churnguard` to `retentionlab` and the
chi-square module from `inference` to `association`, because two different modules called
`inference` (statistical inference and model inference) was confusing. The rename was purely
structural: the model file's checksum and every saved metric stayed identical. The model's version
ID still says `churnguard`, on purpose: a deployed artifact's ID should never be rewritten.

---

## 2. Modeling Logic and Tradeoffs

**a. Constraints and priorities**

- **Budget first.** A retention team can only contact a fraction of customers, so this is
  capacity-constrained targeting: what matters is recall@k and precision@k at the capacity the
  team actually has, not overall accuracy.
- **Probabilities must mean something.** Calibration matters because a business would multiply
  risk by customer value; a well-ordered but miscalibrated score would mislead that math.
- **Honesty about causality.** The data has no record of any retention action, so the project
  can say who looks risky but not who would be saved.

**b. Tradeoffs**

- **PR-AUC over accuracy or F1.** In the champion/challenger comparison, the gradient-boosting
  challenger won on PR-AUC (0.643 vs 0.628)
  and Brier score but lost on F1. Picking by F1 would have chosen the other model for reasons
  that don't match the business use. The margin is small, and the README says so.
- **Calibration helped only slightly.** Sigmoid calibration improved validation Brier from
  0.13919 to 0.13809 and slightly lowered PR-AUC. I kept it and reported the small gain plainly
  rather than overstating it.
- **No hyperparameter search.** With about 1,000 validation rows, an aggressive search mostly
  overfits the validation set. Fixed, conservative settings were a deliberate choice.
- **Power analysis instead of an effect estimate.** Rather than inventing a treatment effect, the
  project sizes the A/B test that could measure one: about 3,915 customers per arm to detect a
  3-point drop at 80% power.

---

## 3. AI Collaboration

**a. How I used Codex**

This was an agentic coding workflow with me as the human in the loop. Each phase followed the
same cycle: I described the idea,
Codex proposed a design, I checked whether that design made sense for the problem, and only then
did it write the modular Python files. After each build I ran the tests and read the generated
reports before starting the next phase. The full phase log is in
[`ai_interactions.md`](ai_interactions.md).

**b. What I checked at each checkpoint**

- **Design before code.** I approved the class and module layout before implementation, so
  structural problems were caught while they were cheap to fix.
- **Leakage.** I confirmed that identifiers and the target never reach a model and that all
  preprocessing is fitted on training data only. Tests now enforce both.
- **Claims against evidence.** Every number in the README is checked by
  `scripts/verify_readme_claims.py`, so the agent could not write a result the pipeline did not produce.
- **Definition of done: executed, not claimed.** When something could not be run (the Docker daemon was unavailable,
  remote CI had not run yet), it was recorded as untested instead of described as working.
- **Acceptance criteria per phase.** For the serving phase, the analytical code and the model
  file had to stay unchanged, confirmed by a full rerun that reproduced every ranking. For the
  rename in Phase 6, the model checksum and every saved metric had to come out identical.

**c. What I learned about being the lead**

The agent is fast at producing code that looks finished. My value was deciding what "correct"
meant before it started, and checking the result against that definition rather than against
how polished it looked. Writing the rules down in [`AGENTS.md`](AGENTS.md) made that repeatable.

---

## 4. Testing and Verification

**a. What is tested**

89 tests, all passing on Python 3.10:

- **Data and leakage:** splits are disjoint, cover the data, keep prevalence within a point,
  and are reproducible; learned imputation medians equal the training medians.
- **Analysis:** policy math, sample sizes, chi-square statistics and pipeline wiring.
- **Serving:** a tampered model file is rejected, schemas reject impossible inputs, batch scoring
  matches single scoring, ranking handles ties and capacity edges, and errors never leak payloads.
- **Monitoring:** the reference uses training rows only, and deliberately shifted batches are flagged.

**b. Confidence**

**4/5.** Every result is backed by executed tests and saved evidence, and the deployment was
verified endpoint by endpoint. What is missing: the Docker image has not been built locally, there
is no load test, and the test metrics come from a single split without confidence intervals.

---

## 5. Reflection

**What went well**

- The design-first loop. Because the module design was agreed before coding, adding the whole
  serving layer in Phase 2 did not require changing any modeling, evaluation or policy module.
- Treating every claim as something to verify. The README check, the checksum and the
  reproducibility rerun meant I could trust the numbers I was presenting.

**What I would improve**

- **Keep the documentation current at every checkpoint.** The README was written in stages, and
  by the end it contradicted itself in four places (for example, "no deployment attempted" next
  to "deployed and verified"). The numbers were all correct; the status sentences were stale.
- **Add uncertainty.** Bootstrap confidence intervals on the test metrics and the policy lift
  would show how much of a one-point difference is noise.
- **Rank by value, not just risk.** Probability × customer value × offer cost can reorder the
  list, and it is the version a business would actually use.
- **Build and run the Docker image locally**, so the container path is verified by me and not only by CI.

**Key takeaway**

Prediction and decision are different problems. A model can rank churners well and still tell
you nothing about whether contacting them helps; the honest next step is the experiment this
project sizes, not a bigger model.
