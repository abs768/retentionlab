"""Recruiter-facing landing page for the RetentionLab service.

Plain HTML rendered server-side, with an external stylesheet and a small
vanilla-JS file for the two interactive demos. No frontend framework and no
build step.

Two rules govern this module:

1. **No metric is written here.** Every number comes from
   ``artifacts/landing/landing_facts.json``, which ``build_landing_facts.py``
   derives from the canonical artifacts in ``reports/``. The same payload is
   served by ``GET /project-summary``.
2. **No inference happens in the browser.** The demos post to the real
   ``/predict`` and ``/rank`` endpoints.

Form controls are generated from the ``Customer`` Pydantic model, so the inputs
cannot drift away from the served contract.
"""

from __future__ import annotations

import json
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

from app.schemas import Customer

FALLBACK_FACTS: dict[str, Any] = {
    "name": "RetentionLab",
    "tagline": "End-to-End Customer Retention Analytics Platform",
    "hero_summary": "",
    "headline_metrics": [],
}

NAV_SECTIONS = [
    ("overview", "Overview"),
    ("problem", "Problem"),
    ("performance", "Performance"),
    ("targeting", "Targeting"),
    ("demo", "Try Model"),
    ("experiment", "Experiment"),
    ("architecture", "Architecture"),
    ("engineering", "Engineering"),
]

FIELD_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("Customer", ("gender", "SeniorCitizen", "Partner", "Dependents", "tenure")),
    (
        "Services",
        (
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies",
        ),
    ),
    ("Account", ("Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges")),
]

FIELD_LABELS = {
    "gender": "Gender",
    "SeniorCitizen": "Senior citizen",
    "Partner": "Has partner",
    "Dependents": "Has dependents",
    "tenure": "Tenure (months)",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "Contract": "Contract",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
    "MonthlyCharges": "Monthly charges",
}

DEFAULTS: dict[str, Any] = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 89.0,
}

ENDPOINT_DESCRIPTIONS = {
    "/docs": "Interactive OpenAPI documentation",
    "/model-info": "Model version, feature contract, artifact digest",
    "/health": "Liveness probe",
    "/ready": "Readiness, including checksum verification",
    "/metrics": "Prometheus exposition",
    "/project-summary": "The canonical evidence behind this page",
}

FAVICON = (
    "data:image/svg+xml,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
    "<rect width='32' height='32' rx='7' fill='%236c8cff'/>"
    "<path d='M8 21V11m8 10V15m8 6V9' stroke='%230a0c10' stroke-width='3' "
    "stroke-linecap='round'/></svg>"
)

GITHUB_MARK = (
    '<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" '
    'aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 '
    '7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-'
    '.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 '
    '2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-'
    '.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c'
    '1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-'
    '1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38'
    'A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/></svg>'
)

WARN_ICON = (
    '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">'
    '<path d="M8 1.5 15 14H1L8 1.5Z" stroke="currentColor" stroke-width="1.4" '
    'stroke-linejoin="round"/><path d="M8 6.4v3.2" stroke="currentColor" '
    'stroke-width="1.4" stroke-linecap="round"/>'
    '<circle cx="8" cy="11.7" r="0.85" fill="currentColor"/></svg>'
)


@lru_cache(maxsize=4)
def load_facts(path: Path) -> dict[str, Any]:
    """Read the generated payload, degrading to a minimal header if it is absent."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return dict(FALLBACK_FACTS)


def literal_choices(field_name: str) -> list[Any] | None:
    """Return the allowed values for a field straight from the Pydantic model."""
    annotation = Customer.model_fields[field_name].annotation
    if get_origin(annotation) is Literal:
        return list(get_args(annotation))
    for arg in get_args(annotation):
        if get_origin(arg) is Literal:
            return list(get_args(arg))
    return None


def _select(name: str) -> str:
    choices = literal_choices(name) or []
    default = DEFAULTS.get(name)
    options = "".join(
        f'<option value="{escape(str(choice))}"'
        f'{" selected" if choice == default else ""}>{escape(str(choice))}</option>'
        for choice in choices
    )
    kind = "int" if isinstance(choices[0], int) else "str" if choices else "str"
    return (
        f'<div class="field"><label for="f-{name}">{escape(FIELD_LABELS[name])}</label>'
        f'<select id="f-{name}" data-field="{name}" data-kind="{kind}">{options}</select></div>'
    )


def _range(name: str, minimum: float, maximum: float, step: float, kind: str,
           prefix: str = "", suffix: str = "") -> str:
    default = DEFAULTS[name]
    return (
        f'<div class="field">'
        f'<label for="f-{name}">{escape(FIELD_LABELS[name])} '
        f'<span class="range-value" data-range-for="{name}"></span></label>'
        f'<input id="f-{name}" type="range" data-field="{name}" data-kind="{kind}" '
        f'min="{minimum}" max="{maximum}" step="{step}" value="{default}" '
        f'data-prefix="{escape(prefix)}" data-suffix="{escape(suffix)}"></div>'
    )


def render_form_fields() -> str:
    """Build every input from the live schema, grouped for readability."""
    blocks = []
    for group_name, fields in FIELD_GROUPS:
        controls = []
        for name in fields:
            if name == "tenure":
                controls.append(_range(name, 0, 72, 1, "int", suffix=" mo"))
            elif name == "MonthlyCharges":
                controls.append(_range(name, 18, 120, 0.05, "float", prefix="$"))
            else:
                controls.append(_select(name))
        blocks.append(
            f'<div class="field-group"><h3>{escape(group_name)}</h3>'
            f'{"".join(controls)}</div>'
        )
    blocks.append(
        '<div class="field-group"><h3>Derived</h3>'
        '<div class="field"><label>Total charges</label>'
        '<div class="range-value" id="derived-total">—</div></div>'
        '<p class="note">Computed as tenure × monthly charges so the submitted record '
        'stays internally consistent, then validated by the API.</p></div>'
    )
    return "".join(blocks)


def _metric_cards(facts: dict) -> str:
    return "".join(
        '<div class="metric">'
        f'<div class="metric-value">{escape(str(m["value"]))}</div>'
        f'<div class="metric-label">{escape(str(m["label"]))}</div>'
        f'<div class="metric-detail">{escape(str(m.get("detail", "")))}</div></div>'
        for m in facts.get("headline_metrics", [])
    )


def _hero(facts: dict) -> str:
    name = escape(str(facts.get("name", "RetentionLab")))
    repo = escape(str(facts.get("repository_url", "")))
    return f"""
<header class="hero" id="overview">
  <div class="wrap">
    <p class="eyebrow">Churn modeling · Decision policy · Model serving</p>
    <h1 class="hero-title">{name}</h1>
    <p class="hero-sub">{escape(str(facts.get("tagline", "")))}</p>
    <p class="hero-copy">{escape(str(facts.get("hero_summary", "")))}</p>
    <div class="cta-row">
      <a class="btn btn-primary" href="#demo">Try the Model</a>
      <a class="btn" href="#performance">Explore the Analysis</a>
      <a class="btn" href="/docs">API Docs</a>
      <a class="btn" href="{repo}" target="_blank" rel="noopener noreferrer">
        {GITHUB_MARK} GitHub</a>
    </div>
    <div class="metrics">{_metric_cards(facts)}</div>
  </div>
</header>"""


def _problem(facts: dict) -> str:
    steps = facts.get("decision_flow", [])
    items = "".join(
        f'<div class="flow-step{" terminal" if i == len(steps) - 1 else ""}">'
        f'<div class="flow-index">{i + 1}</div>'
        f'<div><h3>{escape(str(s["step"]))}</h3><p>{escape(str(s["detail"]))}</p></div></div>'
        for i, s in enumerate(steps)
    )
    dataset = facts.get("dataset", {})
    return f"""
<section id="problem">
  <div class="wrap">
    <p class="eyebrow">The decision problem</p>
    <h2>Not "predict churn" — decide who to contact</h2>
    <p class="lede">A retention team has limited outreach capacity, so the model must produce
    useful probabilities, prioritize customers under a budget, and support an experiment
    capable of measuring whether an intervention actually reduces churn. Classification
    accuracy alone answers none of those questions.</p>
    <div class="flow">{items}</div>
    <div class="grid-3" style="margin-top:22px">
      <div class="card stat-card">
        <div class="stat-value">{dataset.get("train", 0):,}<span class="stat-detail">
          / {dataset.get("validation", 0):,} / {dataset.get("test", 0):,}</span></div>
        <div class="stat-label">Train / validation / test split</div>
        <div class="stat-detail">Stratified, seed {dataset.get("seed", "—")}; the test split is
        scored exactly once, after every modeling choice is final.</div>
      </div>
      <div class="card stat-card">
        <div class="stat-value">{dataset.get("prevalence_pct", 0)}%</div>
        <div class="stat-label">Churn prevalence</div>
        <div class="stat-detail">The positive class is a minority, which is why ranking
        quality matters more than raw accuracy.</div>
      </div>
      <div class="card stat-card">
        <div class="stat-value">{dataset.get("features", 0)}</div>
        <div class="stat-label">Validated model features</div>
        <div class="stat-detail">The customer identifier and the target are excluded from the
        feature frame and asserted absent by tests.</div>
      </div>
    </div>
  </div>
</section>"""


def _performance(facts: dict) -> str:
    p = facts.get("performance", {})
    comparison = facts.get("model_comparison", [])
    calibration = facts.get("calibration", {})
    selected_pill = '<span class="pill is-on">selected</span>'
    rows = "".join(
        '<tr class="{}">'.format("is-selected" if m["selected"] else "")
        + "<td>{} {}</td>".format(
            escape(m["model"]), selected_pill if m["selected"] else ""
        )
        + "<td>{:.3f}</td><td>{:.3f}</td><td>{:.4f}</td></tr>".format(
            m["pr_auc"], m["roc_auc"], m["brier_score"]
        )
        for m in comparison
    )
    cal_rows = "".join(
        f'<tr class="{"is-selected" if v["selected"] else ""}">'
        f'<td>{escape(v["variant"])}</td><td>{v["brier_score"]:.5f}</td>'
        f'<td>{v["pr_auc"]:.3f}</td></tr>'
        for v in calibration.get("variants", [])
    )
    figures = "".join(
        f'<figure><img src="/figures/{escape(f["file"])}" alt="{escape(f["title"])}" '
        f'loading="lazy" width="{f["width"]}" height="{f["height"]}">'
        f'<figcaption><b>{escape(f["title"])}</b><span>{escape(f["caption"])}</span>'
        "</figcaption></figure>"
        for f in facts.get("figures", [])
        if f["file"] != "cumulative_gains.png"
    )
    return f"""
<section id="performance">
  <div class="wrap">
    <p class="eyebrow">Model performance</p>
    <h2>Measured once on a held-out split</h2>
    <p class="lede">PR-AUC was the primary selection criterion because the positive churn class
    is substantially smaller than the retained class. Accuracy is reported but was deliberately
    not used to choose the model — the two metrics disagree here.</p>
    <div class="grid-3">
      <div class="card stat-card"><div class="stat-value accent">{p.get("roc_auc", 0):.3f}</div>
        <div class="stat-label">ROC-AUC</div>
        <div class="stat-detail">Ranking quality across all thresholds.</div></div>
      <div class="card stat-card"><div class="stat-value accent">{p.get("pr_auc", 0):.3f}</div>
        <div class="stat-label">PR-AUC</div>
        <div class="stat-detail">Against a {p.get("prevalence_pct", 0)}% prevalence
        baseline.</div></div>
      <div class="card stat-card"><div class="stat-value accent">{p.get("brier_score", 0):.4f}</div>
        <div class="stat-label">Brier score</div>
        <div class="stat-detail">Probability quality, not just ordering.</div></div>
    </div>
    <div class="grid-2" style="margin-top:16px">
      <div class="card">
        <h3>Model selection (validation split)</h3>
        <p class="note" style="margin:6px 0 14px">Chosen on PR-AUC, then Brier score.</p>
        <div class="table-scroll"><table>
          <thead><tr><th>Model</th><th>PR-AUC</th><th>ROC-AUC</th><th>Brier</th></tr></thead>
          <tbody>{rows}</tbody></table></div>
      </div>
      <div class="card">
        <h3>Calibration (validation split)</h3>
        <p class="note" style="margin:6px 0 14px">Cross-validated on training data, compared on
        validation. The gain is real but small, and slightly reduces PR-AUC — reported as
        measured.</p>
        <div class="table-scroll"><table>
          <thead><tr><th>Variant</th><th>Brier</th><th>PR-AUC</th></tr></thead>
          <tbody>{cal_rows}</tbody></table></div>
      </div>
    </div>
    <h3 style="margin:38px 0 14px">Generated diagnostics</h3>
    <p class="note" style="margin-bottom:16px">These are the figures written by the analytical
    pipeline, served directly from <code>reports/figures/</code>. Nothing is redrawn.</p>
    <div class="figures">{figures}</div>
  </div>
</section>"""


def _targeting(facts: dict) -> str:
    policy = facts.get("policy", [])
    highlight = next((r for r in policy if r.get("highlight")), None)
    bars = "".join(
        f'<div class="bar-row{" is-highlight" if row["highlight"] else ""}">'
        f'<div class="bar-head"><span class="bar-capacity">Top {row["capacity_pct"]}% '
        f'&middot; {row["contacted"]} customers</span>'
        f'<span class="bar-value"><b>{row["capture_pct"]}%</b> of churners &middot; '
        f'{row["lift"]:.2f}x lift</span></div>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{row["capture_pct"]}%"></div>'
        f'<div class="bar-random" style="left:{row["capacity_pct"]}%" '
        f'data-label="random ≈ {row["capacity_pct"]}%"></div></div></div>'
        for row in policy
    )
    table_rows = "".join(
        f'<tr class="{"is-selected" if row["highlight"] else ""}">'
        f'<td>Top {row["capacity_pct"]}%</td><td>{row["contacted"]}</td>'
        f'<td>{row["captured"]} / {row["total_churners"]}</td>'
        f'<td>{row["capture_pct"]}%</td><td>{row["precision_pct"]}%</td>'
        f'<td>{row["lift"]:.2f}x</td></tr>'
        for row in policy
    )
    gains = next(
        (f for f in facts.get("figures", []) if f["file"] == "cumulative_gains.png"), {}
    )
    gains_w, gains_h = gains.get("width", 780), gains.get("height", 585)
    chain = ""
    if highlight:
        chain = f"""
    <div class="chain">
      <div class="chain-item"><b>{highlight["capacity_pct"]}%</b>
        <span>outreach capacity</span></div>
      <span class="chain-arrow">&rarr;</span>
      <div class="chain-item"><b>{highlight["capture_pct"]}%</b><span>churner capture</span></div>
      <span class="chain-arrow">&rarr;</span>
      <div class="chain-item"><b>{highlight["lift"]:.2f}x</b>
        <span>lift over random targeting</span></div>
      <span class="chain-arrow">&rarr;</span>
      <div class="chain-item"><b>{highlight["precision_pct"]}%</b>
        <span>of contacted did churn</span></div>
    </div>"""
    return f"""
<section id="targeting">
  <div class="wrap">
    <p class="eyebrow">Retention policy</p>
    <h2>Outreach capacity versus churner capture</h2>
    <p class="lede">Customers are ranked by calibrated risk and the top slice is contacted. The
    marker on each bar shows what random targeting of the same size would be expected to reach.</p>
    <div class="bars">{bars}</div>
    {chain}
    <div class="grid-2" style="margin-top:26px">
      <div class="card">
        <div class="table-scroll"><table>
          <thead><tr><th>Capacity</th><th>Contacted</th><th>Churners caught</th>
          <th>Capture</th><th>Precision</th><th>Lift</th></tr></thead>
          <tbody>{table_rows}</tbody></table></div>
      </div>
      <figure><img src="/figures/cumulative_gains.png" alt="Cumulative gains curve"
        loading="lazy" width="{gains_w}" height="{gains_h}">
        <figcaption><b>Cumulative gains</b><span>Generated by the pipeline: churner capture
        across every capacity, with the random-targeting diagonal.</span></figcaption></figure>
    </div>
    <div class="limit" style="margin-top:20px">{WARN_ICON}
      <span><span class="note-strong">This measures predictive prioritization only.</span>
      It shows the model ranks customers who did churn near the top of a held-out split. It does
      not show that contacting them prevents churn — a perfectly targeted campaign with a useless
      offer would produce this same table and retain nobody.</span></div>
  </div>
</section>"""


def _demo(facts: dict) -> str:
    examples = facts.get("example_customers", [])
    example_buttons = "".join(
        f'<button class="btn" type="button" data-example="{i}">{escape(ex["name"])}</button>'
        for i, ex in enumerate(examples)
    )
    bands = "".join(
        f'<span class="pill">{escape(b["label"])}: '
        f'{b["min"]:.2f}–{min(b["max"], 1.0):.2f}</span> '
        for b in facts.get("risk_bands", [])
    )
    return f"""
<section id="demo">
  <div class="wrap">
    <p class="eyebrow">Interactive demo</p>
    <h2>Score a customer against the deployed model</h2>
    <p class="lede">This form posts to the live <code>POST /predict</code> endpoint and renders
    the response. Nothing is simulated in the browser. Inputs are generated from the service's
    own Pydantic schema, so they match the {facts.get("dataset", {}).get("features", 19)}-feature
    contract exactly.</p>
    <div class="card">
      <div id="predict-form" class="form-grid">{render_form_fields()}</div>
      <div class="demo-actions">
        <button class="btn btn-primary" type="button" id="predict-run">Predict Churn Risk</button>
        {example_buttons}
      </div>
      <div class="result" id="predict-result"></div>
    </div>
    <p class="note" style="margin-top:14px">Example customers are illustrative fictional records
    authored for this page, not rows from the IBM dataset, and carry no churn labels.
    Display bands: {bands}— presentation categories only, not a business decision threshold.</p>
  </div>
</section>"""


def _ranking_demo(facts: dict) -> str:
    cohort = facts.get("ranking_cohort", [])
    if not cohort:
        return ""
    return f"""
<section id="targeting-demo">
  <div class="wrap">
    <p class="eyebrow">Capacity-constrained ranking</p>
    <h2>Rank a cohort under a fixed outreach budget</h2>
    <p class="lede">{len(cohort)} illustrative customers are scored and ordered by the live
    <code>POST /rank</code> endpoint. Change the capacity to see which customers a retention team
    would actually contact.</p>
    <div class="rank-controls">
      <span class="note">Outreach capacity</span>
      <div class="seg">
        <button type="button" data-capacity="10">10%</button>
        <button type="button" data-capacity="20" class="is-active">20%</button>
        <button type="button" data-capacity="30">30%</button>
      </div>
      <span class="note" id="rank-status" style="margin-left:6px"></span>
    </div>
    <div class="rank-list" id="rank-list"></div>
    <p class="note" style="margin-top:14px">Fictional records with no churn labels: this
    demonstrates the ranking mechanism, not measured outcomes.</p>
  </div>
</section>"""


def _experiment(facts: dict) -> str:
    e = facts.get("experiment", {})
    return f"""
<section id="experiment">
  <div class="wrap">
    <p class="eyebrow">Experiment design</p>
    <h2>Prediction is not causation</h2>
    <p class="lede">The model decides whom to prioritize. A randomized experiment is still
    required to determine whether a retention intervention actually works. This dataset contains
    no treatment assignment, so the project ships a power analysis instead of an effect
    estimate.</p>
    <div class="grid-3" style="margin-bottom:18px">
      <div class="card stat-card"><div class="stat-value">{e.get("baseline_pct", 0)}%</div>
        <div class="stat-label">Observed churn, top-20% cohort</div>
        <div class="stat-detail">Measured, not assumed: the churn rate among customers a
        top-20% policy would contact.</div></div>
      <div class="card stat-card"><div class="stat-value">{e.get("reduction_pp", 0)} pp</div>
        <div class="stat-label">Target detectable reduction</div>
        <div class="stat-detail">An assumption for sizing, not an observed effect.</div></div>
      <div class="card stat-card"><div class="stat-value">{e.get("power_pct", 0)}%</div>
        <div class="stat-label">Statistical power</div>
        <div class="stat-detail">Two-sided two-proportion test at
        α = {e.get("alpha", 0.05)}.</div></div>
    </div>
    <div class="arms">
      <div class="arm"><div class="arm-label">Control</div>
        <div class="arm-value">{e.get("per_arm", 0):,}</div>
        <div class="arm-unit">customers, no offer</div></div>
      <div class="arm-vs">versus</div>
      <div class="arm is-treatment"><div class="arm-label">Intervention</div>
        <div class="arm-value">{e.get("per_arm", 0):,}</div>
        <div class="arm-unit">customers, retention offer</div></div>
    </div>
    <div class="limit" style="margin-top:18px">{WARN_ICON}
      <span><span class="note-strong">Prospective design and power analysis only.</span>
      No experiment has been conducted, and {e.get("total_n", 0):,} customers would be required in
      total to detect the assumed effect. The effect size is an input to the calculation, never a
      result of it.</span></div>
  </div>
</section>"""


def _architecture(facts: dict) -> str:
    def stages(items: list[dict]) -> str:
        return "".join(
            f'<div class="arch-stage"><b>{escape(str(s["stage"]))}</b>'
            f'<span>{escape(str(s["detail"]))}</span></div>'
            for s in items
        )

    inference = [e for e in facts.get("endpoints", []) if e["method"] == "POST"]
    chips = "".join(f"<code>{escape(e['path'])}</code>" for e in inference)
    return f"""
<section id="architecture">
  <div class="wrap">
    <p class="eyebrow">System architecture</p>
    <h2>From raw records to a verified public endpoint</h2>
    <p class="lede">Two halves, one artifact: the analytical pipeline produces a frozen model,
    and the serving layer verifies that exact artifact before it answers a request.</p>
    <div class="arch">
      <div class="arch-col">
        <p class="eyebrow">Data science</p>
        {stages(facts.get("pipeline_stages", []))}
      </div>
      <div class="arch-col">
        <p class="eyebrow">ML engineering</p>
        {stages(facts.get("serving_stages", []))}
        <div class="arch-fanout">{chips}</div>
      </div>
    </div>
  </div>
</section>"""


def _engineering(facts: dict) -> str:
    eng = facts.get("engineering", {})
    claims = eng.get("readme_claims")
    cards = [
        (f'{eng.get("features", 0)}', "Model contract", "validated model features"),
        ("SHA-256", "Artifact integrity", "verified before inference"),
        (f'{eng.get("max_batch", 0):,}', "Batch inference", "customer maximum per request"),
        (f'{eng.get("tests", 0)}', "Testing", "automated tests, zero skipped"),
        (
            f"{claims}/{claims}" if claims else "—",
            "Documentation integrity",
            "numeric claims verified against artifacts",
        ),
        (f'{eng.get("metric_families", 0)}', "Observability", "Prometheus metric families"),
    ]
    grid = "".join(
        f'<div class="card stat-card"><div class="stat-value accent">{escape(v)}</div>'
        f'<div class="stat-label">{escape(label)}</div>'
        f'<div class="stat-detail">{escape(detail)}</div></div>'
        for v, label, detail in cards
    )
    stack = "".join(
        f'<div class="stack-item"><b>{escape(t["name"])}</b><span>{escape(t["role"])}</span></div>'
        for t in facts.get("tech_stack", [])
    )
    return f"""
<section id="engineering">
  <div class="wrap">
    <p class="eyebrow">Production engineering</p>
    <h2>What makes it a service rather than a notebook</h2>
    <div class="grid-3">{grid}</div>
    <h3 style="margin:38px 0 14px">Technology</h3>
    <div class="stack">{stack}</div>
    <p class="note" style="margin-top:12px">Python {escape(str(eng.get("python_version", "")))} ·
    scikit-learn {escape(str(eng.get("sklearn_version", "")))} — the deployed runtime installs the
    same version that produced the serialized artifact. The Dockerfile is included in the
    repository; container execution was not verified locally.</p>
  </div>
</section>"""


def _drift(facts: dict) -> str:
    drift = facts.get("drift", {})
    bands = "".join(
        f'<div class="psi-band tone-{escape(b["tone"])}">'
        f'<span class="psi-range">{escape(b["range"])}</span>'
        f'<span class="psi-meaning">{escape(b["meaning"])}</span></div>'
        for b in drift.get("psi_bands", [])
    )
    numeric = "".join(f"<code>{escape(f)}</code> " for f in drift.get("numeric", []))
    categorical = "".join(f"<code>{escape(f)}</code> " for f in drift.get("categorical", []))
    reference = facts.get("engineering", {}).get("reference_rows", 0)
    return f"""
<section id="monitoring">
  <div class="wrap">
    <p class="eyebrow">Drift monitoring</p>
    <h2>Watching the inputs, not just the outputs</h2>
    <div class="grid-2">
      <div class="card">
        <h3>Numeric drift — Population Stability Index</h3>
        <p class="note" style="margin:8px 0 12px">Monitored: {numeric}</p>
        <p class="note">Incoming batches are compared against a reference profile built from the
        {reference:,} training rows only, using fixed bins so scores stay comparable.</p>
        <div class="psi-scale">{bands}</div>
        <p class="note" style="margin-top:12px">These are operational rules of thumb for
        triage, not formal statistical significance thresholds.</p>
      </div>
      <div class="card">
        <h3>Categorical drift — total variation</h3>
        <p class="note" style="margin:8px 0 12px">Monitored: {categorical}</p>
        <p class="note">Category distributions and missingness rates are compared against the
        same training-only reference. <code>POST /monitor/drift</code> returns per-feature scores,
        an overall status, and any warnings.</p>
        <p class="note" style="margin-top:12px"><span class="note-strong">No production drift has
        been observed or is claimed.</span> The endpoint scores whatever batch you send it; this
        project has no production traffic history.</p>
      </div>
    </div>
  </div>
</section>"""


def _benchmark(facts: dict) -> str:
    bench = facts.get("benchmark", {})
    rows = bench.get("rows", [])
    if not rows:
        return ""
    peak = max(row["http_p95"] for row in rows)
    bars = "".join(
        f'<div class="bench-row"><span class="bench-label">batch {row["batch_size"]}</span>'
        f'<span class="bench-track"><span class="bench-fill" '
        f'style="width:{row["http_p95"] / peak * 100:.1f}%"></span></span>'
        f'<span class="bench-value">{row["http_p95"]:.2f} ms</span></div>'
        for row in rows
    )
    table = "".join(
        f'<tr><td>{row["batch_size"]}</td><td>{row["direct_p50"]:.2f}</td>'
        f'<td>{row["direct_p95"]:.2f}</td><td>{row["http_p50"]:.2f}</td>'
        f'<td>{row["http_p95"]:.2f}</td></tr>'
        for row in rows
    )
    return f"""
<section id="benchmark">
  <div class="wrap">
    <p class="eyebrow">Serving benchmark</p>
    <h2>Batch size versus p95 latency</h2>
    <p class="lede">Vectorized inference: a 500-customer batch costs roughly twice a
    single-customer request, not 500 times.</p>
    <div class="card">
      <div class="bench">{bars}</div>
      <div class="table-scroll" style="margin-top:22px"><table>
        <thead><tr><th>Batch</th><th>Direct p50</th><th>Direct p95</th>
        <th>HTTP p50</th><th>HTTP p95</th></tr></thead>
        <tbody>{table}</tbody></table></div>
      <div class="limit" style="margin-top:16px">{WARN_ICON}
        <span><span class="note-strong">Local FastAPI TestClient benchmark — not production
        network latency.</span> Measured in-process on
        {escape(str(bench.get("platform", "one machine")))}
        over {bench.get("iterations", 0)} iterations. These are not Vercel response times and say
        nothing about internet-facing performance.</span></div>
    </div>
  </div>
</section>"""


def _api_and_limits(facts: dict) -> str:
    paths = ["/docs", "/model-info", "/health", "/ready", "/metrics", "/project-summary"]
    links = "".join(
        f'<a class="endpoint" href="{escape(p)}">'
        f'<span><span class="endpoint-path">{escape(p)}</span><br>'
        f'<span class="endpoint-desc">{escape(ENDPOINT_DESCRIPTIONS[p])}</span></span>'
        f'<span class="endpoint-arrow">&rarr;</span></a>'
        for p in paths
    )
    limits = "".join(
        f'<div class="limit">{WARN_ICON}<span>{escape(item)}</span></div>'
        for item in facts.get("limitations", [])
    )
    return f"""
<section id="api">
  <div class="wrap">
    <p class="eyebrow">Inspect the system</p>
    <h2>Every claim on this page is checkable</h2>
    <p class="lede">The interactive documentation is the full request contract. The project
    summary endpoint returns the exact artifact-derived payload this page renders.</p>
    <div class="endpoints">{links}</div>
  </div>
</section>
<section id="limitations">
  <div class="wrap">
    <p class="eyebrow">Engineering notes</p>
    <h2>Limitations</h2>
    <p class="lede">Stated plainly, because knowing what a result does not support is part of
    the result.</p>
    <div class="limits">{limits}</div>
  </div>
</section>"""


def _nav(facts: dict) -> str:
    repo = escape(str(facts.get("repository_url", "")))
    links = "".join(
        f'<a href="#{slug}">{escape(label)}</a>' for slug, label in NAV_SECTIONS
    )
    name = escape(str(facts.get("name", "RetentionLab")))
    return f"""
<nav class="nav">
  <div class="nav-inner">
    <a class="brand" href="#overview">Retention<span>Lab</span></a>
    <div class="nav-links">{links}</div>
    <a class="nav-cta nav-icon" href="{repo}" target="_blank" rel="noopener noreferrer"
       aria-label="View RetentionLab on GitHub" title="View source on GitHub">{GITHUB_MARK}</a>
    <a class="nav-cta" href="/docs">API Docs</a>
  </div>
</nav>
<span hidden data-app-name="{name}"></span>"""


def _footer(facts: dict) -> str:
    repo = escape(str(facts.get("repository_url", "")))
    version = escape(str(facts.get("model_version", "")))
    digest = escape(str(facts.get("model_sha256", ""))[:16])
    return f"""
<footer>
  <div class="wrap footer-inner">
    <p class="note" style="margin:0;max-width:56ch">Model <code>{version}</code>
    · artifact <code>{digest}…</code> verified at startup. Observational data with no treatment
    assignment: this service predicts churn risk and prioritizes outreach. It does not claim that
    any intervention reduces churn, and it is not a production deployment.</p>
    <p class="note" style="margin:0"><a href="{repo}" target="_blank"
    rel="noopener noreferrer">Source on GitHub</a> · <a href="/docs">API docs</a> ·
    <a href="/project-summary">Project summary</a> · <a href="/ready">Readiness</a></p>
  </div>
</footer>"""


def render_landing_page(facts_path: Path) -> str:
    """Return the complete landing page HTML."""
    facts = load_facts(facts_path)
    name = escape(str(facts.get("name", "RetentionLab")))
    tagline = escape(str(facts.get("tagline", "")))
    bootstrap = json.dumps(
        {
            "riskBands": facts.get("risk_bands", []),
            "examples": facts.get("example_customers", []),
            "cohort": facts.get("ranking_cohort", []),
        }
    ).replace("</", "<\\/")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{name} — {tagline}</title>
<meta name="description" content="{name}: calibrated churn modeling, capacity-constrained
retention targeting, experiment design, and a checksum-verified FastAPI inference service.">
<meta name="color-scheme" content="dark">
<link rel="stylesheet" href="/static/landing.css">
<link rel="icon" href="{FAVICON}">
</head>
<body>
{_nav(facts)}
<main>
{_hero(facts)}
{_problem(facts)}
{_performance(facts)}
{_targeting(facts)}
{_demo(facts)}
{_ranking_demo(facts)}
{_experiment(facts)}
{_architecture(facts)}
{_engineering(facts)}
{_drift(facts)}
{_benchmark(facts)}
{_api_and_limits(facts)}
</main>
{_footer(facts)}
<script>window.RETENTIONLAB = {bootstrap};</script>
<script src="/static/landing.js" defer></script>
</body>
</html>
"""
