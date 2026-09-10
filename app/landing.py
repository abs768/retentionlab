"""Recruiter-facing landing page rendered from verified artifact facts.

Plain HTML/CSS with a small vanilla-JS demo form. No frontend framework and no
build step: the page is a single string produced from
``artifacts/landing/landing_facts.json``, which
``scripts/build_landing_facts.py`` derives from the canonical analytical
outputs. Nothing here re-computes or restates a metric independently.
"""

from __future__ import annotations

import json
from functools import lru_cache
from html import escape
from pathlib import Path

DEMO_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 94.4,
    "TotalCharges": 188.8,
}

FALLBACK_FACTS = {
    "name": "RetentionLab",
    "tagline": "End-to-End Customer Retention Analytics Platform",
    "headline_metrics": [],
    "capabilities": [],
}


@lru_cache(maxsize=4)
def load_facts(path: Path) -> dict:
    """Read the generated landing facts, degrading to a minimal header if absent."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return dict(FALLBACK_FACTS)


def _metric_cards(facts: dict) -> str:
    cards = []
    for metric in facts.get("headline_metrics", []):
        cards.append(
            '<div class="card">'
            f'<div class="value">{escape(str(metric["value"]))}</div>'
            f'<div class="label">{escape(str(metric["label"]))}</div>'
            f'<div class="detail">{escape(str(metric.get("detail", "")))}</div>'
            "</div>"
        )
    return "\n".join(cards)


def _capability_chips(facts: dict) -> str:
    return "\n".join(
        f'<span class="chip">{escape(str(item))}</span>'
        for item in facts.get("capabilities", [])
    )


def render_landing_page(facts_path: Path) -> str:
    """Return the complete landing page HTML."""
    facts = load_facts(facts_path)
    name = escape(str(facts.get("name", "RetentionLab")))
    tagline = escape(str(facts.get("tagline", "")))
    model_version = escape(str(facts.get("model_version", "")))
    experiment_note = escape(str(facts.get("experiment_note", "")))
    demo_json = json.dumps(DEMO_CUSTOMER)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{name} — {tagline}</title>
<meta name="description" content="{name}: {tagline}. Calibrated churn modeling,
 capacity-constrained retention targeting, experiment design, and a verified
 FastAPI inference service.">
<style>
  :root {{
    --bg: #0f1420; --panel: #171e2e; --line: #26314a;
    --text: #e8edf7; --muted: #96a3bd; --accent: #5b9dff; --accent-2: #35d0a5;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--text);
    font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
          Helvetica, Arial, sans-serif;
  }}
  .wrap {{ max-width: 1060px; margin: 0 auto; padding: 48px 20px 64px; }}
  header {{ border-bottom: 1px solid var(--line); padding-bottom: 28px; margin-bottom: 32px; }}
  h1 {{ margin: 0 0 6px; font-size: 2.1rem; letter-spacing: -0.02em; }}
  .tagline {{ color: var(--muted); font-size: 1.08rem; margin: 0 0 20px; }}
  .actions {{ display: flex; flex-wrap: wrap; gap: 10px; }}
  .btn {{
    display: inline-block; padding: 9px 16px; border-radius: 8px; text-decoration: none;
    font-weight: 600; font-size: 0.92rem; border: 1px solid var(--line);
    color: var(--text); background: var(--panel);
  }}
  .btn.primary {{ background: var(--accent); border-color: var(--accent); color: #08101f; }}
  h2 {{ font-size: 1.05rem; text-transform: uppercase; letter-spacing: 0.08em;
       color: var(--muted); margin: 36px 0 14px; font-weight: 600; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; }}
  .card {{ background: var(--panel); border: 1px solid var(--line);
          border-radius: 10px; padding: 18px; }}
  .value {{ font-size: 1.95rem; font-weight: 700; color: var(--accent-2);
           letter-spacing: -0.02em; }}
  .label {{ font-size: 0.93rem; margin-top: 4px; }}
  .detail {{ font-size: 0.8rem; color: var(--muted); margin-top: 6px; }}
  .chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .chip {{ background: var(--panel); border: 1px solid var(--line); border-radius: 999px;
          padding: 6px 13px; font-size: 0.86rem; color: var(--text); }}
  .panel {{ background: var(--panel); border: 1px solid var(--line);
           border-radius: 10px; padding: 20px; }}
  form {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }}
  label {{ display: block; font-size: 0.78rem; color: var(--muted); margin-bottom: 4px; }}
  select, input {{
    width: 100%; padding: 8px 10px; border-radius: 7px; border: 1px solid var(--line);
    background: #101728; color: var(--text); font-size: 0.9rem;
  }}
  .run {{ margin-top: 14px; }}
  button {{
    padding: 10px 20px; border-radius: 8px; border: none; background: var(--accent);
    color: #08101f; font-weight: 700; font-size: 0.93rem; cursor: pointer;
  }}
  button:disabled {{ opacity: 0.55; cursor: default; }}
  #result {{ margin-top: 14px; font-size: 0.95rem; min-height: 22px; }}
  .prob {{ font-size: 1.5rem; font-weight: 700; color: var(--accent-2); }}
  .note {{ color: var(--muted); font-size: 0.85rem; }}
  footer {{ margin-top: 40px; border-top: 1px solid var(--line); padding-top: 20px;
           color: var(--muted); font-size: 0.84rem; }}
  code {{ background: #101728; padding: 1px 5px; border-radius: 4px; font-size: 0.85em; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>{name}</h1>
    <p class="tagline">{tagline}</p>
    <div class="actions">
      <a class="btn primary" href="/docs">API Docs</a>
      <a class="btn" href="/model-info">Model Info</a>
      <a class="btn" href="/health">Health</a>
      <a class="btn" href="/ready">Readiness</a>
    </div>
  </header>

  <h2>Verified Results</h2>
  <div class="grid">
{_metric_cards(facts)}
  </div>
  <p class="note" style="margin-top:12px">
    Metrics come from a single held-out test split of IBM's fictional Telco sample and are
    generated from <code>reports/run_summary.json</code>. Lift describes retrospective ranking
    quality, not the causal effect of contacting anyone. {experiment_note}
  </p>

  <h2>Capabilities</h2>
  <div class="chips">
{_capability_chips(facts)}
  </div>

  <h2>Live Prediction</h2>
  <div class="panel">
    <p class="note" style="margin-top:0">
      Calls this service's <code>POST /predict</code> with the 19-feature contract.
      The churn label is never an input.
    </p>
    <form id="demo">
      <div>
        <label for="Contract">Contract</label>
        <select id="Contract" name="Contract">
          <option>Month-to-month</option><option>One year</option><option>Two year</option>
        </select>
      </div>
      <div>
        <label for="InternetService">Internet Service</label>
        <select id="InternetService" name="InternetService">
          <option>Fiber optic</option><option>DSL</option><option>No</option>
        </select>
      </div>
      <div>
        <label for="tenure">Tenure (months)</label>
        <input id="tenure" name="tenure" type="number" min="0" step="1" value="2">
      </div>
      <div>
        <label for="MonthlyCharges">Monthly Charges</label>
        <input id="MonthlyCharges" name="MonthlyCharges" type="number"
               min="0" step="0.05" value="94.40">
      </div>
      <div>
        <label for="PaymentMethod">Payment Method</label>
        <select id="PaymentMethod" name="PaymentMethod">
          <option>Electronic check</option><option>Mailed check</option>
          <option>Bank transfer (automatic)</option><option>Credit card (automatic)</option>
        </select>
      </div>
      <div>
        <label for="TechSupport">Tech Support</label>
        <select id="TechSupport" name="TechSupport">
          <option>No</option><option>Yes</option><option>No internet service</option>
        </select>
      </div>
    </form>
    <div class="run"><button id="go" type="button">Predict churn probability</button></div>
    <div id="result"></div>
  </div>

  <footer>
    Model <code>{model_version}</code>. Observational data with no treatment assignment: this
    service predicts churn risk and prioritizes outreach, and does not claim that any
    intervention reduces churn. Not a production deployment.
  </footer>
</div>
<script>
(function () {{
  var base = {demo_json};
  var button = document.getElementById("go");
  var output = document.getElementById("result");

  button.addEventListener("click", function () {{
    var payload = Object.assign({{}}, base);
    var form = document.getElementById("demo");
    payload.Contract = form.Contract.value;
    payload.InternetService = form.InternetService.value;
    payload.PaymentMethod = form.PaymentMethod.value;
    payload.TechSupport = form.TechSupport.value;
    payload.tenure = parseInt(form.tenure.value, 10);
    payload.MonthlyCharges = parseFloat(form.MonthlyCharges.value);
    payload.TotalCharges = Math.round(payload.tenure * payload.MonthlyCharges * 100) / 100;

    if (payload.InternetService === "No") {{
      ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
       "StreamingTV", "StreamingMovies"].forEach(function (key) {{
        payload[key] = "No internet service";
      }});
    }}

    button.disabled = true;
    output.textContent = "Scoring…";
    fetch("/predict", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify(payload)
    }})
      .then(function (response) {{
        return response.json().then(function (body) {{
          return {{ ok: response.ok, status: response.status, body: body }};
        }});
      }})
      .then(function (result) {{
        if (!result.ok) {{
          output.textContent = "Service returned HTTP " + result.status + ".";
          return;
        }}
        var probability = result.body.churn_probability;
        output.innerHTML =
          '<span class="prob">' + (probability * 100).toFixed(1) + "%</span>" +
          ' <span class="note">calibrated churn probability · model ' +
          result.body.model_version + "</span>";
      }})
      .catch(function () {{
        output.textContent = "Request failed.";
      }})
      .finally(function () {{
        button.disabled = false;
      }});
  }});
}})();
</script>
</body>
</html>
"""
