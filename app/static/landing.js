/* RetentionLab landing page interactions.
 *
 * Both demos call the real service. No probability is ever computed, cached,
 * or approximated in the browser: every number shown comes from a response to
 * POST /predict or POST /rank.
 */
(function () {
  "use strict";

  var config = window.RETENTIONLAB || {};
  var riskBands = config.riskBands || [];
  var examples = config.examples || [];
  var cohort = config.cohort || [];

  function $(id) { return document.getElementById(id); }

  function bandFor(probability) {
    for (var i = 0; i < riskBands.length; i += 1) {
      if (probability >= riskBands[i].min && probability < riskBands[i].max) {
        return riskBands[i];
      }
    }
    return riskBands[riskBands.length - 1] || { label: "", tone: "warning" };
  }

  function postJSON(path, payload) {
    return fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then(function (response) {
      return response.json().catch(function () { return null; }).then(function (body) {
        return { ok: response.ok, status: response.status, body: body };
      });
    });
  }

  /* ---------- Single-customer prediction ---------- */

  var form = $("predict-form");

  function readForm() {
    var payload = {};
    var elements = form.querySelectorAll("[data-field]");
    for (var i = 0; i < elements.length; i += 1) {
      var el = elements[i];
      var name = el.getAttribute("data-field");
      var kind = el.getAttribute("data-kind");
      if (kind === "int") {
        payload[name] = parseInt(el.value, 10);
      } else if (kind === "float") {
        payload[name] = Math.round(parseFloat(el.value) * 100) / 100;
      } else {
        payload[name] = el.value;
      }
    }
    // TotalCharges is derived so the submitted record stays internally consistent.
    payload.TotalCharges = Math.round(payload.tenure * payload.MonthlyCharges * 100) / 100;
    return payload;
  }

  function applyCustomer(customer) {
    var elements = form.querySelectorAll("[data-field]");
    for (var i = 0; i < elements.length; i += 1) {
      var el = elements[i];
      var name = el.getAttribute("data-field");
      if (Object.prototype.hasOwnProperty.call(customer, name)) {
        el.value = customer[name];
        if (el.type === "range") { syncRange(el); }
      }
    }
    syncDerived();
  }

  function syncRange(el) {
    var output = document.querySelector('[data-range-for="' + el.getAttribute("data-field") + '"]');
    if (output) {
      var prefix = el.getAttribute("data-prefix") || "";
      var suffix = el.getAttribute("data-suffix") || "";
      output.textContent = prefix + el.value + suffix;
    }
  }

  function syncDerived() {
    var tenure = parseInt(form.querySelector('[data-field="tenure"]').value, 10);
    var monthly = parseFloat(form.querySelector('[data-field="MonthlyCharges"]').value);
    var total = $("derived-total");
    if (total) {
      total.textContent = "$" + (Math.round(tenure * monthly * 100) / 100).toFixed(2);
    }
  }

  function renderPrediction(result) {
    var box = $("predict-result");
    box.classList.add("is-visible");
    if (!result.ok) {
      var detail = result.body && result.body.detail;
      var message = typeof detail === "string" ? detail : "the request was rejected";
      box.innerHTML = '<p class="error-text">Service returned HTTP ' + result.status +
        " — " + message + ".</p>";
      return;
    }
    var probability = result.body.churn_probability;
    var percent = (probability * 100).toFixed(1);
    var band = bandFor(probability);

    box.innerHTML =
      '<div class="result-head">' +
        '<div class="result-prob">' + percent + "%</div>" +
        '<span class="pill is-on">' + band.label + " band</span>" +
      "</div>" +
      '<div class="result-meter"><div class="result-meter-fill band-' + band.tone + '"></div></div>' +
      '<p class="note">Calibrated churn probability returned by <code>POST /predict</code>. ' +
      "Bands are display categories for this demo only — the probability is the " +
      "authoritative model output and no business threshold is implied.</p>" +
      '<dl class="result-meta">' +
        "<div><dt>Model</dt><dd>" + result.body.model_version + "</dd></div>" +
        "<div><dt>Probability</dt><dd>" + probability.toFixed(6) + "</dd></div>" +
        "<div><dt>Customer ID</dt><dd>" + (result.body.customer_id || "not supplied") + "</dd></div>" +
      "</dl>";

    // Fill after paint so the width transition runs.
    window.requestAnimationFrame(function () {
      var fill = box.querySelector(".result-meter-fill");
      if (fill) { fill.style.width = Math.max(probability * 100, 1.5) + "%"; }
    });
  }

  if (form) {
    form.addEventListener("input", function (event) {
      if (event.target.type === "range") { syncRange(event.target); }
      syncDerived();
    });

    var ranges = form.querySelectorAll('input[type="range"]');
    for (var r = 0; r < ranges.length; r += 1) { syncRange(ranges[r]); }
    syncDerived();

    $("predict-run").addEventListener("click", function () {
      var button = $("predict-run");
      var box = $("predict-result");
      button.disabled = true;
      box.classList.add("is-visible");
      box.innerHTML = '<p class="note">Scoring against the deployed model…</p>';
      postJSON("/predict", readForm())
        .then(renderPrediction)
        .catch(function () {
          box.innerHTML = '<p class="error-text">Request failed. The service may be ' +
            "starting up — try again in a moment.</p>";
        })
        .then(function () { button.disabled = false; });
    });

    var exampleButtons = document.querySelectorAll("[data-example]");
    for (var e = 0; e < exampleButtons.length; e += 1) {
      exampleButtons[e].addEventListener("click", function (event) {
        var index = parseInt(event.currentTarget.getAttribute("data-example"), 10);
        if (examples[index]) { applyCustomer(examples[index].customer); }
      });
    }
  }

  /* ---------- Capacity-constrained ranking ---------- */

  var rankList = $("rank-list");
  var rankStatus = $("rank-status");
  var rankButtons = document.querySelectorAll("[data-capacity]");

  function renderRanking(result, capacityPct) {
    if (!result.ok || !result.body || !result.body.customers) {
      rankList.innerHTML = "";
      rankStatus.innerHTML = '<span class="error-text">Ranking request failed (HTTP ' +
        result.status + ").</span>";
      return;
    }
    var rows = result.body.customers;
    var maximum = rows.reduce(function (acc, row) {
      return Math.max(acc, row.churn_probability);
    }, 0.0001);

    rankList.innerHTML = rows.map(function (row) {
      var width = Math.max((row.churn_probability / maximum) * 100, 2);
      return '<div class="rank-row' + (row.selected_for_outreach ? " is-selected" : "") + '">' +
        '<span class="rank-num">#' + row.rank + "</span>" +
        '<span class="rank-id">' + (row.customer_id || "—") + "</span>" +
        '<span class="rank-bar"><span class="rank-bar-fill" style="width:' + width + '%"></span></span>' +
        '<span class="rank-prob">' + (row.churn_probability * 100).toFixed(1) + "%</span>" +
        '<span class="rank-flag">' +
          (row.selected_for_outreach ? "Selected" : "Not contacted") + "</span>" +
      "</div>";
    }).join("");

    rankStatus.innerHTML = "<b>" + result.body.selected_count + "</b> of <b>" +
      result.body.total_customers + "</b> illustrative customers selected at " +
      capacityPct + "% outreach capacity, ranked by <code>POST /rank</code>.";
  }

  function runRanking(capacityPct) {
    for (var i = 0; i < rankButtons.length; i += 1) {
      rankButtons[i].classList.toggle(
        "is-active",
        rankButtons[i].getAttribute("data-capacity") === String(capacityPct)
      );
    }
    rankStatus.textContent = "Ranking " + cohort.length + " customers…";
    var payload = {
      customers: cohort.map(function (entry) {
        return Object.assign({ customerID: entry.id }, entry.customer);
      }),
      capacity_fraction: capacityPct / 100
    };
    postJSON("/rank", payload)
      .then(function (result) { renderRanking(result, capacityPct); })
      .catch(function () {
        rankStatus.innerHTML = '<span class="error-text">Ranking request failed.</span>';
      });
  }

  if (rankList && cohort.length) {
    for (var b = 0; b < rankButtons.length; b += 1) {
      rankButtons[b].addEventListener("click", function (event) {
        runRanking(parseInt(event.currentTarget.getAttribute("data-capacity"), 10));
      });
    }
    // Only fetch once the section is actually reached, to avoid a cold-start
    // request the visitor never sees.
    var section = $("targeting-demo");
    if ("IntersectionObserver" in window && section) {
      var observer = new IntersectionObserver(function (entries) {
        if (entries[0].isIntersecting) {
          observer.disconnect();
          runRanking(20);
        }
      }, { rootMargin: "160px" });
      observer.observe(section);
    } else {
      runRanking(20);
    }
  }
})();
