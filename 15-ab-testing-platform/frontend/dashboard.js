/**
 * ModelLab Dashboard - Experiment management and visualization.
 *
 * Handles experiment CRUD, real-time result fetching, and Chart.js
 * visualizations for traffic splits, confidence intervals, and
 * Bayesian posterior distributions.
 */

const API_BASE = window.location.origin;

// Chart instances (destroy before re-creating)
let trafficChart = null;
let ciChart = null;
let posteriorChart = null;

// Current selected experiment
let selectedExperimentId = null;

// ----------------------------------------------------------------
// Initialization
// ----------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  refreshAll();
  // Auto-refresh every 15 seconds
  setInterval(refreshAll, 15000);
});

async function refreshAll() {
  await loadExperiments();
  if (selectedExperimentId) {
    await loadExperimentDetail(selectedExperimentId);
  }
}

// ----------------------------------------------------------------
// API Calls
// ----------------------------------------------------------------

async function apiCall(path, options = {}) {
  try {
    const response = await fetch(`${API_BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!response.ok) {
      const error = await response
        .json()
        .catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }
    return await response.json();
  } catch (err) {
    console.error(`API error (${path}):`, err);
    throw err;
  }
}

// ----------------------------------------------------------------
// Experiment List
// ----------------------------------------------------------------

async function loadExperiments() {
  try {
    const stateFilter = document.getElementById("state-filter").value;
    const queryParam = stateFilter ? `?state=${stateFilter}` : "";
    const experiments = await apiCall(`/experiments${queryParam}`);
    renderExperimentList(experiments);
    updateStatusBar(experiments);
  } catch (err) {
    console.error("Failed to load experiments:", err);
  }
}

function renderExperimentList(experiments) {
  const container = document.getElementById("experiments-list");

  if (!experiments || experiments.length === 0) {
    container.innerHTML =
      '<p class="empty-state">No experiments found. Create one to get started.</p>';
    return;
  }

  container.innerHTML = experiments
    .map((exp) => {
      const variants = exp.variants || [];
      const totalSamples = Object.values(exp.sample_counts || {}).reduce(
        (a, b) => a + b,
        0,
      );

      return `
            <div class="experiment-card" onclick="selectExperiment('${exp.id}')">
                <div class="card-header">
                    <span class="card-name">${escapeHtml(exp.name)}</span>
                    <span class="state-badge ${exp.state}">${exp.state}</span>
                </div>
                <div class="card-variants">
                    ${variants
                      .map(
                        (v) => `
                        <span class="variant-tag ${v.is_control ? "control" : "treatment"}">
                            ${escapeHtml(v.name)} (${v.traffic_percentage}%)
                        </span>
                    `,
                      )
                      .join("")}
                </div>
                <div class="card-meta">
                    <span>Samples: ${totalSamples.toLocaleString()}</span>
                    <span>${formatDate(exp.created_at)}</span>
                </div>
            </div>
        `;
    })
    .join("");
}

function updateStatusBar(experiments) {
  const total = experiments.length;
  const running = experiments.filter((e) => e.state === "running").length;
  const completed = experiments.filter((e) => e.state === "completed").length;
  const totalEvents = experiments.reduce(
    (sum, e) =>
      sum + Object.values(e.sample_counts || {}).reduce((a, b) => a + b, 0),
    0,
  );

  document.getElementById("total-experiments").textContent = total;
  document.getElementById("running-experiments").textContent = running;
  document.getElementById("completed-experiments").textContent = completed;
  document.getElementById("total-events").textContent =
    totalEvents.toLocaleString();
}

function filterExperiments() {
  loadExperiments();
}

// ----------------------------------------------------------------
// Experiment Detail
// ----------------------------------------------------------------

async function selectExperiment(experimentId) {
  selectedExperimentId = experimentId;
  await loadExperimentDetail(experimentId);
}

async function loadExperimentDetail(experimentId) {
  try {
    const [detail, results, health] = await Promise.all([
      apiCall(`/experiments/${experimentId}`),
      apiCall(`/experiments/${experimentId}/results`).catch(() => null),
      apiCall(`/experiments/${experimentId}/health`).catch(() => null),
    ]);

    renderDetail(detail);
    if (results) renderResults(results);
    if (health) renderHealth(health);

    document.getElementById("experiment-detail").classList.remove("hidden");
    document
      .getElementById("experiment-detail")
      .scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    console.error("Failed to load experiment detail:", err);
  }
}

function renderDetail(detail) {
  document.getElementById("detail-name").textContent = detail.name;

  // Meta info
  const metaHtml = `
        <div class="meta-item">
            <div class="meta-label">State</div>
            <div class="meta-value"><span class="state-badge ${detail.state}">${detail.state}</span></div>
        </div>
        <div class="meta-item">
            <div class="meta-label">Traffic Allocation</div>
            <div class="meta-value">${detail.traffic_allocation}</div>
        </div>
        <div class="meta-item">
            <div class="meta-label">Rollout</div>
            <div class="meta-value">${detail.current_rollout_percentage}%</div>
        </div>
        <div class="meta-item">
            <div class="meta-label">Created</div>
            <div class="meta-value">${formatDate(detail.created_at)}</div>
        </div>
        <div class="meta-item">
            <div class="meta-label">Events</div>
            <div class="meta-value">${(detail.event_count || 0).toLocaleString()}</div>
        </div>
        ${
          detail.hypothesis
            ? `
        <div class="meta-item" style="grid-column: 1 / -1">
            <div class="meta-label">Hypothesis</div>
            <div class="meta-value" style="font-family: inherit">${escapeHtml(detail.hypothesis)}</div>
        </div>`
            : ""
        }
    `;
  document.getElementById("detail-meta").innerHTML = metaHtml;

  // Action buttons
  const actions = [];
  if (detail.state === "draft") {
    actions.push(
      `<button class="btn btn-success btn-sm" onclick="doAction('${detail.id}', 'start')">Start</button>`,
    );
    actions.push(
      `<button class="btn btn-danger btn-sm" onclick="doAction('${detail.id}', 'cancel')">Cancel</button>`,
    );
  } else if (detail.state === "running") {
    actions.push(
      `<button class="btn btn-secondary btn-sm" onclick="doAction('${detail.id}', 'pause')">Pause</button>`,
    );
    actions.push(
      `<button class="btn btn-primary btn-sm" onclick="doAction('${detail.id}', 'stop')">Stop & Analyze</button>`,
    );
  } else if (detail.state === "paused") {
    actions.push(
      `<button class="btn btn-success btn-sm" onclick="doAction('${detail.id}', 'start')">Resume</button>`,
    );
  } else if (detail.state === "analyzing") {
    actions.push(
      `<button class="btn btn-primary btn-sm" onclick="doAction('${detail.id}', 'complete')">Complete</button>`,
    );
  }
  document.getElementById("detail-actions").innerHTML = actions.join("");

  // Traffic chart
  renderTrafficChart(detail);
}

async function doAction(experimentId, action) {
  try {
    let endpoint;
    if (action === "cancel") {
      // Cancel is not a direct endpoint in our API, so we use stop then complete
      endpoint = `/experiments/${experimentId}/stop`;
    } else {
      endpoint = `/experiments/${experimentId}/${action}`;
    }
    await apiCall(endpoint, { method: "POST" });
    await refreshAll();
  } catch (err) {
    alert(`Action failed: ${err.message}`);
  }
}

// ----------------------------------------------------------------
// Results Rendering
// ----------------------------------------------------------------

function renderResults(results) {
  const section = document.getElementById("results-section");

  // Frequentist
  const freqEl = document.querySelector("#frequentist-results .result-content");
  if (results.frequentist) {
    const f = results.frequentist;
    freqEl.innerHTML = `
            <div class="result-row">
                <span class="label">Test</span>
                <span class="value">${f.test_type}</span>
            </div>
            <div class="result-row">
                <span class="label">p-value</span>
                <span class="value ${f.is_significant ? "significant" : "not-significant"}">${f.p_value.toFixed(6)}</span>
            </div>
            <div class="result-row">
                <span class="label">Significant</span>
                <span class="value ${f.is_significant ? "significant" : "not-significant"}">${f.is_significant ? "Yes" : "No"}</span>
            </div>
            <div class="result-row">
                <span class="label">Effect Size</span>
                <span class="value">${(f.effect_size * 100).toFixed(3)}%</span>
            </div>
            <div class="result-row">
                <span class="label">Relative Effect</span>
                <span class="value">${(f.relative_effect * 100).toFixed(2)}%</span>
            </div>
            <div class="result-row">
                <span class="label">95% CI</span>
                <span class="value">[${(f.confidence_interval[0] * 100).toFixed(3)}%, ${(f.confidence_interval[1] * 100).toFixed(3)}%]</span>
            </div>
            <div class="result-row">
                <span class="label">Control Rate</span>
                <span class="value">${(f.control_rate * 100).toFixed(2)}%</span>
            </div>
            <div class="result-row">
                <span class="label">Treatment Rate</span>
                <span class="value">${(f.treatment_rate * 100).toFixed(2)}%</span>
            </div>
        `;

    renderCIChart(f);
  } else {
    freqEl.innerHTML = '<p class="empty-state">No frequentist results yet</p>';
  }

  // Bayesian
  const bayesEl = document.querySelector("#bayesian-results .result-content");
  if (results.bayesian) {
    const b = results.bayesian;
    bayesEl.innerHTML = `
            <div class="result-row">
                <span class="label">P(Treatment Better)</span>
                <span class="value ${b.probability_treatment_better > 0.95 ? "significant" : ""}">${(b.probability_treatment_better * 100).toFixed(2)}%</span>
            </div>
            <div class="result-row">
                <span class="label">Expected Loss (Treatment)</span>
                <span class="value">${(b.expected_loss_treatment * 100).toFixed(4)}%</span>
            </div>
            <div class="result-row">
                <span class="label">Expected Loss (Control)</span>
                <span class="value">${(b.expected_loss_control * 100).toFixed(4)}%</span>
            </div>
            <div class="result-row">
                <span class="label">95% Credible Interval</span>
                <span class="value">[${(b.credible_interval[0] * 100).toFixed(3)}%, ${(b.credible_interval[1] * 100).toFixed(3)}%]</span>
            </div>
            <div class="result-row">
                <span class="label">Risk Threshold Met</span>
                <span class="value ${b.risk_threshold_met ? "significant" : ""}">${b.risk_threshold_met ? "Yes" : "No"}</span>
            </div>
        `;

    renderPosteriorChart(b);
  } else {
    bayesEl.innerHTML = '<p class="empty-state">No Bayesian results yet</p>';
  }

  // Recommendation
  const recEl = document.getElementById("recommendation");
  recEl.textContent = results.recommendation || "No recommendation available.";
}

// ----------------------------------------------------------------
// Health Rendering
// ----------------------------------------------------------------

function renderHealth(health) {
  const container = document.getElementById("health-content");

  if (!health.alerts || health.alerts.length === 0) {
    container.innerHTML =
      '<div class="health-ok">All health checks passed</div>';
    return;
  }

  const severityIcons = {
    critical: "\u26D4",
    warning: "\u26A0",
    info: "\u2139",
  };

  container.innerHTML = health.alerts
    .map(
      (alert) => `
        <div class="alert-item ${alert.severity}">
            <span class="alert-icon">${severityIcons[alert.severity] || ""}</span>
            <div>
                <strong>${alert.type}</strong>: ${escapeHtml(alert.message)}
            </div>
        </div>
    `,
    )
    .join("");
}

// ----------------------------------------------------------------
// Charts
// ----------------------------------------------------------------

function renderTrafficChart(detail) {
  const ctx = document.getElementById("traffic-chart").getContext("2d");
  if (trafficChart) trafficChart.destroy();

  const variants = detail.variants || [];
  const labels = variants.map((v) => v.name);
  const data = variants.map((v) => v.traffic_percentage);
  const colors = ["#4f8ff7", "#34d399", "#fbbf24", "#a78bfa", "#ef4444"];

  trafficChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: labels,
      datasets: [
        {
          data: data,
          backgroundColor: colors.slice(0, labels.length),
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: "#9aa0a6", font: { size: 11 } },
        },
      },
    },
  });
}

function renderCIChart(freq) {
  const ctx = document.getElementById("ci-chart").getContext("2d");
  if (ciChart) ciChart.destroy();

  const controlRate = freq.control_rate * 100;
  const treatmentRate = freq.treatment_rate * 100;
  const ciLow = freq.confidence_interval[0] * 100;
  const ciHigh = freq.confidence_interval[1] * 100;
  const diff = freq.effect_size * 100;

  ciChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Control", "Treatment", "Difference"],
      datasets: [
        {
          label: "Rate / Difference (%)",
          data: [controlRate, treatmentRate, diff],
          backgroundColor: [
            "#4f8ff7",
            "#34d399",
            diff > 0 ? "#34d399" : "#ef4444",
          ],
          borderWidth: 0,
          barPercentage: 0.6,
        },
      ],
    },
    options: {
      responsive: true,
      indexAxis: "y",
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            afterLabel: function (context) {
              if (context.dataIndex === 2) {
                return `95% CI: [${ciLow.toFixed(3)}%, ${ciHigh.toFixed(3)}%]`;
              }
              return "";
            },
          },
        },
      },
      scales: {
        x: {
          grid: { color: "#2d3140" },
          ticks: { color: "#9aa0a6", callback: (v) => v.toFixed(1) + "%" },
        },
        y: {
          grid: { display: false },
          ticks: { color: "#9aa0a6" },
        },
      },
    },
  });
}

function renderPosteriorChart(bayes) {
  const ctx = document.getElementById("posterior-chart").getContext("2d");
  if (posteriorChart) posteriorChart.destroy();

  // Generate Beta PDF samples for visualization
  const postA = bayes.posterior_control;
  const postB = bayes.posterior_treatment;

  // Generate x values around the posterior means
  const xMin = Math.max(0, Math.min(postA.mean, postB.mean) - 0.05);
  const xMax = Math.min(1, Math.max(postA.mean, postB.mean) + 0.05);
  const nPoints = 100;
  const xValues = [];
  for (let i = 0; i <= nPoints; i++) {
    xValues.push(xMin + ((xMax - xMin) * i) / nPoints);
  }

  // Approximate Beta PDF using the normal approximation for display
  function betaPDF(x, alpha, beta) {
    if (x <= 0 || x >= 1) return 0;
    const mean = alpha / (alpha + beta);
    const variance =
      (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1));
    const std = Math.sqrt(variance);
    if (std === 0) return 0;
    const z = (x - mean) / std;
    return Math.exp(-0.5 * z * z) / (std * Math.sqrt(2 * Math.PI));
  }

  const controlPDF = xValues.map((x) =>
    betaPDF(x, postA.alpha || 1, postA.beta || 1),
  );
  const treatmentPDF = xValues.map((x) =>
    betaPDF(x, postB.alpha || 1, postB.beta || 1),
  );

  const labels = xValues.map((x) => (x * 100).toFixed(1) + "%");

  posteriorChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Control",
          data: controlPDF,
          borderColor: "#4f8ff7",
          backgroundColor: "rgba(79, 143, 247, 0.1)",
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          borderWidth: 2,
        },
        {
          label: "Treatment",
          data: treatmentPDF,
          borderColor: "#34d399",
          backgroundColor: "rgba(52, 211, 153, 0.1)",
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: "#9aa0a6", font: { size: 11 } },
        },
      },
      scales: {
        x: {
          display: true,
          grid: { color: "#2d3140" },
          ticks: {
            color: "#9aa0a6",
            maxTicksLimit: 6,
            font: { size: 10 },
          },
        },
        y: {
          display: false,
        },
      },
    },
  });
}

// ----------------------------------------------------------------
// Create Experiment
// ----------------------------------------------------------------

function showCreateModal() {
  document.getElementById("create-modal").classList.remove("hidden");
}

function hideCreateModal() {
  document.getElementById("create-modal").classList.add("hidden");
}

async function createExperiment(event) {
  event.preventDefault();

  const controlPct = parseFloat(document.getElementById("control-pct").value);
  const treatmentPct = parseFloat(
    document.getElementById("treatment-pct").value,
  );

  if (Math.abs(controlPct + treatmentPct - 100) > 0.01) {
    alert("Traffic percentages must sum to 100%");
    return;
  }

  const payload = {
    name: document.getElementById("exp-name").value,
    hypothesis: document.getElementById("exp-hypothesis").value,
    description: document.getElementById("exp-description").value,
    variants: [
      {
        name: document.getElementById("control-name").value,
        traffic_percentage: controlPct,
        is_control: true,
      },
      {
        name: document.getElementById("treatment-name").value,
        traffic_percentage: treatmentPct,
        is_control: false,
      },
    ],
    success_metrics: [
      {
        name: document.getElementById("metric-name").value,
        metric_type: document.getElementById("metric-type").value,
        minimum_detectable_effect: parseFloat(
          document.getElementById("metric-mde").value,
        ),
      },
    ],
    traffic_allocation: document.getElementById("traffic-alloc").value,
    owner: document.getElementById("exp-owner").value,
  };

  try {
    await apiCall("/experiments", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    hideCreateModal();
    document.getElementById("create-form").reset();
    await refreshAll();
  } catch (err) {
    alert(`Failed to create experiment: ${err.message}`);
  }
}

// ----------------------------------------------------------------
// Power Analysis
// ----------------------------------------------------------------

async function runPowerAnalysis() {
  const baseline = document.getElementById("pa-baseline").value;
  const mde = document.getElementById("pa-mde").value;
  const alpha = document.getElementById("pa-alpha").value;
  const power = document.getElementById("pa-power").value;

  try {
    const result = await apiCall(
      `/power-analysis?baseline_rate=${baseline}&mde=${mde}&alpha=${alpha}&power=${power}`,
    );

    const el = document.getElementById("power-result");
    el.classList.remove("hidden");
    el.innerHTML = `
            <div><strong>Required Sample Size Per Group:</strong> ${result.required_sample_size_per_group.toLocaleString()}</div>
            <div><strong>Total Required:</strong> ${result.total_required.toLocaleString()}</div>
            <div><strong>Power:</strong> ${(result.power * 100).toFixed(0)}%</div>
            <div><strong>Significance Level:</strong> ${(result.alpha * 100).toFixed(1)}%</div>
        `;
  } catch (err) {
    alert(`Power analysis failed: ${err.message}`);
  }
}

// ----------------------------------------------------------------
// Utilities
// ----------------------------------------------------------------

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function formatDate(isoString) {
  if (!isoString) return "N/A";
  const date = new Date(isoString);
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}
