/**
 * RxPredict Frontend Application
 *
 * Handles API integration, form validation, latency display,
 * prediction history, and response distribution visualization.
 */

const API_BASE = window.location.origin;
const PREDICT_URL = `${API_BASE}/predict`;
const BATCH_PREDICT_URL = `${API_BASE}/batch-predict`;
const HEALTH_URL = `${API_BASE}/health`;

// State
let predictionHistory = [];
let healthCheckInterval = null;

// --- Initialization ---

document.addEventListener("DOMContentLoaded", () => {
  initThemeToggle();
  initForm();
  startHealthCheck();
});

// --- Theme Toggle ---

function initThemeToggle() {
  const toggle = document.getElementById("theme-toggle");
  const html = document.documentElement;
  const icon = toggle.querySelector(".theme-icon");

  const savedTheme = localStorage.getItem("rxpredict-theme") || "light";
  html.setAttribute("data-theme", savedTheme);
  icon.textContent = savedTheme === "dark" ? "\u2600" : "\u263D";

  toggle.addEventListener("click", () => {
    const current = html.getAttribute("data-theme");
    const next = current === "dark" ? "light" : "dark";
    html.setAttribute("data-theme", next);
    localStorage.setItem("rxpredict-theme", next);
    icon.textContent = next === "dark" ? "\u2600" : "\u263D";
  });
}

// --- Health Check ---

function startHealthCheck() {
  checkHealth();
  healthCheckInterval = setInterval(checkHealth, 15000);
}

async function checkHealth() {
  const statusBadge = document.getElementById("status-badge");
  try {
    const start = performance.now();
    const response = await fetch(HEALTH_URL, {
      signal: AbortSignal.timeout(5000),
    });
    const elapsed = performance.now() - start;
    const data = await response.json();

    if (data.status === "healthy") {
      statusBadge.textContent = `Status: Healthy (${Math.round(elapsed)}ms)`;
      statusBadge.className = "badge badge-success";
    } else if (data.status === "degraded") {
      statusBadge.textContent = "Status: Degraded";
      statusBadge.className = "badge badge-warning";
    } else {
      statusBadge.textContent = "Status: Unhealthy";
      statusBadge.className = "badge badge-danger";
    }
  } catch {
    statusBadge.textContent = "Status: Offline";
    statusBadge.className = "badge badge-danger";
  }
}

// --- Form ---

function initForm() {
  const form = document.getElementById("prediction-form");
  form.addEventListener("submit", handleSubmit);
}

async function handleSubmit(event) {
  event.preventDefault();

  const btn = document.getElementById("predict-btn");
  const loading = document.getElementById("loading");
  const results = document.getElementById("results");
  const errorDiv = document.getElementById("error");

  // Validate
  if (!validateForm()) return;

  // UI state
  btn.disabled = true;
  btn.textContent = "Predicting...";
  loading.classList.remove("hidden");
  results.classList.add("hidden");
  errorDiv.classList.add("hidden");

  try {
    const payload = buildPayload();
    const start = performance.now();

    const response = await fetch(PREDICT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const totalLatency = performance.now() - start;

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(
        errData.detail || `HTTP ${response.status}: ${response.statusText}`,
      );
    }

    const data = await response.json();
    const serverLatency =
      parseFloat(response.headers.get("X-Response-Time")) || totalLatency;

    displayResults(data, totalLatency);
    addToHistory(data, payload, totalLatency);
    updateLatencyBadge(serverLatency);
  } catch (err) {
    showError(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict Response";
    loading.classList.add("hidden");
  }
}

function validateForm() {
  const age = parseInt(document.getElementById("age").value);
  const weight = parseFloat(document.getElementById("weight").value);
  const height = parseFloat(document.getElementById("height").value);
  const drugName = document.getElementById("drug-name").value.trim();

  if (isNaN(age) || age < 0 || age > 120) {
    showError("Age must be between 0 and 120");
    return false;
  }
  if (isNaN(weight) || weight < 1 || weight > 500) {
    showError("Weight must be between 1 and 500 kg");
    return false;
  }
  if (isNaN(height) || height < 30 || height > 300) {
    showError("Height must be between 30 and 300 cm");
    return false;
  }
  if (!drugName) {
    showError("Drug name is required");
    return false;
  }
  return true;
}

function buildPayload() {
  const form = document.getElementById("prediction-form");
  const age = parseInt(form.age.value);
  const weightKg = parseFloat(form.weight_kg.value);
  const heightCm = parseFloat(form.height_cm.value);
  const heightM = heightCm / 100;
  const bmi = Math.round((weightKg / (heightM * heightM)) * 10) / 10;

  // Collect selected genetic variants
  const cyp2d6 = getSelectedValues("cyp2d6");
  const cyp2c19 = getSelectedValues("cyp2c19");

  // Collect conditions
  const conditions = [];
  document
    .querySelectorAll('input[name="conditions"]:checked')
    .forEach((cb) => {
      conditions.push(cb.value);
    });

  return {
    genetic_profile: {
      CYP2D6: cyp2d6,
      CYP2C19: cyp2c19,
      CYP3A4: ["*1"],
      CYP2C9: ["*1"],
      VKORC1: [],
      DPYD: ["*1"],
      TPMT: ["*1"],
      UGT1A1: ["*1"],
      SLCO1B1: ["*1A"],
      "HLA-B": [],
    },
    metabolizer_phenotype: form.metabolizer_phenotype.value,
    demographics: {
      age: age,
      weight_kg: weightKg,
      height_cm: heightCm,
      bmi: bmi,
      sex: form.sex.value,
      ethnicity: form.ethnicity.value,
    },
    drug: {
      name: form.drug_name.value,
      drug_class: form.drug_class.value,
      dosage_mg: parseFloat(form.dosage_mg.value),
      max_dosage_mg: 1000,
    },
    medical_history: {
      num_current_medications:
        parseInt(form.num_current_medications.value) || 0,
      num_allergies: parseInt(form.num_allergies.value) || 0,
      num_adverse_reactions: 0,
      conditions: conditions,
      pregnant: false,
      age: age,
    },
  };
}

function getSelectedValues(selectId) {
  const select = document.getElementById(selectId);
  const values = [];
  for (const option of select.selectedOptions) {
    values.push(option.value);
  }
  return values;
}

// --- Results Display ---

function displayResults(data, totalLatency) {
  const results = document.getElementById("results");
  results.classList.remove("hidden");

  // Predicted class
  const classEl = document.getElementById("result-class");
  classEl.textContent = formatClassName(data.predicted_class);

  // Risk level
  const riskEl = document.getElementById("result-risk");
  riskEl.textContent = formatRiskLevel(data.risk_level);
  riskEl.className = `result-risk risk-${getRiskCategory(data.risk_level)}`;

  // Latency
  const latencyEl = document.getElementById("result-latency");
  const inferenceMs = data.inference_time_ms || 0;
  latencyEl.textContent = `${inferenceMs.toFixed(1)}ms`;
  latencyEl.className =
    inferenceMs > 100 ? "latency-value slow" : "latency-value";

  // Confidence bar
  const probability = data.response_probability * 100;
  const ciLower = data.confidence_lower * 100;
  const ciUpper = data.confidence_upper * 100;

  document.getElementById("confidence-fill").style.width = `${probability}%`;
  document.getElementById("confidence-label").textContent =
    `${probability.toFixed(1)}%`;

  const rangeEl = document.getElementById("confidence-range");
  rangeEl.style.left = `${ciLower}%`;
  rangeEl.style.width = `${ciUpper - ciLower}%`;

  document.getElementById("ci-lower").textContent = `${ciLower.toFixed(1)}%`;
  document.getElementById("ci-estimate").textContent =
    `${probability.toFixed(1)}%`;
  document.getElementById("ci-upper").textContent = `${ciUpper.toFixed(1)}%`;

  // Draw distribution chart
  drawDistributionChart(data);

  // Feature importance
  if (
    data.feature_importance &&
    Object.keys(data.feature_importance).length > 0
  ) {
    const section = document.getElementById("feature-importance-section");
    section.classList.remove("hidden");
    renderFeatureImportance(data.feature_importance);
  }

  // Metadata
  document.getElementById("model-version").textContent =
    data.model_version || "--";
  document.getElementById("request-id").textContent = data.request_id || "--";
  const cacheStatus = document.getElementById("cache-status");
  if (data.cache_hit) {
    cacheStatus.textContent = "Served from cache";
    cacheStatus.style.color = "var(--accent-info)";
  } else {
    cacheStatus.textContent = "Fresh prediction";
    cacheStatus.style.color = "var(--text-muted)";
  }
}

function drawDistributionChart(data) {
  const canvas = document.getElementById("distribution-chart");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;

  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
  ctx.scale(dpr, dpr);

  const w = canvas.clientWidth;
  const h = canvas.clientHeight;

  ctx.clearRect(0, 0, w, h);

  const classes = [
    "Poor Response",
    "Partial Response",
    "Good Response",
    "Excellent Response",
  ];
  const classKeys = [
    "poor_response",
    "partial_response",
    "good_response",
    "excellent_response",
  ];
  const colors = ["#dc2626", "#d97706", "#059669", "#0891b2"];

  // Simulate distribution from the single prediction probability
  const prob = data.response_probability;
  const predictedIdx = classKeys.indexOf(data.predicted_class);
  const probs = [0.15, 0.25, 0.35, 0.25]; // defaults
  if (predictedIdx >= 0) {
    probs[predictedIdx] = prob;
    const remaining = 1 - prob;
    const otherCount = probs.length - 1;
    for (let i = 0; i < probs.length; i++) {
      if (i !== predictedIdx) {
        probs[i] = remaining / otherCount;
      }
    }
  }

  const barWidth = (w - 80) / classes.length;
  const maxBarHeight = h - 60;
  const startX = 50;
  const startY = h - 30;

  // Draw bars
  for (let i = 0; i < classes.length; i++) {
    const x = startX + i * barWidth + 8;
    const barH = probs[i] * maxBarHeight;
    const y = startY - barH;

    ctx.fillStyle = colors[i];
    ctx.beginPath();
    const r = 4;
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + barWidth - 16 - r, y);
    ctx.quadraticCurveTo(x + barWidth - 16, y, x + barWidth - 16, y + r);
    ctx.lineTo(x + barWidth - 16, startY);
    ctx.lineTo(x, startY);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.fill();

    // Highlight predicted class
    if (i === predictedIdx) {
      ctx.strokeStyle = "var(--text-primary)";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Percentage label
    ctx.fillStyle =
      getComputedStyle(document.documentElement)
        .getPropertyValue("--text-primary")
        .trim() || "#1a2332";
    ctx.font = "bold 12px system-ui";
    ctx.textAlign = "center";
    ctx.fillText(
      `${(probs[i] * 100).toFixed(0)}%`,
      x + (barWidth - 16) / 2,
      y - 8,
    );

    // Class label
    ctx.fillStyle =
      getComputedStyle(document.documentElement)
        .getPropertyValue("--text-muted")
        .trim() || "#718096";
    ctx.font = "11px system-ui";
    const label = classes[i].split(" ")[0];
    ctx.fillText(label, x + (barWidth - 16) / 2, startY + 16);
  }
}

function renderFeatureImportance(features) {
  const container = document.getElementById("feature-bars");
  container.innerHTML = "";

  const entries = Object.entries(features).sort((a, b) => b[1] - a[1]);
  const maxVal = entries.length > 0 ? entries[0][1] : 1;

  for (const [name, value] of entries) {
    const pct = (value / maxVal) * 100;
    const item = document.createElement("div");
    item.className = "feature-bar-item";
    item.innerHTML = `
            <span class="feature-bar-name">${formatFeatureName(name)}</span>
            <div class="feature-bar-track">
                <div class="feature-bar-fill" style="width: ${pct}%"></div>
            </div>
            <span class="feature-bar-value">${value.toFixed(4)}</span>
        `;
    container.appendChild(item);
  }
}

// --- History ---

function addToHistory(data, payload, totalLatency) {
  const entry = {
    time: new Date().toLocaleTimeString(),
    drug: payload.drug.name,
    predictedClass: formatClassName(data.predicted_class),
    probability: (data.response_probability * 100).toFixed(1) + "%",
    riskLevel: formatRiskLevel(data.risk_level),
    latency: `${data.inference_time_ms.toFixed(1)}ms`,
    cacheHit: data.cache_hit ? "Yes" : "No",
  };

  predictionHistory.unshift(entry);
  if (predictionHistory.length > 50) predictionHistory.pop();

  renderHistory();
}

function renderHistory() {
  const tbody = document.getElementById("history-body");
  if (predictionHistory.length === 0) {
    tbody.innerHTML = `<tr class="empty-row"><td colspan="7">No predictions yet.</td></tr>`;
    return;
  }

  tbody.innerHTML = predictionHistory
    .map(
      (entry) => `
        <tr>
            <td>${entry.time}</td>
            <td>${entry.drug}</td>
            <td>${entry.predictedClass}</td>
            <td><strong>${entry.probability}</strong></td>
            <td>${entry.riskLevel}</td>
            <td><code>${entry.latency}</code></td>
            <td>${entry.cacheHit}</td>
        </tr>
    `,
    )
    .join("");
}

// --- Latency Badge ---

function updateLatencyBadge(ms) {
  const badge = document.getElementById("latency-badge");
  const rounded = Math.round(ms * 10) / 10;
  badge.textContent = `Latency: ${rounded}ms`;

  if (rounded < 50) {
    badge.className = "badge badge-success";
  } else if (rounded < 100) {
    badge.className = "badge badge-warning";
  } else {
    badge.className = "badge badge-danger";
  }
}

// --- Error Handling ---

function showError(message) {
  const errorDiv = document.getElementById("error");
  const errorMsg = document.getElementById("error-message");
  const results = document.getElementById("results");

  results.classList.add("hidden");
  errorDiv.classList.remove("hidden");
  errorMsg.textContent = message;
}

// --- Formatters ---

function formatClassName(cls) {
  if (!cls) return "--";
  return cls.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatRiskLevel(risk) {
  if (!risk) return "--";
  return risk.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function getRiskCategory(risk) {
  if (!risk) return "moderate";
  if (risk.includes("high")) return "high";
  if (risk.includes("moderate")) return "moderate";
  if (risk.includes("low")) return "low";
  if (risk.includes("minimal")) return "minimal";
  return "moderate";
}

function formatFeatureName(name) {
  return name
    .replace(/^(gene_|demo_|drug_|hist_)/, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}
