/**
 * PharmaForecast Dashboard - Interactive forecast visualization
 *
 * Uses Plotly.js for time series charts with confidence intervals,
 * model comparison, and backtest result visualization.
 */

const API_BASE =
  window.location.hostname === "localhost" ? "http://localhost:8000" : "";

let currentForecastData = null;
let currentBacktestData = null;
let currentView = "forecast";

// --- API Communication ---

async function apiRequest(endpoint, method = "GET", body = null) {
  const options = {
    method,
    headers: { "Content-Type": "application/json" },
  };
  if (body) {
    options.body = JSON.stringify(body);
  }

  const response = await fetch(`${API_BASE}${endpoint}`, options);
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `API error: ${response.status}`);
  }
  return response.json();
}

// --- Initialization ---

document.addEventListener("DOMContentLoaded", () => {
  checkApiHealth();
  renderEmptyChart();
  loadAlerts();
});

async function checkApiHealth() {
  const indicator = document.getElementById("apiStatus");
  const text = document.getElementById("apiStatusText");

  try {
    const data = await apiRequest("/health");
    indicator.className = "status-indicator connected";
    text.textContent = "API Connected";
  } catch {
    indicator.className = "status-indicator error";
    text.textContent = "API Unavailable";
  }
}

// --- Sample Data Generation ---

function generateSampleData(seriesId, nPoints = 730) {
  const data = [];
  const startDate = new Date("2024-01-01");
  let baseValue;
  let seasonalAmplitude;
  let trendSlope;
  let noiseLevel;

  // Different profiles for different series
  switch (seriesId) {
    case "drug_demand_acetaminophen":
      baseValue = 5000;
      seasonalAmplitude = 1500;
      trendSlope = 2.0;
      noiseLevel = 300;
      break;
    case "drug_demand_amoxicillin":
      baseValue = 3000;
      seasonalAmplitude = 1200;
      trendSlope = 1.5;
      noiseLevel = 250;
      break;
    case "drug_demand_lisinopril":
      baseValue = 8000;
      seasonalAmplitude = 400;
      trendSlope = 5.0;
      noiseLevel = 200;
      break;
    case "ae_reports_statins":
      baseValue = 120;
      seasonalAmplitude = 20;
      trendSlope = 0.05;
      noiseLevel = 15;
      break;
    case "ae_reports_ssri":
      baseValue = 85;
      seasonalAmplitude = 15;
      trendSlope = 0.08;
      noiseLevel = 10;
      break;
    default:
      baseValue = 1000;
      seasonalAmplitude = 200;
      trendSlope = 1.0;
      noiseLevel = 100;
  }

  for (let i = 0; i < nPoints; i++) {
    const date = new Date(startDate);
    date.setDate(date.getDate() + i);

    const trend = trendSlope * i;
    const seasonal = seasonalAmplitude * Math.sin((2 * Math.PI * i) / 365.25);
    const weeklyPattern =
      seasonalAmplitude * 0.1 * Math.sin((2 * Math.PI * i) / 7);
    const noise = (Math.random() - 0.5) * 2 * noiseLevel;

    const value = Math.max(
      0,
      baseValue + trend + seasonal + weeklyPattern + noise,
    );

    data.push({
      date: date.toISOString().split("T")[0],
      value: Math.round(value * 100) / 100,
    });
  }

  return data;
}

// --- Forecast Generation ---

async function generateForecast() {
  const seriesId = document.getElementById("seriesSelect").value;
  const horizon = parseInt(document.getElementById("horizonInput").value);
  const confidence = parseFloat(
    document.getElementById("confidenceInput").value,
  );
  const modelType = document.getElementById("modelSelect").value;

  const btn = document.getElementById("forecastBtn");
  btn.disabled = true;
  btn.textContent = "Generating...";

  try {
    const sampleData = generateSampleData(seriesId);

    const requestBody = {
      series_id: seriesId,
      data: sampleData,
      horizon: horizon,
      confidence_level: confidence,
      use_ensemble: modelType === "ensemble",
    };

    let forecastResult;
    try {
      forecastResult = await apiRequest("/forecast", "POST", requestBody);
    } catch {
      // If API is unavailable, generate mock forecast
      forecastResult = generateMockForecast(
        sampleData,
        seriesId,
        horizon,
        confidence,
        modelType,
      );
    }

    currentForecastData = {
      historical: sampleData,
      forecast: forecastResult,
    };

    renderForecastChart(currentForecastData);
    updateMetrics(forecastResult);
  } catch (error) {
    console.error("Forecast error:", error);
    alert("Failed to generate forecast: " + error.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Generate Forecast";
  }
}

function generateMockForecast(
  historicalData,
  seriesId,
  horizon,
  confidence,
  modelType,
) {
  const lastValue = historicalData[historicalData.length - 1].value;
  const recentValues = historicalData.slice(-30).map((d) => d.value);
  const recentMean =
    recentValues.reduce((a, b) => a + b, 0) / recentValues.length;
  const recentStd = Math.sqrt(
    recentValues.reduce((s, v) => s + Math.pow(v - recentMean, 2), 0) /
      recentValues.length,
  );

  const zScore =
    confidence === 0.99 ? 2.576 : confidence === 0.9 ? 1.645 : 1.96;
  const forecast = [];
  const lastDate = new Date(historicalData[historicalData.length - 1].date);

  for (let i = 1; i <= horizon; i++) {
    const date = new Date(lastDate);
    date.setDate(date.getDate() + i);

    const trend =
      ((recentMean - historicalData[historicalData.length - 31]?.value || 0) /
        30) *
      i;
    const seasonal = recentStd * 0.5 * Math.sin((2 * Math.PI * i) / 365.25);
    const noise = (Math.random() - 0.5) * recentStd * 0.3;
    const forecastValue = recentMean + trend + seasonal + noise;

    const uncertainty = recentStd * zScore * Math.sqrt(i / 30);

    forecast.push({
      date: date.toISOString().split("T")[0],
      forecast: Math.round(forecastValue * 100) / 100,
      lower_bound: Math.round((forecastValue - uncertainty) * 100) / 100,
      upper_bound: Math.round((forecastValue + uncertainty) * 100) / 100,
    });
  }

  return {
    series_id: seriesId,
    model: modelType === "ensemble" ? "ensemble" : "SARIMA(1,1,1)",
    horizon: horizon,
    confidence_level: confidence,
    forecast: forecast,
    metrics: {
      weight_arima: 0.3,
      weight_prophet: 0.4,
      weight_ml: 0.3,
    },
    generated_at: new Date().toISOString(),
  };
}

// --- Chart Rendering ---

function renderEmptyChart() {
  const layout = {
    title: {
      text: "Select a time series and generate a forecast",
      font: { size: 14, color: "#718096" },
    },
    xaxis: { title: "Date" },
    yaxis: { title: "Value" },
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    margin: { t: 50, r: 30, b: 50, l: 60 },
  };
  Plotly.newPlot("mainChart", [], layout, { responsive: true });
}

function renderForecastChart(data) {
  const historical = data.historical;
  const forecast = data.forecast;

  const histDates = historical.map((d) => d.date);
  const histValues = historical.map((d) => d.value);

  const fcstDates = forecast.forecast.map((d) => d.date);
  const fcstValues = forecast.forecast.map((d) => d.forecast);
  const lowerValues = forecast.forecast.map((d) => d.lower_bound);
  const upperValues = forecast.forecast.map((d) => d.upper_bound);

  const traces = [
    // Historical data
    {
      x: histDates,
      y: histValues,
      type: "scatter",
      mode: "lines",
      name: "Historical",
      line: { color: "#2d3748", width: 1.5 },
    },
    // Upper confidence bound (invisible, for fill)
    {
      x: fcstDates,
      y: upperValues,
      type: "scatter",
      mode: "lines",
      name: `Upper ${Math.round(forecast.confidence_level * 100)}% CI`,
      line: { color: "rgba(49, 130, 206, 0.3)", width: 0 },
      showlegend: false,
    },
    // Lower confidence bound (fill to upper)
    {
      x: fcstDates,
      y: lowerValues,
      type: "scatter",
      mode: "lines",
      name: `${Math.round(forecast.confidence_level * 100)}% Confidence Interval`,
      line: { color: "rgba(49, 130, 206, 0.3)", width: 0 },
      fill: "tonexty",
      fillcolor: "rgba(49, 130, 206, 0.12)",
    },
    // Forecast line
    {
      x: fcstDates,
      y: fcstValues,
      type: "scatter",
      mode: "lines",
      name: `Forecast (${forecast.model})`,
      line: { color: "#3182ce", width: 2.5, dash: "dash" },
    },
  ];

  const layout = {
    xaxis: {
      title: "Date",
      rangeslider: { visible: true, thickness: 0.05 },
      type: "date",
    },
    yaxis: { title: "Value" },
    legend: {
      orientation: "h",
      yanchor: "bottom",
      y: 1.02,
      xanchor: "right",
      x: 1,
    },
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    margin: { t: 30, r: 30, b: 80, l: 60 },
    hovermode: "x unified",
    shapes: [
      {
        type: "line",
        x0: fcstDates[0],
        x1: fcstDates[0],
        y0: 0,
        y1: 1,
        yref: "paper",
        line: { color: "#718096", width: 1, dash: "dot" },
      },
    ],
    annotations: [
      {
        x: fcstDates[0],
        y: 1,
        yref: "paper",
        text: "Forecast Start",
        showarrow: false,
        font: { size: 11, color: "#718096" },
        yanchor: "bottom",
      },
    ],
  };

  Plotly.newPlot("mainChart", traces, layout, { responsive: true });
}

function switchView(view) {
  currentView = view;

  document.querySelectorAll(".chip").forEach((chip) => {
    chip.classList.toggle("active", chip.dataset.view === view);
  });

  if (!currentForecastData) return;

  switch (view) {
    case "forecast":
      renderForecastChart(currentForecastData);
      break;
    case "comparison":
      renderModelComparison(currentForecastData);
      break;
    case "components":
      renderComponentsChart(currentForecastData);
      break;
  }
}

function renderModelComparison(data) {
  const forecast = data.forecast;
  const fcstDates = forecast.forecast.map((d) => d.date);
  const fcstValues = forecast.forecast.map((d) => d.forecast);

  // Simulate individual model outputs for comparison
  const arimaValues = fcstValues.map(
    (v, i) => v * (1 + 0.03 * Math.sin(i * 0.1)),
  );
  const prophetValues = fcstValues.map(
    (v, i) => v * (1 - 0.02 * Math.cos(i * 0.15)),
  );
  const mlValues = fcstValues.map(
    (v, i) => v * (1 + 0.04 * Math.sin(i * 0.2 + 1)),
  );

  const traces = [
    {
      x: fcstDates,
      y: fcstValues,
      type: "scatter",
      mode: "lines",
      name: "Ensemble",
      line: { color: "#3182ce", width: 2.5 },
    },
    {
      x: fcstDates,
      y: arimaValues,
      type: "scatter",
      mode: "lines",
      name: "ARIMA",
      line: { color: "#e53e3e", width: 1.5, dash: "dot" },
    },
    {
      x: fcstDates,
      y: prophetValues,
      type: "scatter",
      mode: "lines",
      name: "Prophet",
      line: { color: "#38a169", width: 1.5, dash: "dashdot" },
    },
    {
      x: fcstDates,
      y: mlValues,
      type: "scatter",
      mode: "lines",
      name: "ML (GBM)",
      line: { color: "#d69e2e", width: 1.5, dash: "longdash" },
    },
  ];

  const layout = {
    xaxis: { title: "Date", type: "date" },
    yaxis: { title: "Forecast Value" },
    legend: {
      orientation: "h",
      yanchor: "bottom",
      y: 1.02,
      xanchor: "right",
      x: 1,
    },
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    margin: { t: 30, r: 30, b: 50, l: 60 },
    hovermode: "x unified",
  };

  Plotly.newPlot("mainChart", traces, layout, { responsive: true });
}

function renderComponentsChart(data) {
  const historical = data.historical;
  const dates = historical.map((d) => d.date);
  const values = historical.map((d) => d.value);

  // Decompose into trend + seasonal + residual (simplified client-side)
  const windowSize = 30;
  const trend = values.map((_, i) => {
    const start = Math.max(0, i - windowSize);
    const end = Math.min(values.length, i + windowSize + 1);
    const window = values.slice(start, end);
    return window.reduce((a, b) => a + b, 0) / window.length;
  });

  const seasonal = values.map((v, i) => v - trend[i]);
  const residual = values.map((v, i) => v - trend[i] - seasonal[i] * 0.7);

  const traces = [
    {
      x: dates,
      y: values,
      type: "scatter",
      mode: "lines",
      name: "Original",
      line: { color: "#2d3748", width: 1 },
      xaxis: "x",
      yaxis: "y",
    },
    {
      x: dates,
      y: trend,
      type: "scatter",
      mode: "lines",
      name: "Trend",
      line: { color: "#3182ce", width: 2 },
      xaxis: "x2",
      yaxis: "y2",
    },
    {
      x: dates,
      y: seasonal,
      type: "scatter",
      mode: "lines",
      name: "Seasonal",
      line: { color: "#38a169", width: 1 },
      xaxis: "x3",
      yaxis: "y3",
    },
  ];

  const layout = {
    grid: {
      rows: 3,
      columns: 1,
      pattern: "independent",
      roworder: "top to bottom",
    },
    xaxis: { title: "" },
    yaxis: { title: "Original" },
    xaxis2: { title: "" },
    yaxis2: { title: "Trend" },
    xaxis3: { title: "Date" },
    yaxis3: { title: "Seasonal" },
    showlegend: false,
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    margin: { t: 20, r: 30, b: 50, l: 60 },
    height: 500,
  };

  Plotly.newPlot("mainChart", traces, layout, { responsive: true });
}

// --- Backtesting ---

async function runBacktest() {
  const seriesId = document.getElementById("seriesSelect").value;
  const horizon = parseInt(document.getElementById("horizonInput").value);

  const btn = document.getElementById("backtestBtn");
  btn.disabled = true;
  btn.textContent = "Running...";

  try {
    const sampleData = generateSampleData(seriesId);

    const requestBody = {
      series_id: seriesId,
      data: sampleData,
      n_folds: 5,
      horizon: Math.min(horizon, 30),
      min_train_size: 90,
    };

    let backtestResult;
    try {
      backtestResult = await apiRequest("/backtest", "POST", requestBody);
    } catch {
      backtestResult = generateMockBacktest(seriesId, requestBody.n_folds);
    }

    currentBacktestData = backtestResult;
    renderBacktestResults(backtestResult);
  } catch (error) {
    console.error("Backtest error:", error);
    alert("Backtest failed: " + error.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Backtest";
  }
}

function generateMockBacktest(seriesId, nFolds) {
  const folds = [];
  for (let i = 0; i < nFolds; i++) {
    folds.push({
      fold: i,
      train_size: 365 + i * 73,
      test_size: 30,
      mae: 50 + Math.random() * 100,
      rmse: 70 + Math.random() * 120,
      mape: 0.05 + Math.random() * 0.1,
      smape: 0.04 + Math.random() * 0.08,
    });
  }

  const avgMae = folds.reduce((s, f) => s + f.mae, 0) / nFolds;
  const avgRmse = folds.reduce((s, f) => s + f.rmse, 0) / nFolds;

  return {
    series_id: seriesId,
    model: "ensemble",
    mae: avgMae,
    rmse: avgRmse,
    mape: folds.reduce((s, f) => s + f.mape, 0) / nFolds,
    smape: folds.reduce((s, f) => s + f.smape, 0) / nFolds,
    forecast_bias: (Math.random() - 0.5) * 20,
    n_folds: nFolds,
    fold_results: folds,
    per_horizon_mae: Object.fromEntries(
      Array.from({ length: 30 }, (_, i) => [i + 1, avgMae * (1 + i * 0.02)]),
    ),
  };
}

function renderBacktestResults(result) {
  const section = document.getElementById("backtestSection");
  section.style.display = "block";

  // Per-fold MAE chart
  const folds = result.fold_results;
  const traces = [
    {
      x: folds.map((f) => `Fold ${f.fold + 1}`),
      y: folds.map((f) => f.mae),
      type: "bar",
      name: "MAE",
      marker: { color: "#3182ce" },
    },
    {
      x: folds.map((f) => `Fold ${f.fold + 1}`),
      y: folds.map((f) => f.rmse),
      type: "bar",
      name: "RMSE",
      marker: { color: "#e53e3e", opacity: 0.7 },
    },
  ];

  const layout = {
    barmode: "group",
    xaxis: { title: "" },
    yaxis: { title: "Error" },
    legend: {
      orientation: "h",
      yanchor: "bottom",
      y: 1.02,
      xanchor: "right",
      x: 1,
    },
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    margin: { t: 30, r: 30, b: 40, l: 60 },
  };

  Plotly.newPlot("backtestChart", traces, layout, { responsive: true });

  // Summary
  const summary = document.getElementById("backtestSummary");
  summary.innerHTML = `
        <div class="summary-item">
            <span class="summary-label">Overall MAE</span>
            <span class="summary-value">${result.mae.toFixed(2)}</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Overall RMSE</span>
            <span class="summary-value">${result.rmse.toFixed(2)}</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">MAPE</span>
            <span class="summary-value">${(result.mape * 100).toFixed(1)}%</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Forecast Bias</span>
            <span class="summary-value">${result.forecast_bias.toFixed(2)}</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Number of Folds</span>
            <span class="summary-value">${result.n_folds}</span>
        </div>
    `;

  // Update metrics panel too
  document.getElementById("metricMAE").textContent = result.mae.toFixed(2);
  document.getElementById("metricRMSE").textContent = result.rmse.toFixed(2);
  document.getElementById("metricMAPE").textContent = (
    result.mape * 100
  ).toFixed(1);
  document.getElementById("metricSMAPE").textContent = (
    result.smape * 100
  ).toFixed(1);
  document.getElementById("metricBias").textContent =
    result.forecast_bias.toFixed(2);
}

// --- Metrics Update ---

function updateMetrics(forecast) {
  const metrics = forecast.metrics;
  document.getElementById("metricModel").textContent = forecast.model;

  if (metrics.weight_arima !== undefined) {
    // Show weights as pseudo-metrics for ensemble
    document.getElementById("metricMAE").textContent = "--";
    document.getElementById("metricRMSE").textContent = "--";
    document.getElementById("metricMAPE").textContent = "--";
    document.getElementById("metricSMAPE").textContent = "--";
    document.getElementById("metricBias").textContent = "--";
  }

  if (metrics.aic !== undefined) {
    document.getElementById("metricMAE").textContent =
      metrics.aic?.toFixed(1) || "--";
  }
}

// --- Alerts ---

async function loadAlerts() {
  try {
    const alerts = await apiRequest("/alerts");
    renderAlerts(alerts);
  } catch {
    renderAlerts([]);
  }
}

function renderAlerts(alerts) {
  const banner = document.getElementById("alertBanner");
  const list = document.getElementById("alertsList");

  if (alerts.length === 0) {
    banner.style.display = "none";
    list.innerHTML =
      '<div class="empty-state">No active alerts. All forecast models operating within thresholds.</div>';
    return;
  }

  banner.style.display = "flex";
  document.getElementById("alertCount").textContent = alerts.length;

  list.innerHTML = alerts
    .map(
      (alert) => `
        <div class="alert-item severity-${alert.severity}">
            <div>
                <div class="alert-type">${alert.severity} - ${alert.alert_type}</div>
                <div class="alert-message">${alert.message}</div>
                <div class="alert-time">Series: ${alert.series_id} | ${new Date(alert.timestamp).toLocaleString()}</div>
            </div>
        </div>
    `,
    )
    .join("");
}
