/**
 * BiomarkerDash - Clinical Dashboard JavaScript
 *
 * Manages WebSocket connections, real-time chart rendering via Canvas API,
 * alert notifications, and patient data management.
 */

(function () {
  "use strict";

  // ----------------------------------------------------------------
  // Configuration
  // ----------------------------------------------------------------

  const WS_RECONNECT_BASE_MS = 1000;
  const WS_RECONNECT_MAX_MS = 30000;
  const METRICS_POLL_INTERVAL_MS = 5000;
  const CHART_MAX_POINTS = 200;
  const CHART_ANIMATION_MS = 150;

  // Normal ranges for biomarkers
  const NORMAL_RANGES = {
    glucose: [70, 100],
    hemoglobin: [12, 17.5],
    wbc: [4.5, 11],
    platelet: [150, 400],
    creatinine: [0.6, 1.2],
    alt: [7, 56],
    ast: [10, 40],
    heart_rate: [60, 100],
    blood_pressure_sys: [90, 120],
    blood_pressure_dia: [60, 80],
    temperature: [97, 99.5],
    oxygen_sat: [95, 100],
    potassium: [3.5, 5.0],
    sodium: [136, 145],
    tsh: [0.4, 4.0],
    cholesterol_total: [0, 200],
    troponin: [0, 0.04],
    crp: [0, 3.0],
  };

  const UNITS = {
    glucose: "mg/dL",
    hemoglobin: "g/dL",
    wbc: "K/uL",
    platelet: "K/uL",
    creatinine: "mg/dL",
    alt: "U/L",
    ast: "U/L",
    heart_rate: "bpm",
    blood_pressure_sys: "mmHg",
    blood_pressure_dia: "mmHg",
    temperature: "\u00b0F",
    oxygen_sat: "%",
    potassium: "mEq/L",
    sodium: "mEq/L",
    tsh: "mIU/L",
    cholesterol_total: "mg/dL",
    troponin: "ng/mL",
    crp: "mg/L",
  };

  // ----------------------------------------------------------------
  // State
  // ----------------------------------------------------------------

  let ws = null;
  let wsReconnectAttempts = 0;
  let selectedPatientId = null;
  let metricsInterval = null;

  // biomarkerType -> [{value, timestamp, anomaly}]
  const biomarkerData = {};
  // alert_id -> alert object
  const alerts = {};
  let currentAlertFilter = "all";
  let dataPointCount = 0;

  // Trend data for current chart selection
  let currentTrend = null;

  // ----------------------------------------------------------------
  // DOM References
  // ----------------------------------------------------------------

  const dom = {
    patientSelect: document.getElementById("patientSelect"),
    connectBtn: document.getElementById("connectBtn"),
    loadDemoBtn: document.getElementById("loadDemoBtn"),
    connectionStatus: document.getElementById("connectionStatus"),
    patientInfo: document.getElementById("patientInfo"),
    infoPatientId: document.getElementById("infoPatientId"),
    infoStatus: document.getElementById("infoStatus"),
    infoDataPoints: document.getElementById("infoDataPoints"),
    infoLastUpdate: document.getElementById("infoLastUpdate"),
    vitalsGrid: document.getElementById("vitalsGrid"),
    mainChart: document.getElementById("mainChart"),
    chartBiomarker: document.getElementById("chartBiomarker"),
    chartTimeRange: document.getElementById("chartTimeRange"),
    trendDirection: document.getElementById("trendDirection"),
    trendRate: document.getElementById("trendRate"),
    trendPredicted: document.getElementById("trendPredicted"),
    alertCount: document.getElementById("alertCount"),
    alertList: document.getElementById("alertList"),
    darkModeToggle: document.getElementById("darkModeToggle"),
    clock: document.getElementById("clock"),
    metricProcessed: document.getElementById("metricProcessed"),
    metricQueue: document.getElementById("metricQueue"),
    metricWS: document.getElementById("metricWS"),
    metricErrors: document.getElementById("metricErrors"),
  };

  // ----------------------------------------------------------------
  // Initialization
  // ----------------------------------------------------------------

  function init() {
    dom.connectBtn.addEventListener("click", onConnect);
    dom.loadDemoBtn.addEventListener("click", loadDemoData);
    dom.darkModeToggle.addEventListener("click", toggleDarkMode);
    dom.chartBiomarker.addEventListener("change", renderChart);
    dom.chartTimeRange.addEventListener("change", renderChart);

    // Alert filters
    document.querySelectorAll(".btn-filter").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelector(".btn-filter.active").classList.remove("active");
        btn.classList.add("active");
        currentAlertFilter = btn.dataset.severity;
        renderAlerts();
      });
    });

    // Clock update
    updateClock();
    setInterval(updateClock, 1000);

    // Metrics polling
    metricsInterval = setInterval(pollMetrics, METRICS_POLL_INTERVAL_MS);

    // Check stored dark mode preference
    if (localStorage.getItem("darkMode") === "true") {
      document.body.classList.add("dark-mode");
    }

    // Initial chart render
    renderChart();
  }

  // ----------------------------------------------------------------
  // WebSocket Management
  // ----------------------------------------------------------------

  function onConnect() {
    const patientId = dom.patientSelect.value;
    if (!patientId) {
      showNotification("Please select a patient first.", "warning");
      return;
    }

    if (ws) {
      ws.close();
    }

    selectedPatientId = patientId;
    clearData();
    connectWebSocket(patientId);
  }

  function connectWebSocket(patientId) {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/biomarkers/${patientId}`;

    updateConnectionStatus("connecting");

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      wsReconnectAttempts = 0;
      updateConnectionStatus("connected");
      dom.patientInfo.style.display = "block";
      dom.infoPatientId.textContent = patientId;
      dom.infoStatus.textContent = "Monitoring";
      showNotification(`Connected to patient ${patientId}`, "success");
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        handleMessage(msg);
      } catch (e) {
        console.error("Failed to parse WebSocket message:", e);
      }
    };

    ws.onclose = (event) => {
      updateConnectionStatus("disconnected");
      if (selectedPatientId && !event.wasClean) {
        scheduleReconnect(patientId);
      }
    };

    ws.onerror = () => {
      updateConnectionStatus("error");
    };

    // Ping interval to keep connection alive
    const pingInterval = setInterval(() => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "ping" }));
      } else {
        clearInterval(pingInterval);
      }
    }, 30000);
  }

  function scheduleReconnect(patientId) {
    const delay = Math.min(
      WS_RECONNECT_BASE_MS * Math.pow(2, wsReconnectAttempts),
      WS_RECONNECT_MAX_MS,
    );
    wsReconnectAttempts++;
    console.log(`Reconnecting in ${delay}ms (attempt ${wsReconnectAttempts})`);
    setTimeout(() => connectWebSocket(patientId), delay);
  }

  function updateConnectionStatus(status) {
    const el = dom.connectionStatus;
    el.className = "connection-status";
    const textEl = el.querySelector(".status-text");

    switch (status) {
      case "connected":
        el.classList.add("connected");
        textEl.textContent = "Connected";
        break;
      case "connecting":
        textEl.textContent = "Connecting...";
        break;
      case "disconnected":
        textEl.textContent = "Disconnected";
        break;
      case "error":
        textEl.textContent = "Error";
        break;
    }
  }

  // ----------------------------------------------------------------
  // Message Handling
  // ----------------------------------------------------------------

  function handleMessage(msg) {
    switch (msg.type) {
      case "reading":
        handleReading(msg.data);
        break;
      case "alert":
        handleAlert(msg.data);
        break;
      case "snapshot":
        handleSnapshot(msg.data);
        break;
      case "pong":
        break;
      default:
        console.log("Unknown message type:", msg.type);
    }
  }

  function handleReading(data) {
    const bt = data.biomarker_type;
    if (!biomarkerData[bt]) {
      biomarkerData[bt] = [];
    }

    const point = {
      value: data.value,
      timestamp: new Date(data.timestamp),
      anomaly: data.anomaly,
      unit: data.unit || UNITS[bt] || "",
    };

    biomarkerData[bt].push(point);

    // Trim to max points
    if (biomarkerData[bt].length > CHART_MAX_POINTS) {
      biomarkerData[bt] = biomarkerData[bt].slice(-CHART_MAX_POINTS);
    }

    // Update trend data if available
    if (data.trend) {
      currentTrend = data.trend;
    }

    dataPointCount++;
    updateVitalCard(bt, data.value, data.anomaly);
    updatePatientInfo();

    // Re-render chart if this is the currently selected biomarker
    if (bt === dom.chartBiomarker.value) {
      renderChart();
    }
  }

  function handleAlert(data) {
    alerts[data.alert_id] = {
      ...data,
      created_at: new Date(data.created_at),
    };
    renderAlerts();
    dom.alertCount.textContent = Object.keys(alerts).length;

    // Flash notification for high+ severity
    if (data.severity === "critical" || data.severity === "high") {
      showNotification(data.title, data.severity);
    }
  }

  function handleSnapshot(data) {
    const biomarkers = data.biomarkers || {};
    for (const [bt, readings] of Object.entries(biomarkers)) {
      biomarkerData[bt] = readings.map((r) => ({
        value: r.value,
        timestamp: new Date(r.timestamp),
        anomaly: null,
        unit: r.unit || UNITS[bt] || "",
      }));

      if (readings.length > 0) {
        const last = readings[readings.length - 1];
        updateVitalCard(bt, last.value, null);
      }
    }
    dataPointCount = Object.values(biomarkerData).reduce(
      (sum, arr) => sum + arr.length,
      0,
    );
    updatePatientInfo();
    renderChart();
  }

  // ----------------------------------------------------------------
  // Vital Signs Updates
  // ----------------------------------------------------------------

  function updateVitalCard(biomarkerType, value, anomaly) {
    const valueEl = document.getElementById(`vital-${biomarkerType}`);
    if (!valueEl) return;

    const card = valueEl.closest(".vital-card");
    const displayValue = formatValue(value, biomarkerType);

    valueEl.textContent = displayValue;
    valueEl.classList.add("value-change");
    setTimeout(() => valueEl.classList.remove("value-change"), 400);

    // Color coding based on normal range
    card.classList.remove("anomaly", "warning");
    valueEl.classList.remove("critical", "high", "normal");

    const range = NORMAL_RANGES[biomarkerType];
    if (range) {
      if (value < range[0] || value > range[1]) {
        const deviation =
          value < range[0]
            ? (range[0] - value) / (range[1] - range[0])
            : (value - range[1]) / (range[1] - range[0]);

        if (deviation > 0.3 || (anomaly && anomaly.severity === "critical")) {
          card.classList.add("anomaly");
          valueEl.classList.add("critical");
        } else {
          card.classList.add("warning");
          valueEl.classList.add("high");
        }
      } else {
        valueEl.classList.add("normal");
      }
    }
  }

  function updatePatientInfo() {
    dom.infoDataPoints.textContent = dataPointCount;
    dom.infoLastUpdate.textContent = new Date().toLocaleTimeString();
  }

  function formatValue(value, biomarkerType) {
    if (biomarkerType === "oxygen_sat" || biomarkerType === "heart_rate") {
      return Math.round(value);
    }
    if (biomarkerType === "troponin") {
      return value.toFixed(3);
    }
    return value.toFixed(1);
  }

  // ----------------------------------------------------------------
  // Chart Rendering (Canvas API)
  // ----------------------------------------------------------------

  function renderChart() {
    const canvas = dom.mainChart;
    const ctx = canvas.getContext("2d");

    // Handle DPI scaling
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    const bt = dom.chartBiomarker.value;
    const data = biomarkerData[bt] || [];
    const range = NORMAL_RANGES[bt] || [0, 100];

    // Colors from CSS variables
    const isDark = document.body.classList.contains("dark-mode");
    const bgColor = isDark ? "#161b22" : "#f0f2f5";
    const gridColor = isDark ? "#30363d" : "#e2e8f0";
    const textColor = isDark ? "#8b949e" : "#4a5568";
    const lineColor = "#3182ce";
    const normalBandColor = isDark
      ? "rgba(56,161,105,0.08)"
      : "rgba(56,161,105,0.1)";
    const normalBorderColor = "rgba(56,161,105,0.4)";
    const anomalyColor = "#e53e3e";

    // Clear
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, width, height);

    const padding = { top: 20, right: 20, bottom: 40, left: 55 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    if (data.length < 2) {
      ctx.fillStyle = textColor;
      ctx.font = "14px -apple-system, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Waiting for data...", width / 2, height / 2);
      updateTrendInfo(null);
      return;
    }

    // Determine Y range
    const values = data.map((d) => d.value);
    let yMin = Math.min(...values, range[0]) * 0.95;
    let yMax = Math.max(...values, range[1]) * 1.05;
    if (yMax - yMin < 1) {
      yMin -= 1;
      yMax += 1;
    }

    // Time range filter
    const hoursFilter = parseInt(dom.chartTimeRange.value, 10);
    const timeMin = new Date(Date.now() - hoursFilter * 3600 * 1000);
    const filteredData = data.filter((d) => d.timestamp >= timeMin);
    const displayData = filteredData.length >= 2 ? filteredData : data;

    const tMin = displayData[0].timestamp.getTime();
    const tMax = displayData[displayData.length - 1].timestamp.getTime();
    const tRange = tMax - tMin || 1;

    // Mapping functions
    const mapX = (t) => padding.left + ((t - tMin) / tRange) * chartW;
    const mapY = (v) => padding.top + (1 - (v - yMin) / (yMax - yMin)) * chartH;

    // Draw normal range band
    const bandTop = mapY(range[1]);
    const bandBottom = mapY(range[0]);
    ctx.fillStyle = normalBandColor;
    ctx.fillRect(padding.left, bandTop, chartW, bandBottom - bandTop);

    // Normal range borders
    ctx.strokeStyle = normalBorderColor;
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(padding.left, bandTop);
    ctx.lineTo(padding.left + chartW, bandTop);
    ctx.moveTo(padding.left, bandBottom);
    ctx.lineTo(padding.left + chartW, bandBottom);
    ctx.stroke();
    ctx.setLineDash([]);

    // Grid lines (Y axis)
    const yTicks = 5;
    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 0.5;
    ctx.fillStyle = textColor;
    ctx.font = "11px -apple-system, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    for (let i = 0; i <= yTicks; i++) {
      const val = yMin + (yMax - yMin) * (i / yTicks);
      const y = mapY(val);
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + chartW, y);
      ctx.stroke();
      ctx.fillText(val.toFixed(1), padding.left - 8, y);
    }

    // X axis labels
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const xTicks = Math.min(6, displayData.length);
    for (let i = 0; i < xTicks; i++) {
      const idx = Math.floor((i / (xTicks - 1)) * (displayData.length - 1));
      const d = displayData[idx];
      const x = mapX(d.timestamp.getTime());
      const label = d.timestamp.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      });
      ctx.fillText(label, x, height - padding.bottom + 8);
    }

    // Draw data line
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.beginPath();
    displayData.forEach((d, i) => {
      const x = mapX(d.timestamp.getTime());
      const y = mapY(d.value);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Fill under the line
    ctx.fillStyle = isDark ? "rgba(49,130,206,0.15)" : "rgba(49,130,206,0.08)";
    ctx.beginPath();
    displayData.forEach((d, i) => {
      const x = mapX(d.timestamp.getTime());
      const y = mapY(d.value);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.lineTo(mapX(tMax), padding.top + chartH);
    ctx.lineTo(mapX(tMin), padding.top + chartH);
    ctx.closePath();
    ctx.fill();

    // Draw anomaly points
    displayData.forEach((d) => {
      if (d.anomaly && d.anomaly.is_anomaly) {
        const x = mapX(d.timestamp.getTime());
        const y = mapY(d.value);
        ctx.fillStyle = anomalyColor;
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    });

    // Draw latest point
    if (displayData.length > 0) {
      const last = displayData[displayData.length - 1];
      const x = mapX(last.timestamp.getTime());
      const y = mapY(last.value);
      ctx.fillStyle = lineColor;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Y axis label
    ctx.save();
    ctx.translate(12, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = textColor;
    ctx.font = "11px -apple-system, sans-serif";
    ctx.fillText(UNITS[bt] || "", 0, 0);
    ctx.restore();

    // Update trend info
    updateTrendInfo(currentTrend);
  }

  function updateTrendInfo(trend) {
    if (!trend) {
      dom.trendDirection.textContent = "--";
      dom.trendDirection.className = "trend-direction";
      dom.trendRate.textContent = "--";
      dom.trendPredicted.textContent = "--";
      return;
    }

    dom.trendDirection.textContent = trend.direction;
    dom.trendDirection.className = `trend-direction ${trend.direction}`;
    dom.trendRate.textContent = `${trend.rate_of_change >= 0 ? "+" : ""}${trend.rate_of_change.toFixed(4)}/hr`;

    if (trend.predicted_value_24h !== null) {
      const pred = trend.predicted_value_24h.toFixed(2);
      const exitClass = trend.predicted_exit_normal ? " worsening" : "";
      dom.trendPredicted.textContent = pred;
      dom.trendPredicted.className = `trend-predicted${exitClass}`;
    } else {
      dom.trendPredicted.textContent = "--";
    }
  }

  // ----------------------------------------------------------------
  // Alerts
  // ----------------------------------------------------------------

  function renderAlerts() {
    const container = dom.alertList;
    const sortedAlerts = Object.values(alerts).sort(
      (a, b) => b.created_at - a.created_at,
    );

    const filtered =
      currentAlertFilter === "all"
        ? sortedAlerts
        : sortedAlerts.filter((a) => a.severity === currentAlertFilter);

    if (filtered.length === 0) {
      container.innerHTML = '<div class="alert-empty">No active alerts</div>';
      return;
    }

    container.innerHTML = filtered
      .map(
        (alert) => `
            <div class="alert-item ${alert.severity}">
                <div class="alert-header">
                    <span class="alert-title">${escapeHtml(alert.title)}</span>
                    <span class="alert-severity ${alert.severity}">${alert.severity.toUpperCase()}</span>
                </div>
                <div class="alert-message">${escapeHtml(alert.message)}</div>
                <div class="alert-time">${alert.created_at.toLocaleString()}</div>
                <div class="alert-actions">
                    <button class="btn btn-sm" onclick="BiomarkerDash.acknowledgeAlert('${alert.alert_id}')">Acknowledge</button>
                </div>
            </div>
        `,
      )
      .join("");
  }

  // ----------------------------------------------------------------
  // Demo Data
  // ----------------------------------------------------------------

  async function loadDemoData() {
    const patientId = dom.patientSelect.value || "P001";
    selectedPatientId = patientId;

    showNotification("Generating demo data...", "info");

    // Register patient
    try {
      await fetch("/api/patients", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient_id: patientId,
          age: 62,
          sex: "male",
          conditions: ["hypertension", "diabetes_type2"],
          medications: ["metformin", "lisinopril"],
        }),
      });
    } catch (e) {
      console.log("Could not register patient (backend may not be running)");
    }

    // Generate synthetic data locally for demo visualization
    const biomarkerTypes = [
      "heart_rate",
      "blood_pressure_sys",
      "oxygen_sat",
      "temperature",
      "glucose",
      "potassium",
    ];

    const baseValues = {
      heart_rate: 75,
      blood_pressure_sys: 125,
      oxygen_sat: 97,
      temperature: 98.6,
      glucose: 110,
      potassium: 4.2,
    };

    const noise = {
      heart_rate: 8,
      blood_pressure_sys: 10,
      oxygen_sat: 1.5,
      temperature: 0.4,
      glucose: 15,
      potassium: 0.3,
    };

    const now = Date.now();
    for (const bt of biomarkerTypes) {
      biomarkerData[bt] = [];
      for (let i = 0; i < 60; i++) {
        const t = now - (60 - i) * 60 * 1000;
        const base = baseValues[bt];
        const n = noise[bt];
        const val = base + (Math.random() - 0.5) * 2 * n;

        // Inject an anomaly at point 45
        let anomaly = null;
        let displayVal = val;
        if (i === 45 && (bt === "heart_rate" || bt === "glucose")) {
          displayVal = bt === "heart_rate" ? 145 : 250;
          anomaly = {
            is_anomaly: true,
            severity: "high",
            score: 0.85,
          };
        }

        biomarkerData[bt].push({
          value: displayVal,
          timestamp: new Date(t),
          anomaly: anomaly,
          unit: UNITS[bt],
        });
      }

      // Update vital card with latest value
      const latest = biomarkerData[bt][biomarkerData[bt].length - 1];
      updateVitalCard(bt, latest.value, latest.anomaly);
    }

    // Create demo alerts
    const demoAlerts = [
      {
        alert_id: "demo-1",
        patient_id: patientId,
        biomarker_type: "heart_rate",
        severity: "high",
        title: "Elevated Heart Rate",
        message:
          "Heart rate = 145 bpm exceeds normal range [60, 100]. Evaluate for tachycardia.",
        value: 145,
        created_at: new Date(now - 15 * 60000),
      },
      {
        alert_id: "demo-2",
        patient_id: patientId,
        biomarker_type: "glucose",
        severity: "medium",
        title: "Elevated Glucose",
        message:
          "Glucose = 250 mg/dL is above normal range [70, 100]. Monitor for hyperglycemia.",
        value: 250,
        created_at: new Date(now - 15 * 60000),
      },
      {
        alert_id: "demo-3",
        patient_id: patientId,
        biomarker_type: "blood_pressure_sys",
        severity: "low",
        title: "Trend Alert: BP Systolic worsening",
        message:
          "Systolic BP trending upward (rate: +0.12/hr). Predicted to exit range within 24h.",
        value: 132,
        created_at: new Date(now - 5 * 60000),
      },
    ];

    for (const a of demoAlerts) {
      alerts[a.alert_id] = a;
    }

    currentTrend = {
      direction: "worsening",
      rate_of_change: 0.12,
      predicted_value_24h: 132.5,
      predicted_exit_normal: true,
      confidence: 0.72,
    };

    dataPointCount = Object.values(biomarkerData).reduce(
      (sum, arr) => sum + arr.length,
      0,
    );

    dom.patientInfo.style.display = "block";
    dom.infoPatientId.textContent = patientId;
    dom.infoStatus.textContent = "Demo Mode";
    updatePatientInfo();
    renderAlerts();
    renderChart();
    dom.alertCount.textContent = Object.keys(alerts).length;

    showNotification("Demo data loaded successfully", "success");
  }

  // ----------------------------------------------------------------
  // Metrics Polling
  // ----------------------------------------------------------------

  async function pollMetrics() {
    try {
      const resp = await fetch("/api/metrics");
      if (!resp.ok) return;
      const data = await resp.json();
      dom.metricProcessed.textContent = data.processed || 0;
      dom.metricQueue.textContent = data.queue_size || 0;
      dom.metricWS.textContent = data.ws_connections || 0;
      dom.metricErrors.textContent = data.errors || 0;
    } catch {
      // Silently ignore fetch errors (backend might not be running)
    }
  }

  // ----------------------------------------------------------------
  // Alert Actions
  // ----------------------------------------------------------------

  async function acknowledgeAlert(alertId) {
    try {
      await fetch(`/api/alerts/${alertId}/acknowledge?user=dashboard_user`, {
        method: "POST",
      });
    } catch {
      // OK if backend is not running
    }

    delete alerts[alertId];
    renderAlerts();
    dom.alertCount.textContent = Object.keys(alerts).length;
  }

  // ----------------------------------------------------------------
  // Utilities
  // ----------------------------------------------------------------

  function clearData() {
    Object.keys(biomarkerData).forEach((k) => delete biomarkerData[k]);
    Object.keys(alerts).forEach((k) => delete alerts[k]);
    dataPointCount = 0;
    currentTrend = null;
    dom.alertCount.textContent = "0";
    renderAlerts();
    renderChart();

    // Reset vital cards
    document.querySelectorAll(".vital-value").forEach((el) => {
      el.textContent = "--";
      el.className = "vital-value";
    });
    document.querySelectorAll(".vital-card").forEach((el) => {
      el.classList.remove("anomaly", "warning");
    });
  }

  function toggleDarkMode() {
    document.body.classList.toggle("dark-mode");
    localStorage.setItem(
      "darkMode",
      document.body.classList.contains("dark-mode"),
    );
    renderChart();
  }

  function updateClock() {
    dom.clock.textContent = new Date().toLocaleTimeString();
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function showNotification(message, type) {
    // Simple notification via console (could be extended to toast UI)
    const prefix =
      type === "success" ? "[OK]" : type === "warning" ? "[WARN]" : "[INFO]";
    console.log(`${prefix} ${message}`);
  }

  // ----------------------------------------------------------------
  // Expose public API
  // ----------------------------------------------------------------

  window.BiomarkerDash = {
    acknowledgeAlert: acknowledgeAlert,
  };

  // Boot
  document.addEventListener("DOMContentLoaded", init);
})();
