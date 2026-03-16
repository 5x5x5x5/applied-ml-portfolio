/**
 * PharmaAgents Frontend - Multi-Agent Chat Interface
 *
 * Manages WebSocket connection, agent conversation rendering,
 * workflow parameter collection, and structured output display.
 */

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const AGENT_COLORS = {
  LiteratureAgent: {
    bg: "#2563eb",
    label: "Literature",
    initials: "LR",
    persona: "Dr. Reeves",
  },
  SafetyAgent: {
    bg: "#dc2626",
    label: "Safety",
    initials: "SA",
    persona: "Dr. Okafor",
  },
  ChemistryAgent: {
    bg: "#059669",
    label: "Chemistry",
    initials: "CH",
    persona: "Dr. Patel",
  },
  RegulatoryAgent: {
    bg: "#d97706",
    label: "Regulatory",
    initials: "RG",
    persona: "Dr. Marsh",
  },
  system: {
    bg: "#7c3aed",
    label: "System",
    initials: "PA",
    persona: "Coordinator",
  },
};

const API_BASE = window.location.origin;
const WS_URL = `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/ws/session`;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let ws = null;
let selectedAgent = ""; // '' = auto-route
let currentWorkflow = null;
let isProcessing = false;

// ---------------------------------------------------------------------------
// DOM References
// ---------------------------------------------------------------------------

const conversation = document.getElementById("conversation");
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");
const charCount = document.getElementById("charCount");
const selectedAgentLabel = document.getElementById("selectedAgent");
const connectionStatus = document.getElementById("connectionStatus");
const resultsPanel = document.getElementById("resultsPanel");
const panelContent = document.getElementById("panelContent");
const panelToggle = document.getElementById("panelToggle");
const workflowModal = document.getElementById("workflowModal");
const modalTitle = document.getElementById("modalTitle");
const modalBody = document.getElementById("modalBody");
const modalCancel = document.getElementById("modalCancel");
const modalClose = document.getElementById("modalClose");
const modalSubmit = document.getElementById("modalSubmit");

// ---------------------------------------------------------------------------
// WebSocket Connection
// ---------------------------------------------------------------------------

function connectWebSocket() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    updateConnectionStatus(true);
    console.log("[WS] Connected");
  };

  ws.onclose = () => {
    updateConnectionStatus(false);
    console.log("[WS] Disconnected. Reconnecting in 3s...");
    setTimeout(connectWebSocket, 3000);
  };

  ws.onerror = (err) => {
    console.error("[WS] Error:", err);
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    handleWSMessage(data);
  };
}

function updateConnectionStatus(connected) {
  const dot = connectionStatus.querySelector(".status-dot");
  const text = connectionStatus.querySelector(".status-text");
  dot.className = `status-dot ${connected ? "connected" : "disconnected"}`;
  text.textContent = connected ? "Connected" : "Disconnected";
}

// ---------------------------------------------------------------------------
// Message Handling
// ---------------------------------------------------------------------------

function handleWSMessage(data) {
  switch (data.type) {
    case "status":
      addSystemMessage(data.message);
      break;

    case "agent_start":
      addThinkingIndicator(data.agent, data.task_id);
      break;

    case "agent_complete":
      removeThinkingIndicator(data.agent);
      addAgentMessage(data.agent, data.response);
      updateResultsPanel(data.agent, data.response);
      break;

    case "synthesis":
      removePendingSynthesis();
      addSynthesisMessage(data.text, data.conflicts);
      setProcessing(false);
      break;

    case "workflow_step_start":
      addSystemMessage(
        `Step ${data.step}: ${data.description} (${data.agent})`,
      );
      addThinkingIndicator(data.agent, `wf-step-${data.step}`);
      break;

    case "workflow_step_complete":
      removeThinkingIndicator(data.agent);
      break;

    case "workflow_complete":
      addSynthesisMessage(data.synthesis, []);
      addSystemMessage(
        `Workflow completed in ${data.processing_time_s.toFixed(1)}s`,
      );
      setProcessing(false);
      break;

    case "error":
      addSystemMessage(`Error: ${data.message}`, true);
      setProcessing(false);
      break;

    default:
      console.log("[WS] Unhandled message type:", data.type, data);
  }
}

// ---------------------------------------------------------------------------
// Sending Messages
// ---------------------------------------------------------------------------

function sendQuery() {
  const question = queryInput.value.trim();
  if (!question || isProcessing) return;
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    addSystemMessage("Not connected. Falling back to REST API...", true);
    sendQueryREST(question);
    return;
  }

  addUserMessage(question);
  setProcessing(true);

  const payload = { type: "query", question };
  if (selectedAgent) {
    payload.agent = selectedAgent;
  }

  ws.send(JSON.stringify(payload));
  queryInput.value = "";
  updateCharCount();
}

async function sendQueryREST(question) {
  addUserMessage(question);
  setProcessing(true);

  try {
    const response = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        agent: selectedAgent || null,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      addSystemMessage(`Error: ${error.detail}`, true);
      setProcessing(false);
      return;
    }

    const data = await response.json();

    for (const agentResp of data.agent_responses) {
      addAgentMessage(agentResp.agent_name, agentResp);
      updateResultsPanel(agentResp.agent_name, agentResp);
    }

    if (data.synthesis) {
      addSynthesisMessage(data.synthesis, data.conflicts || []);
    }
  } catch (err) {
    addSystemMessage(`Network error: ${err.message}`, true);
  }

  setProcessing(false);
  queryInput.value = "";
  updateCharCount();
}

function sendWorkflow(workflowName, parameters) {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    addSystemMessage("WebSocket not connected. Cannot execute workflow.", true);
    return;
  }

  setProcessing(true);
  addUserMessage(`[Workflow: ${workflowName.replace(/_/g, " ")}]`);

  ws.send(
    JSON.stringify({
      type: "workflow",
      workflow_name: workflowName,
      parameters,
    }),
  );
}

// ---------------------------------------------------------------------------
// UI Rendering
// ---------------------------------------------------------------------------

function clearWelcome() {
  const welcome = conversation.querySelector(".welcome-message");
  if (welcome) welcome.remove();
}

function addUserMessage(text) {
  clearWelcome();
  const div = document.createElement("div");
  div.className = "message message-user";
  div.innerHTML = `<div class="message-content">${escapeHtml(text)}</div>`;
  conversation.appendChild(div);
  scrollToBottom();
}

function addAgentMessage(agentName, response) {
  clearWelcome();
  const config = AGENT_COLORS[agentName] || AGENT_COLORS.system;
  const div = document.createElement("div");
  div.className = "message message-agent";
  div.dataset.agent = agentName;

  const toolsHtml =
    response.tools_used && response.tools_used.length
      ? `<div class="tools-used">${response.tools_used
          .map((t) => `<span class="tool-tag">${escapeHtml(t)}</span>`)
          .join("")}</div>`
      : "";

  const confidencePct = Math.round((response.confidence || 0) * 100);

  div.innerHTML = `
        <div class="agent-avatar ${config.label.toLowerCase()}"
             style="background: ${config.bg};">${config.initials}</div>
        <div class="message-content">
            <div class="agent-label">
                <span style="color: ${config.bg};">${config.label} Agent</span>
                <span class="agent-role">${config.persona}</span>
            </div>
            <div class="message-text">${formatAgentText(response.text || "")}</div>
            ${toolsHtml}
            <div class="message-meta">
                <span>Confidence: ${confidencePct}%</span>
                <span>${(response.processing_time_s || 0).toFixed(1)}s</span>
            </div>
        </div>
    `;
  conversation.appendChild(div);
  scrollToBottom();
}

function addSystemMessage(text, isError = false) {
  clearWelcome();
  const div = document.createElement("div");
  div.className = "message message-agent";

  const config = AGENT_COLORS.system;
  div.innerHTML = `
        <div class="agent-avatar system" style="background: ${config.bg};">${config.initials}</div>
        <div class="message-content" style="${isError ? "border-color: var(--accent-red);" : ""}">
            <div class="agent-label">
                <span style="color: ${isError ? "var(--accent-red)" : config.bg};">
                    ${isError ? "Error" : "System"}
                </span>
            </div>
            <div class="message-text">${escapeHtml(text)}</div>
        </div>
    `;
  conversation.appendChild(div);
  scrollToBottom();
}

function addSynthesisMessage(text, conflicts) {
  const div = document.createElement("div");
  div.className = "message message-agent";
  div.dataset.synthesis = "true";

  let conflictHtml = "";
  if (conflicts && conflicts.length > 0) {
    conflictHtml = `
            <div style="margin-top: 12px;">
                ${conflicts
                  .map(
                    (c) => `
                    <div class="conflict-badge" style="margin-bottom: 6px;">
                        Conflict: ${escapeHtml(c.conflict_description || "")}
                    </div>
                `,
                  )
                  .join("")}
            </div>
        `;
  }

  div.innerHTML = `
        <div class="agent-avatar system" style="background: var(--accent-purple);">&#9883;</div>
        <div class="message-content synthesis-block">
            <h4>Integrated Analysis</h4>
            <div class="message-text">${formatAgentText(text)}</div>
            ${conflictHtml}
        </div>
    `;
  conversation.appendChild(div);
  scrollToBottom();
}

function removePendingSynthesis() {
  // Remove any pending synthesis indicators
}

function addThinkingIndicator(agentName, taskId) {
  const config = AGENT_COLORS[agentName] || AGENT_COLORS.system;
  const div = document.createElement("div");
  div.className = "message message-thinking";
  div.id = `thinking-${agentName}`;

  div.innerHTML = `
        <div class="agent-avatar ${config.label.toLowerCase()}"
             style="background: ${config.bg};">${config.initials}</div>
        <span style="color: var(--text-muted); font-size: 12px;">
            ${config.label} Agent is thinking...
        </span>
        <div class="thinking-dots">
            <span></span><span></span><span></span>
        </div>
    `;
  conversation.appendChild(div);
  scrollToBottom();
}

function removeThinkingIndicator(agentName) {
  const el = document.getElementById(`thinking-${agentName}`);
  if (el) el.remove();
}

// ---------------------------------------------------------------------------
// Results Panel
// ---------------------------------------------------------------------------

function updateResultsPanel(agentName, response) {
  const config = AGENT_COLORS[agentName] || AGENT_COLORS.system;

  // Remove placeholder
  const placeholder = panelContent.querySelector(".panel-placeholder");
  if (placeholder) placeholder.remove();

  const section = document.createElement("div");
  section.className = "data-section";

  const confidencePct = Math.round((response.confidence || 0) * 100);
  const confidenceClass =
    confidencePct >= 70 ? "high" : confidencePct >= 40 ? "medium" : "low";
  const confidenceColor =
    confidenceClass === "high"
      ? "var(--accent-green)"
      : confidenceClass === "medium"
        ? "var(--accent-amber)"
        : "var(--accent-red)";

  let dataHtml = `
        <div class="data-item">
            <span class="label">Confidence</span>
            <span class="value ${confidenceClass}">${confidencePct}%</span>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill"
                 style="width: ${confidencePct}%; background: ${confidenceColor};"></div>
        </div>
        <div class="data-item">
            <span class="label">Processing Time</span>
            <span class="value">${(response.processing_time_s || 0).toFixed(2)}s</span>
        </div>
        <div class="data-item">
            <span class="label">Tools Used</span>
            <span class="value">${(response.tools_used || []).length}</span>
        </div>
    `;

  // Show structured data if available
  if (
    response.structured_data &&
    Object.keys(response.structured_data).length > 0
  ) {
    dataHtml += renderStructuredData(response.structured_data);
  }

  section.innerHTML = `
        <h3 style="color: ${config.bg};">${config.label} Agent</h3>
        ${dataHtml}
    `;

  panelContent.appendChild(section);
}

function renderStructuredData(data, depth = 0) {
  if (depth > 3) return '<span class="value">...</span>';

  let html = "";
  for (const [key, value] of Object.entries(data)) {
    const label = key.replace(/_/g, " ");
    if (typeof value === "object" && value !== null && !Array.isArray(value)) {
      html += `<div class="data-item" style="padding-left: ${depth * 12}px;">
                <span class="label" style="font-weight: 600;">${label}</span>
            </div>`;
      html += renderStructuredData(value, depth + 1);
    } else if (Array.isArray(value)) {
      html += `<div class="data-item" style="padding-left: ${depth * 12}px;">
                <span class="label">${label}</span>
                <span class="value">[${value.length} items]</span>
            </div>`;
    } else {
      let valueClass = "";
      const strVal = String(value);
      if (
        ["high", "favorable", "pass", "true", "low_risk"].some((v) =>
          strVal.toLowerCase().includes(v),
        )
      ) {
        valueClass = "high";
      } else if (
        ["critical", "unfavorable", "fail", "high_risk", "signal"].some((v) =>
          strVal.toLowerCase().includes(v),
        )
      ) {
        valueClass = "signal";
      }

      html += `<div class="data-item" style="padding-left: ${depth * 12}px;">
                <span class="label">${label}</span>
                <span class="value ${valueClass}">${escapeHtml(strVal)}</span>
            </div>`;
    }
  }
  return html;
}

// ---------------------------------------------------------------------------
// Workflow Modal
// ---------------------------------------------------------------------------

async function openWorkflowModal(workflowName) {
  currentWorkflow = workflowName;
  const displayName = workflowName
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
  modalTitle.textContent = displayName;

  // Fetch required parameters
  try {
    const response = await fetch(
      `${API_BASE}/workflow/${workflowName}/parameters`,
    );
    const data = await response.json();
    const params = data.required_parameters || [];

    // Default values for common parameters
    const defaults = {
      drug_name: "osimertinib",
      indication: "non-small cell lung cancer",
      mechanism: "EGFR T790M inhibitor",
      drug_class: "small_molecule",
      therapeutic_area: "oncology",
      smiles: "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)NC(=O)C=C",
      adverse_event: "QT prolongation",
    };

    modalBody.innerHTML = params
      .map(
        (param) => `
            <div class="form-group">
                <label for="param-${param}">${param.replace(/_/g, " ")}</label>
                <input type="text" id="param-${param}" name="${param}"
                       value="${defaults[param] || ""}"
                       placeholder="Enter ${param.replace(/_/g, " ")}">
            </div>
        `,
      )
      .join("");

    workflowModal.classList.add("active");
  } catch (err) {
    addSystemMessage(
      `Failed to load workflow parameters: ${err.message}`,
      true,
    );
  }
}

function closeWorkflowModal() {
  workflowModal.classList.remove("active");
  currentWorkflow = null;
}

function submitWorkflow() {
  if (!currentWorkflow) return;

  const inputs = modalBody.querySelectorAll("input");
  const parameters = {};
  inputs.forEach((input) => {
    parameters[input.name] = input.value;
  });

  closeWorkflowModal();
  sendWorkflow(currentWorkflow, parameters);
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function formatAgentText(text) {
  // Basic markdown-like formatting
  return escapeHtml(text)
    .replace(
      /^### (.+)$/gm,
      '<h4 style="margin: 12px 0 6px; color: var(--text-primary);">$1</h4>',
    )
    .replace(
      /^## (.+)$/gm,
      '<h3 style="margin: 14px 0 8px; color: var(--text-primary); font-size: 15px;">$1</h3>',
    )
    .replace(
      /^# (.+)$/gm,
      '<h2 style="margin: 16px 0 10px; color: var(--text-primary); font-size: 17px;">$1</h2>',
    )
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(
      /`([^`]+)`/g,
      '<code style="background: var(--bg-primary); padding: 2px 6px; border-radius: 3px; font-size: 12px;">$1</code>',
    )
    .replace(
      /^- (.+)$/gm,
      '<div style="padding-left: 16px; margin: 2px 0;">&#8226; $1</div>',
    )
    .replace(/\n/g, "<br>");
}

function scrollToBottom() {
  setTimeout(() => {
    conversation.scrollTop = conversation.scrollHeight;
  }, 50);
}

function setProcessing(state) {
  isProcessing = state;
  sendBtn.disabled = state;
  queryInput.disabled = state;
}

function updateCharCount() {
  charCount.textContent = queryInput.value.length;
}

// ---------------------------------------------------------------------------
// Event Listeners
// ---------------------------------------------------------------------------

// Send button
sendBtn.addEventListener("click", sendQuery);

// Enter to send (Shift+Enter for newline)
queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendQuery();
  }
});

queryInput.addEventListener("input", updateCharCount);

// Agent selection
document.querySelectorAll(".agent-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document
      .querySelectorAll(".agent-btn")
      .forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    selectedAgent = btn.dataset.agent;
    const label = selectedAgent
      ? `Agent: ${AGENT_COLORS[selectedAgent]?.label || selectedAgent}`
      : "Mode: Auto-Route";
    selectedAgentLabel.textContent = label;
  });
});

// Workflow buttons
document.querySelectorAll(".workflow-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    openWorkflowModal(btn.dataset.workflow);
  });
});

// Example queries
document.querySelectorAll(".example-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    queryInput.value = btn.dataset.query;
    updateCharCount();
    queryInput.focus();
  });
});

// Panel toggle
panelToggle.addEventListener("click", () => {
  resultsPanel.classList.toggle("collapsed");
  panelToggle.textContent = resultsPanel.classList.contains("collapsed")
    ? "\u25B6"
    : "\u25C4";
});

// Modal controls
modalCancel.addEventListener("click", closeWorkflowModal);
modalClose.addEventListener("click", closeWorkflowModal);
modalSubmit.addEventListener("click", submitWorkflow);
workflowModal.addEventListener("click", (e) => {
  if (e.target === workflowModal) closeWorkflowModal();
});

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

connectWebSocket();
