/**
 * PharmAssistAI - Chat Interface JavaScript
 *
 * Handles WebSocket connection for streaming responses, message rendering
 * with markdown support, source citation expansion, and file upload.
 */

// ── Configuration ──────────────────────────────────────────────────────────

const API_BASE = window.location.origin;
const WS_URL = `ws://${window.location.host}/ws/chat`;

// ── State ──────────────────────────────────────────────────────────────────

let ws = null;
let sessionId = null;
let isStreaming = false;
let currentAssistantMessage = null;
let currentAssistantText = "";
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;

// ── Initialization ─────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  initWebSocket();
  initTextarea();
  initNavigation();
  initUpload();
  initSearchInput();
});

// ── WebSocket Connection ───────────────────────────────────────────────────

function initWebSocket() {
  try {
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      updateConnectionStatus(true);
      reconnectAttempts = 0;
      console.log("WebSocket connected");
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleWebSocketMessage(data);
    };

    ws.onclose = () => {
      updateConnectionStatus(false);
      console.log("WebSocket disconnected");
      attemptReconnect();
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      updateConnectionStatus(false);
    };
  } catch (e) {
    console.error("Failed to create WebSocket:", e);
    updateConnectionStatus(false);
  }
}

function attemptReconnect() {
  if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
    reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
    console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts})`);
    setTimeout(initWebSocket, delay);
  }
}

function updateConnectionStatus(connected) {
  const statusEl = document.getElementById("connectionStatus");
  const dot = statusEl.querySelector(".status-dot");
  const text = statusEl.querySelector("span:last-child");

  if (connected) {
    dot.className = "status-dot connected";
    text.textContent = "Connected";
  } else {
    dot.className = "status-dot disconnected";
    text.textContent = "Disconnected";
  }
}

function handleWebSocketMessage(data) {
  switch (data.type) {
    case "answer_chunk":
      handleAnswerChunk(data.content);
      break;
    case "answer_complete":
      handleAnswerComplete(data.metadata);
      break;
    case "citation":
      handleCitation(data);
      break;
    case "error":
      handleError(data.content);
      break;
  }
}

// ── Message Handling ───────────────────────────────────────────────────────

function sendMessage() {
  const input = document.getElementById("questionInput");
  const question = input.value.trim();

  if (!question || isStreaming) return;

  // Hide welcome message
  const welcome = document.getElementById("welcomeMessage");
  if (welcome) welcome.style.display = "none";

  // Add user message
  addMessage("user", question);

  // Clear input
  input.value = "";
  updateCharCount();
  autoResizeTextarea(input);

  // Try WebSocket first, fall back to REST API
  if (ws && ws.readyState === WebSocket.OPEN) {
    sendViaWebSocket(question);
  } else {
    sendViaRest(question);
  }
}

function sendViaWebSocket(question) {
  isStreaming = true;
  currentAssistantText = "";
  currentAssistantMessage = addMessage("assistant", "", true);

  const message = {
    type: "question",
    content: question,
    metadata: {},
  };

  ws.send(JSON.stringify(message));
  updateSendButton();
}

async function sendViaRest(question) {
  isStreaming = true;
  currentAssistantText = "";
  currentAssistantMessage = addMessage("assistant", "", true);
  updateSendButton();

  try {
    const response = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: question,
        n_results: 5,
        stream: false,
        session_id: sessionId,
      }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();
    sessionId = data.session_id;

    // Display the answer
    removeTypingIndicator();
    updateAssistantMessage(data.answer);

    // Add citations if present
    if (data.citations && data.citations.length > 0) {
      addCitations(currentAssistantMessage, data.citations);
    }
  } catch (error) {
    removeTypingIndicator();
    updateAssistantMessage(`Sorry, I encountered an error: ${error.message}`);
  } finally {
    isStreaming = false;
    currentAssistantMessage = null;
    updateSendButton();
  }
}

function handleAnswerChunk(content) {
  currentAssistantText += content;
  removeTypingIndicator();
  updateAssistantMessage(currentAssistantText);
  scrollToBottom();
}

function handleAnswerComplete(metadata) {
  isStreaming = false;
  if (metadata && metadata.session_id) {
    sessionId = metadata.session_id;
  }
  currentAssistantMessage = null;
  updateSendButton();
}

function handleCitation(data) {
  if (currentAssistantMessage && data.metadata) {
    // Citations are handled in the answer text for WebSocket mode
  }
}

function handleError(message) {
  isStreaming = false;
  removeTypingIndicator();
  if (currentAssistantMessage) {
    updateAssistantMessage(`Error: ${message}`);
  } else {
    addMessage("assistant", `Error: ${message}`);
  }
  currentAssistantMessage = null;
  updateSendButton();
}

// ── DOM Manipulation ───────────────────────────────────────────────────────

function addMessage(role, content, showTyping = false) {
  const container = document.getElementById("chatContainer");

  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.innerHTML = role === "assistant" ? "&#9764;" : "&#128100;";

  const contentDiv = document.createElement("div");
  contentDiv.className = "message-content";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";

  if (showTyping) {
    bubble.innerHTML = createTypingIndicator();
  } else {
    bubble.innerHTML = renderMarkdown(content);
  }

  contentDiv.appendChild(bubble);
  messageDiv.appendChild(avatar);
  messageDiv.appendChild(contentDiv);
  container.appendChild(messageDiv);

  scrollToBottom();
  return messageDiv;
}

function updateAssistantMessage(content) {
  if (!currentAssistantMessage) return;

  const bubble = currentAssistantMessage.querySelector(".message-bubble");
  if (bubble) {
    bubble.innerHTML = renderMarkdown(content);
  }
}

function removeTypingIndicator() {
  if (!currentAssistantMessage) return;

  const indicator = currentAssistantMessage.querySelector(".typing-indicator");
  if (indicator) {
    indicator.remove();
  }
}

function createTypingIndicator() {
  return '<div class="typing-indicator"><span></span><span></span><span></span></div>';
}

function addCitations(messageEl, citations) {
  if (!messageEl || !citations || citations.length === 0) return;

  const contentDiv = messageEl.querySelector(".message-content");
  const citDiv = document.createElement("div");
  citDiv.className = "citations-container";

  const toggleBtn = document.createElement("button");
  toggleBtn.className = "citations-toggle";
  toggleBtn.innerHTML = `&#128209; ${citations.length} source${citations.length > 1 ? "s" : ""}`;
  toggleBtn.onclick = () => {
    const list = citDiv.querySelector(".citations-list");
    list.classList.toggle("expanded");
    toggleBtn.innerHTML = list.classList.contains("expanded")
      ? `&#128209; Hide sources`
      : `&#128209; ${citations.length} source${citations.length > 1 ? "s" : ""}`;
  };

  const list = document.createElement("div");
  list.className = "citations-list";

  citations.forEach((cit) => {
    const item = document.createElement("div");
    item.className = "citation-item";
    item.innerHTML = `
            <span class="citation-id">[${cit.citation_id}]</span>
            ${cit.drug_name ? `<span class="citation-drug">${cit.drug_name}</span>` : ""}
            ${cit.section_type ? `<span class="citation-section"> - ${cit.section_type}</span>` : ""}
            <span class="citation-relevance">${(cit.relevance_score * 100).toFixed(0)}% relevant</span>
            <div style="margin-top: 4px; font-size: 0.78rem;">${cit.excerpt || ""}</div>
        `;
    list.appendChild(item);
  });

  citDiv.appendChild(toggleBtn);
  citDiv.appendChild(list);
  contentDiv.appendChild(citDiv);
}

function scrollToBottom() {
  const container = document.getElementById("chatContainer");
  container.scrollTop = container.scrollHeight;
}

// ── Markdown Rendering ─────────────────────────────────────────────────────

function renderMarkdown(text) {
  if (!text) return "";

  let html = escapeHtml(text);

  // Bold: **text** or __text__
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/__(.+?)__/g, "<strong>$1</strong>");

  // Italic: *text* or _text_
  html = html.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, "<em>$1</em>");

  // Inline code: `code`
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

  // Headers: ### Header
  html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
  html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
  html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");

  // Unordered lists: - item or * item
  html = html.replace(/^[\-\*] (.+)$/gm, "<li>$1</li>");
  html = html.replace(/(<li>.*<\/li>\n?)+/g, "<ul>$&</ul>");

  // Numbered lists: 1. item
  html = html.replace(/^\d+\. (.+)$/gm, "<li>$1</li>");

  // Horizontal rule: ---
  html = html.replace(/^---$/gm, "<hr>");

  // Line breaks: double newline = paragraph
  html = html.replace(/\n\n/g, "</p><p>");
  html = html.replace(/\n/g, "<br>");
  html = "<p>" + html + "</p>";

  // Clean up empty paragraphs
  html = html.replace(/<p>\s*<\/p>/g, "");

  // Citation markers: [1], [2], etc.
  html = html.replace(
    /\[(\d+)\]/g,
    '<span class="citation-id" title="Source $1">[$1]</span>',
  );

  return html;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// ── Input Handling ─────────────────────────────────────────────────────────

function initTextarea() {
  const input = document.getElementById("questionInput");

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  input.addEventListener("input", () => {
    updateCharCount();
    autoResizeTextarea(input);
  });
}

function autoResizeTextarea(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 150) + "px";
}

function updateCharCount() {
  const input = document.getElementById("questionInput");
  const count = document.getElementById("charCount");
  count.textContent = `${input.value.length}/2000`;
}

function updateSendButton() {
  const btn = document.getElementById("sendBtn");
  btn.disabled = isStreaming;
}

// ── Navigation ─────────────────────────────────────────────────────────────

function initNavigation() {
  const navBtns = document.querySelectorAll(".nav-btn");
  navBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const view = btn.dataset.view;
      switchView(view);

      navBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
    });
  });
}

function switchView(viewName) {
  const views = document.querySelectorAll(".view");
  views.forEach((v) => v.classList.remove("active"));

  const target = document.getElementById(viewName + "View");
  if (target) target.classList.add("active");
}

// ── Example Questions ──────────────────────────────────────────────────────

function askExample(element) {
  const question = element.textContent.trim();
  const input = document.getElementById("questionInput");
  input.value = question;
  updateCharCount();

  // Switch to chat view
  switchView("chat");
  document
    .querySelectorAll(".nav-btn")
    .forEach((b) => b.classList.remove("active"));
  document.querySelector('.nav-btn[data-view="chat"]').classList.add("active");

  sendMessage();
}

// ── Chat Management ────────────────────────────────────────────────────────

function resetChat() {
  const container = document.getElementById("chatContainer");
  container.innerHTML = "";

  // Show welcome message again
  container.innerHTML = `
        <div class="welcome-message" id="welcomeMessage">
            <div class="welcome-icon">&#9764;</div>
            <h2>Welcome to PharmAssistAI</h2>
            <p>Ask me about pharmaceutical drugs, interactions, dosing guidelines,
            adverse effects, and regulatory information.</p>
            <div class="welcome-chips">
                <button class="chip" onclick="askExample(this)">Aspirin side effects</button>
                <button class="chip" onclick="askExample(this)">Metformin dosing</button>
                <button class="chip" onclick="askExample(this)">Drug interactions</button>
            </div>
        </div>
    `;

  sessionId = null;
  currentAssistantMessage = null;
  currentAssistantText = "";
  isStreaming = false;

  // Reconnect WebSocket for new session
  if (ws) ws.close();
  initWebSocket();
}

// ── Disclaimer ─────────────────────────────────────────────────────────────

function dismissDisclaimer() {
  const banner = document.getElementById("disclaimerBanner");
  banner.classList.add("dismissed");
  document.querySelector(".app-container").classList.add("no-disclaimer");
}

// ── File Upload ────────────────────────────────────────────────────────────

function initUpload() {
  const zone = document.getElementById("uploadZone");
  const fileInput = document.getElementById("fileInput");

  zone.addEventListener("click", () => fileInput.click());

  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("drag-over");
  });

  zone.addEventListener("dragleave", () => {
    zone.classList.remove("drag-over");
  });

  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) {
      uploadFiles(e.dataTransfer.files);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      uploadFiles(fileInput.files);
    }
  });
}

async function uploadFiles(files) {
  const collection = document.getElementById("collectionSelect").value;
  const chunkStrategy = document.getElementById("chunkStrategySelect").value;
  const resultsDiv = document.getElementById("uploadResults");
  const resultContent = document.getElementById("uploadResultContent");

  resultContent.innerHTML = "<p>Uploading and processing...</p>";
  resultsDiv.hidden = false;

  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }

  try {
    const response = await fetch(
      `${API_BASE}/documents/ingest?collection=${collection}&chunk_strategy=${chunkStrategy}`,
      { method: "POST", body: formData },
    );

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.status}`);
    }

    const data = await response.json();

    resultContent.innerHTML = `
            <p><strong>Files processed:</strong> ${data.files_processed}</p>
            <p><strong>Chunks created:</strong> ${data.chunks_created}</p>
            <p><strong>Collection:</strong> ${data.collection}</p>
            ${
              data.errors.length > 0
                ? `<p style="color: var(--accent-red);"><strong>Errors:</strong><br>${data.errors.join("<br>")}</p>`
                : '<p style="color: var(--accent-green);">All files processed successfully.</p>'
            }
        `;
  } catch (error) {
    resultContent.innerHTML = `<p style="color: var(--accent-red);">Error: ${error.message}</p>`;
  }
}

// ── Document Search ────────────────────────────────────────────────────────

function initSearchInput() {
  const input = document.getElementById("searchInput");
  if (input) {
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        searchDocuments();
      }
    });
  }
}

async function searchDocuments() {
  const query = document.getElementById("searchInput").value.trim();
  if (!query) return;

  const resultsDiv = document.getElementById("searchResults");
  resultsDiv.innerHTML =
    '<p style="color: var(--text-muted);">Searching...</p>';

  try {
    const params = new URLSearchParams({ query, n_results: 10 });
    const response = await fetch(`${API_BASE}/documents/search?${params}`);

    if (!response.ok) {
      throw new Error(`Search failed: ${response.status}`);
    }

    const data = await response.json();

    if (data.results.length === 0) {
      resultsDiv.innerHTML =
        '<p style="color: var(--text-muted);">No results found.</p>';
      return;
    }

    resultsDiv.innerHTML = data.results
      .map(
        (result) => `
            <div class="search-result-item">
                <div class="search-result-header">
                    <span class="search-result-drug">
                        ${result.metadata.drug_name || "Unknown Drug"}
                    </span>
                    <span class="search-result-score">
                        ${(result.relevance_score * 100).toFixed(0)}% relevant
                    </span>
                </div>
                <div class="search-result-text">${escapeHtml(result.text).substring(0, 300)}...</div>
                <div class="search-result-meta">
                    ${result.metadata.section_type || ""} |
                    ${result.metadata.document_type || ""} |
                    ${result.collection}
                </div>
            </div>
        `,
      )
      .join("");
  } catch (error) {
    resultsDiv.innerHTML = `<p style="color: var(--accent-red);">Error: ${error.message}</p>`;
  }
}
