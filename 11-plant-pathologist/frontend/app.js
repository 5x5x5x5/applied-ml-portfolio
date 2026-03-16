/**
 * PlantPathologist Frontend Application
 *
 * Handles image upload/capture, API communication, and result rendering
 * for the plant disease detection web interface.
 */

const API_BASE = window.location.origin;

// ---------------------------------------------------------------------------
// DOM References
// ---------------------------------------------------------------------------

const uploadArea = document.getElementById("uploadArea");
const uploadPlaceholder = document.getElementById("uploadPlaceholder");
const previewImage = document.getElementById("previewImage");
const fileInput = document.getElementById("fileInput");
const cameraBtn = document.getElementById("cameraBtn");
const cameraModal = document.getElementById("cameraModal");
const cameraVideo = document.getElementById("cameraVideo");
const cameraCanvas = document.getElementById("cameraCanvas");
const captureBtn = document.getElementById("captureBtn");
const closeCameraBtn = document.getElementById("closeCameraBtn");
const diagnoseBtn = document.getElementById("diagnoseBtn");

const uploadSection = document.getElementById("uploadSection");
const loadingSection = document.getElementById("loadingSection");
const resultsSection = document.getElementById("resultsSection");
const newDiagnosisBtn = document.getElementById("newDiagnosisBtn");

// State
let selectedFile = null;
let cameraStream = null;

// ---------------------------------------------------------------------------
// Disease Library Data (rendered client-side)
// ---------------------------------------------------------------------------

const speciesData = {
  Tomato: [
    "Healthy",
    "Early Blight",
    "Late Blight",
    "Bacterial Spot",
    "Septoria Leaf Spot",
    "Leaf Mold",
    "Yellow Leaf Curl Virus",
    "Target Spot",
  ],
  Potato: ["Healthy", "Early Blight", "Late Blight"],
  Corn: ["Healthy", "Northern Leaf Blight", "Common Rust", "Gray Leaf Spot"],
  Apple: ["Healthy", "Scab", "Black Rot", "Cedar Apple Rust"],
  Grape: ["Healthy", "Black Rot", "Esca", "Leaf Blight"],
};

function renderSpeciesGrid() {
  const grid = document.getElementById("speciesGrid");
  if (!grid) return;

  grid.innerHTML = "";
  for (const [species, diseases] of Object.entries(speciesData)) {
    const card = document.createElement("div");
    card.className = "species-card";

    const title = document.createElement("h3");
    title.textContent = species;
    card.appendChild(title);

    const tagsDiv = document.createElement("div");
    diseases.forEach((disease) => {
      const tag = document.createElement("span");
      tag.className =
        disease === "Healthy" ? "disease-tag healthy-tag" : "disease-tag";
      tag.textContent = disease;
      tagsDiv.appendChild(tag);
    });
    card.appendChild(tagsDiv);
    grid.appendChild(card);
  }
}

// ---------------------------------------------------------------------------
// File Upload & Drag-and-Drop
// ---------------------------------------------------------------------------

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) handleFileSelected(file);
});

uploadArea.addEventListener("click", () => {
  fileInput.click();
});

uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.classList.add("drag-over");
});

uploadArea.addEventListener("dragleave", () => {
  uploadArea.classList.remove("drag-over");
});

uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) {
    handleFileSelected(file);
  }
});

function handleFileSelected(file) {
  selectedFile = file;

  // Show preview
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    previewImage.classList.remove("hidden");
    uploadPlaceholder.classList.add("hidden");
    uploadArea.classList.add("has-image");
  };
  reader.readAsDataURL(file);

  diagnoseBtn.disabled = false;
}

// ---------------------------------------------------------------------------
// Camera Capture
// ---------------------------------------------------------------------------

cameraBtn.addEventListener("click", async () => {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "environment", // Prefer rear camera on mobile
        width: { ideal: 1280 },
        height: { ideal: 960 },
      },
    });
    cameraVideo.srcObject = cameraStream;
    cameraModal.classList.remove("hidden");
  } catch (err) {
    console.error("Camera access denied:", err);
    alert(
      "Could not access camera. Please check permissions or use the upload option.",
    );
  }
});

captureBtn.addEventListener("click", () => {
  const video = cameraVideo;
  const canvas = cameraCanvas;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0);

  // Convert canvas to blob
  canvas.toBlob(
    (blob) => {
      if (!blob) return;
      selectedFile = new File([blob], "capture.jpg", { type: "image/jpeg" });

      // Show preview
      previewImage.src = canvas.toDataURL("image/jpeg");
      previewImage.classList.remove("hidden");
      uploadPlaceholder.classList.add("hidden");
      uploadArea.classList.add("has-image");
      diagnoseBtn.disabled = false;

      closeCamera();
    },
    "image/jpeg",
    0.92,
  );
});

closeCameraBtn.addEventListener("click", closeCamera);

function closeCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  cameraModal.classList.add("hidden");
}

// ---------------------------------------------------------------------------
// Diagnosis API Call
// ---------------------------------------------------------------------------

diagnoseBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  // Show loading
  uploadSection.classList.add("hidden");
  resultsSection.classList.add("hidden");
  loadingSection.classList.remove("hidden");

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch(`${API_BASE}/diagnose`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `Server error: ${response.status}`);
    }

    const result = await response.json();
    renderResults(result);

    loadingSection.classList.add("hidden");
    resultsSection.classList.remove("hidden");
  } catch (err) {
    console.error("Diagnosis failed:", err);
    loadingSection.classList.add("hidden");
    uploadSection.classList.remove("hidden");
    alert(`Diagnosis failed: ${err.message}`);
  }
});

// ---------------------------------------------------------------------------
// Render Results
// ---------------------------------------------------------------------------

function renderResults(data) {
  // Primary result card
  const primaryCard = document.getElementById("primaryResult");
  const statusEl = document.getElementById("resultStatus");
  const confEl = document.getElementById("resultConfidence");
  const nameEl = document.getElementById("resultDiseaseName");
  const speciesEl = document.getElementById("resultSpecies");
  const descEl = document.getElementById("resultDescription");
  const severityEl = document.getElementById("severityBadge");

  // Status badge
  if (data.is_healthy) {
    statusEl.textContent = "Healthy";
    statusEl.className = "result-status healthy";
    primaryCard.className = "result-card primary-result";
  } else {
    statusEl.textContent = "Disease Detected";
    statusEl.className = "result-status diseased";
    primaryCard.className = "result-card primary-result diseased";
  }

  // Confidence
  const confPct = (data.confidence * 100).toFixed(1);
  confEl.textContent = `${confPct}%`;
  if (data.confidence >= 0.8) confEl.className = "result-confidence high";
  else if (data.confidence >= 0.5)
    confEl.className = "result-confidence medium";
  else confEl.className = "result-confidence low";

  // Disease name and details
  nameEl.textContent = data.disease_name;
  speciesEl.textContent = `Plant: ${capitalize(data.plant_species)} (${(data.species_confidence * 100).toFixed(1)}% confidence)`;
  descEl.textContent = data.description;

  // Severity
  severityEl.textContent = `Severity: ${data.severity.toUpperCase()}`;
  severityEl.className = `severity-badge ${data.severity}`;

  // Symptoms
  const symptomsList = document.getElementById("symptomsList");
  symptomsList.innerHTML = "";
  data.symptoms.forEach((s) => {
    const li = document.createElement("li");
    li.textContent = s;
    symptomsList.appendChild(li);
  });

  // Treatment tabs
  const treatmentSection = document.getElementById("treatmentSection");
  if (data.is_healthy) {
    treatmentSection.classList.add("hidden");
  } else {
    treatmentSection.classList.remove("hidden");
    renderList("organicList", data.treatment_organic);
    renderList("chemicalList", data.treatment_chemical);
    renderList("culturalList", data.treatment_cultural);
    renderList("preventionList", data.prevention);
  }

  // Top predictions
  const predList = document.getElementById("predictionsList");
  predList.innerHTML = "";
  data.top_predictions.forEach((pred) => {
    const item = document.createElement("div");
    item.className = "prediction-item";

    const name = document.createElement("span");
    name.className = "prediction-name";
    name.textContent = pred.disease_name;

    const barContainer = document.createElement("div");
    barContainer.className = "prediction-bar-container";

    const bar = document.createElement("div");
    bar.className = "prediction-bar";
    bar.style.width = `${(pred.confidence * 100).toFixed(1)}%`;
    barContainer.appendChild(bar);

    const value = document.createElement("span");
    value.className = "prediction-value";
    value.textContent = `${(pred.confidence * 100).toFixed(1)}%`;

    item.appendChild(name);
    item.appendChild(barContainer);
    item.appendChild(value);
    predList.appendChild(item);
  });

  // Quality details
  const qualityDetails = document.getElementById("qualityDetails");
  let qualityHtml = `<p><strong>Valid:</strong> ${data.image_quality_valid ? "Yes" : "No"}</p>`;
  qualityHtml += `<p><strong>Inference time:</strong> ${data.inference_time_ms.toFixed(1)} ms</p>`;
  qualityHtml += `<p><strong>Calibrated:</strong> ${data.calibrated ? "Yes" : "No"}</p>`;
  if (data.image_quality_issues.length > 0) {
    qualityHtml += "<p><strong>Issues:</strong></p><ul>";
    data.image_quality_issues.forEach((issue) => {
      qualityHtml += `<li>${issue}</li>`;
    });
    qualityHtml += "</ul>";
  }
  qualityDetails.innerHTML = qualityHtml;
}

function renderList(elementId, items) {
  const el = document.getElementById(elementId);
  el.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    el.appendChild(li);
  });
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const tabId = btn.dataset.tab;

    // Deactivate all
    document
      .querySelectorAll(".tab-btn")
      .forEach((b) => b.classList.remove("active"));
    document
      .querySelectorAll(".tab-pane")
      .forEach((p) => p.classList.remove("active"));

    // Activate selected
    btn.classList.add("active");
    document.getElementById(`tab-${tabId}`).classList.add("active");
  });
});

// ---------------------------------------------------------------------------
// Collapsible sections
// ---------------------------------------------------------------------------

document.querySelectorAll(".collapsible-header").forEach((header) => {
  header.addEventListener("click", () => {
    const card = header.closest(".collapsible");
    const content = card.querySelector(".collapsible-content");
    card.classList.toggle("open");
    content.classList.toggle("hidden");
  });
});

// ---------------------------------------------------------------------------
// New Diagnosis
// ---------------------------------------------------------------------------

newDiagnosisBtn.addEventListener("click", () => {
  // Reset state
  selectedFile = null;
  previewImage.src = "";
  previewImage.classList.add("hidden");
  uploadPlaceholder.classList.remove("hidden");
  uploadArea.classList.remove("has-image");
  diagnoseBtn.disabled = true;
  fileInput.value = "";

  // Show upload, hide results
  resultsSection.classList.add("hidden");
  uploadSection.classList.remove("hidden");
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function capitalize(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

// ---------------------------------------------------------------------------
// Initialize
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  renderSpeciesGrid();
});
