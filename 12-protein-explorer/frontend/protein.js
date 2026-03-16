/**
 * ProteinExplorer - Frontend JavaScript
 *
 * Handles sequence parsing, validation, interactive SVG/Canvas charts,
 * color-coded sequence display, and API integration.
 */

const API_BASE = window.location.origin;

// Amino acid classification for coloring
const AA_CLASSES = {
  hydrophobic: new Set(["A", "V", "I", "L", "M", "F", "W"]),
  polar: new Set(["S", "T", "N", "Q"]),
  positive: new Set(["K", "R", "H"]),
  negative: new Set(["D", "E"]),
  aromatic: new Set(["F", "W", "Y"]),
  special: new Set(["C", "G", "P"]),
};

// Amino acid full names
const AA_NAMES = {
  A: "Alanine",
  R: "Arginine",
  N: "Asparagine",
  D: "Aspartate",
  C: "Cysteine",
  Q: "Glutamine",
  E: "Glutamate",
  G: "Glycine",
  H: "Histidine",
  I: "Isoleucine",
  L: "Leucine",
  K: "Lysine",
  M: "Methionine",
  F: "Phenylalanine",
  P: "Proline",
  S: "Serine",
  T: "Threonine",
  W: "Tryptophan",
  Y: "Tyrosine",
  V: "Valine",
};

// Colors for pie chart
const AA_COLORS = {
  A: "#e74c3c",
  R: "#3498db",
  N: "#2ecc71",
  D: "#e67e22",
  C: "#f1c40f",
  Q: "#1abc9c",
  E: "#9b59b6",
  G: "#95a5a6",
  H: "#34495e",
  I: "#e74c3c",
  L: "#c0392b",
  K: "#2980b9",
  M: "#27ae60",
  F: "#d35400",
  P: "#16a085",
  S: "#8e44ad",
  T: "#2c3e50",
  W: "#f39c12",
  Y: "#7f8c8d",
  V: "#c0392b",
};

const VALID_AAS = new Set("ACDEFGHIKLMNPQRSTVWY");

// ---- Sequence Parsing and Validation ----

function parseFasta(text) {
  const lines = text.trim().split("\n");
  let header = "";
  const seqLines = [];

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith(">")) {
      header = trimmed.substring(1).trim();
    } else if (trimmed.length > 0) {
      seqLines.push(trimmed.toUpperCase().replace(/\s/g, ""));
    }
  }

  return { header, sequence: seqLines.join("") };
}

function validateSequence(sequence) {
  const cleaned = sequence.toUpperCase().replace(/\s/g, "").replace(/\n/g, "");
  const invalid = [];

  for (const ch of cleaned) {
    if (!VALID_AAS.has(ch)) {
      invalid.push(ch);
    }
  }

  return {
    valid: invalid.length === 0 && cleaned.length > 0,
    cleaned,
    invalidChars: [...new Set(invalid)],
  };
}

function getResidueClass(aa) {
  if (aa === "C") return "cysteine";
  if (aa === "G") return "glycine";
  if (aa === "P") return "proline";
  if (AA_CLASSES.negative.has(aa)) return "negative";
  if (AA_CLASSES.positive.has(aa)) return "positive";
  if (AA_CLASSES.aromatic.has(aa)) return "aromatic";
  if (AA_CLASSES.polar.has(aa)) return "polar";
  if (AA_CLASSES.hydrophobic.has(aa)) return "hydrophobic";
  return "";
}

// ---- Color-coded Sequence Display ----

function displayColoredSequence(sequence, container) {
  container.innerHTML = "";
  const residuesPerLine = 60;

  for (let i = 0; i < sequence.length; i += residuesPerLine) {
    const lineDiv = document.createElement("div");
    lineDiv.className = "sequence-line";

    // Line number
    const numSpan = document.createElement("span");
    numSpan.className = "residue-number";
    numSpan.textContent = (i + 1).toString();
    lineDiv.appendChild(numSpan);

    // Residues
    const chunk = sequence.substring(i, i + residuesPerLine);
    for (let j = 0; j < chunk.length; j++) {
      const span = document.createElement("span");
      span.className = `residue ${getResidueClass(chunk[j])}`;
      span.textContent = chunk[j];
      span.title = `${AA_NAMES[chunk[j]] || chunk[j]} (${i + j + 1})`;

      // Add space every 10 residues for readability
      if (j > 0 && j % 10 === 0) {
        lineDiv.appendChild(document.createTextNode(" "));
      }

      lineDiv.appendChild(span);
    }

    container.appendChild(lineDiv);
  }
}

// ---- SVG Hydrophobicity Chart ----

function drawHydrophobicityChart(profile, svgEl) {
  const svg = svgEl;
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  const width = 800;
  const height = 300;
  const margin = { top: 20, right: 30, bottom: 40, left: 55 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;

  const n = profile.length;
  const minVal = Math.min(...profile, -2);
  const maxVal = Math.max(...profile, 2);
  const range = maxVal - minVal || 1;

  const xScale = (i) => margin.left + (i / (n - 1)) * plotW;
  const yScale = (v) => margin.top + plotH - ((v - minVal) / range) * plotH;

  const ns = "http://www.w3.org/2000/svg";

  // Zero line
  const zeroY = yScale(0);
  const zeroLine = document.createElementNS(ns, "line");
  zeroLine.setAttribute("x1", margin.left);
  zeroLine.setAttribute("x2", margin.left + plotW);
  zeroLine.setAttribute("y1", zeroY);
  zeroLine.setAttribute("y2", zeroY);
  zeroLine.setAttribute("class", "hydro-zero");
  svg.appendChild(zeroLine);

  // Area fills for positive (hydrophobic) and negative (hydrophilic)
  let posPath = `M ${xScale(0)} ${zeroY}`;
  let negPath = `M ${xScale(0)} ${zeroY}`;

  for (let i = 0; i < n; i++) {
    const x = xScale(i);
    const y = yScale(profile[i]);
    if (profile[i] >= 0) {
      posPath += ` L ${x} ${y}`;
    } else {
      posPath += ` L ${x} ${zeroY}`;
    }
    if (profile[i] <= 0) {
      negPath += ` L ${x} ${y}`;
    } else {
      negPath += ` L ${x} ${zeroY}`;
    }
  }
  posPath += ` L ${xScale(n - 1)} ${zeroY} Z`;
  negPath += ` L ${xScale(n - 1)} ${zeroY} Z`;

  const posArea = document.createElementNS(ns, "path");
  posArea.setAttribute("d", posPath);
  posArea.setAttribute("class", "hydro-area-positive");
  svg.appendChild(posArea);

  const negArea = document.createElementNS(ns, "path");
  negArea.setAttribute("d", negPath);
  negArea.setAttribute("class", "hydro-area-negative");
  svg.appendChild(negArea);

  // Line
  let linePath = `M ${xScale(0)} ${yScale(profile[0])}`;
  for (let i = 1; i < n; i++) {
    linePath += ` L ${xScale(i)} ${yScale(profile[i])}`;
  }

  const line = document.createElementNS(ns, "path");
  line.setAttribute("d", linePath);
  line.setAttribute("class", "hydro-line");
  line.setAttribute("stroke", "#f39c12");
  svg.appendChild(line);

  // Axes
  const xAxis = document.createElementNS(ns, "line");
  xAxis.setAttribute("x1", margin.left);
  xAxis.setAttribute("x2", margin.left + plotW);
  xAxis.setAttribute("y1", margin.top + plotH);
  xAxis.setAttribute("y2", margin.top + plotH);
  xAxis.setAttribute("class", "hydro-axis");
  svg.appendChild(xAxis);

  const yAxis = document.createElementNS(ns, "line");
  yAxis.setAttribute("x1", margin.left);
  yAxis.setAttribute("x2", margin.left);
  yAxis.setAttribute("y1", margin.top);
  yAxis.setAttribute("y2", margin.top + plotH);
  yAxis.setAttribute("class", "hydro-axis");
  svg.appendChild(yAxis);

  // X-axis labels
  const xTicks = Math.min(10, n);
  for (let i = 0; i <= xTicks; i++) {
    const idx = Math.round((i / xTicks) * (n - 1));
    const x = xScale(idx);
    const label = document.createElementNS(ns, "text");
    label.setAttribute("x", x);
    label.setAttribute("y", margin.top + plotH + 18);
    label.setAttribute("class", "hydro-tick-label");
    label.setAttribute("text-anchor", "middle");
    label.textContent = (idx + 1).toString();
    svg.appendChild(label);
  }

  // Y-axis labels
  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {
    const val = minVal + (i / yTicks) * range;
    const y = yScale(val);
    const label = document.createElementNS(ns, "text");
    label.setAttribute("x", margin.left - 8);
    label.setAttribute("y", y + 4);
    label.setAttribute("class", "hydro-tick-label");
    label.setAttribute("text-anchor", "end");
    label.textContent = val.toFixed(1);
    svg.appendChild(label);

    const tick = document.createElementNS(ns, "line");
    tick.setAttribute("x1", margin.left - 4);
    tick.setAttribute("x2", margin.left);
    tick.setAttribute("y1", y);
    tick.setAttribute("y2", y);
    tick.setAttribute("class", "hydro-axis");
    svg.appendChild(tick);
  }

  // Axis labels
  const xLabel = document.createElementNS(ns, "text");
  xLabel.setAttribute("x", margin.left + plotW / 2);
  xLabel.setAttribute("y", height - 2);
  xLabel.setAttribute("class", "hydro-label");
  xLabel.setAttribute("text-anchor", "middle");
  xLabel.textContent = "Residue Position";
  svg.appendChild(xLabel);

  const yLabel = document.createElementNS(ns, "text");
  yLabel.setAttribute("x", 12);
  yLabel.setAttribute("y", margin.top + plotH / 2);
  yLabel.setAttribute("class", "hydro-label");
  yLabel.setAttribute("text-anchor", "middle");
  yLabel.setAttribute("transform", `rotate(-90 12 ${margin.top + plotH / 2})`);
  yLabel.textContent = "Hydrophobicity";
  svg.appendChild(yLabel);

  // Annotation labels
  const hydrophobicLabel = document.createElementNS(ns, "text");
  hydrophobicLabel.setAttribute("x", margin.left + plotW - 5);
  hydrophobicLabel.setAttribute("y", margin.top + 15);
  hydrophobicLabel.setAttribute("class", "hydro-label");
  hydrophobicLabel.setAttribute("text-anchor", "end");
  hydrophobicLabel.setAttribute("fill", "#f39c12");
  hydrophobicLabel.textContent = "Hydrophobic";
  svg.appendChild(hydrophobicLabel);

  const hydrophilicLabel = document.createElementNS(ns, "text");
  hydrophilicLabel.setAttribute("x", margin.left + plotW - 5);
  hydrophilicLabel.setAttribute("y", margin.top + plotH - 5);
  hydrophilicLabel.setAttribute("class", "hydro-label");
  hydrophilicLabel.setAttribute("text-anchor", "end");
  hydrophilicLabel.setAttribute("fill", "#3498db");
  hydrophilicLabel.textContent = "Hydrophilic";
  svg.appendChild(hydrophilicLabel);
}

// ---- Pie Chart for Amino Acid Composition ----

function drawCompositionChart(composition, svgEl) {
  const svg = svgEl;
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  const ns = "http://www.w3.org/2000/svg";
  const cx = 200,
    cy = 185,
    r = 130;

  // Sort by percentage descending
  const entries = Object.entries(composition).sort((a, b) => b[1] - a[1]);

  // Group small slices (< 2%) into "Other"
  const mainEntries = [];
  let otherPct = 0;

  for (const [aa, pct] of entries) {
    if (pct >= 2.0) {
      mainEntries.push([aa, pct]);
    } else {
      otherPct += pct;
    }
  }

  if (otherPct > 0) {
    mainEntries.push(["Other", otherPct]);
  }

  const total = mainEntries.reduce((sum, [, pct]) => sum + pct, 0);

  let currentAngle = -Math.PI / 2; // Start from top

  mainEntries.forEach(([aa, pct], idx) => {
    const sliceAngle = (pct / total) * 2 * Math.PI;
    const startAngle = currentAngle;
    const endAngle = currentAngle + sliceAngle;

    const x1 = cx + r * Math.cos(startAngle);
    const y1 = cy + r * Math.sin(startAngle);
    const x2 = cx + r * Math.cos(endAngle);
    const y2 = cy + r * Math.sin(endAngle);

    const largeArc = sliceAngle > Math.PI ? 1 : 0;

    const path = document.createElementNS(ns, "path");
    path.setAttribute(
      "d",
      `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2} Z`,
    );
    path.setAttribute(
      "fill",
      aa === "Other" ? "#555" : AA_COLORS[aa] || "#666",
    );
    path.setAttribute("stroke", "#1e2d3d");
    path.setAttribute("stroke-width", "2");
    path.style.cursor = "pointer";
    path.setAttribute("opacity", "0.85");

    // Hover effect
    path.addEventListener("mouseenter", () => {
      path.setAttribute("opacity", "1");
      path.setAttribute(
        "transform",
        `translate(${5 * Math.cos(startAngle + sliceAngle / 2)} ${5 * Math.sin(startAngle + sliceAngle / 2)})`,
      );
    });
    path.addEventListener("mouseleave", () => {
      path.setAttribute("opacity", "0.85");
      path.setAttribute("transform", "");
    });

    svg.appendChild(path);

    // Label
    if (sliceAngle > 0.15) {
      // Only label slices big enough
      const labelR = r * 0.7;
      const midAngle = startAngle + sliceAngle / 2;
      const lx = cx + labelR * Math.cos(midAngle);
      const ly = cy + labelR * Math.sin(midAngle);

      const label = document.createElementNS(ns, "text");
      label.setAttribute("x", lx);
      label.setAttribute("y", ly);
      label.setAttribute("class", "pie-label");
      label.setAttribute("text-anchor", "middle");
      label.setAttribute("dominant-baseline", "middle");
      label.textContent = aa;
      svg.appendChild(label);

      if (sliceAngle > 0.3) {
        const valLabel = document.createElementNS(ns, "text");
        valLabel.setAttribute("x", lx);
        valLabel.setAttribute("y", ly + 13);
        valLabel.setAttribute("class", "pie-value");
        valLabel.setAttribute("text-anchor", "middle");
        valLabel.setAttribute("dominant-baseline", "middle");
        valLabel.textContent = `${pct.toFixed(1)}%`;
        svg.appendChild(valLabel);
      }
    }

    currentAngle = endAngle;
  });

  // Title
  const title = document.createElementNS(ns, "text");
  title.setAttribute("x", cx);
  title.setAttribute("y", 30);
  title.setAttribute("class", "hydro-label");
  title.setAttribute("text-anchor", "middle");
  title.setAttribute("font-size", "14");
  title.textContent = "Amino Acid Distribution";
  svg.appendChild(title);
}

// ---- Secondary Structure Visualization ----

function displaySecondaryStructure(prediction, summary, container) {
  container.innerHTML = "";

  prediction.forEach((ss, i) => {
    const div = document.createElement("div");
    div.className = `ss-residue ${ss === "H" ? "helix" : ss === "E" ? "sheet" : "coil"}`;
    div.title = `Position ${i + 1}: ${ss === "H" ? "Helix" : ss === "E" ? "Sheet" : "Coil"}`;
    container.appendChild(div);
  });

  // Update summary bars
  const helixPct = (summary.helix * 100).toFixed(1);
  const sheetPct = (summary.sheet * 100).toFixed(1);
  const coilPct = (summary.coil * 100).toFixed(1);

  document.getElementById("ss-bar-helix").style.width = `${helixPct}%`;
  document.getElementById("ss-bar-sheet").style.width = `${sheetPct}%`;
  document.getElementById("ss-bar-coil").style.width = `${coilPct}%`;

  document.getElementById("ss-helix-pct").textContent = `Helix: ${helixPct}%`;
  document.getElementById("ss-sheet-pct").textContent = `Sheet: ${sheetPct}%`;
  document.getElementById("ss-coil-pct").textContent = `Coil: ${coilPct}%`;
}

// ---- Alignment Viewer ----

function displayAlignment(data) {
  const statsDiv = document.getElementById("alignment-stats");
  const viewerDiv = document.getElementById("alignment-viewer");

  // Stats
  statsDiv.innerHTML = `
        <div class="stat"><span class="stat-value">${data.score.toFixed(1)}</span><span class="stat-label">Score</span></div>
        <div class="stat"><span class="stat-value">${(data.identity * 100).toFixed(1)}%</span><span class="stat-label">Identity</span></div>
        <div class="stat"><span class="stat-value">${(data.similarity * 100).toFixed(1)}%</span><span class="stat-label">Similarity</span></div>
        <div class="stat"><span class="stat-value">${data.gaps}</span><span class="stat-label">Gaps</span></div>
        <div class="stat"><span class="stat-value">${data.gap_opens}</span><span class="stat-label">Gap Opens</span></div>
        <div class="stat"><span class="stat-value">${data.alignment_length}</span><span class="stat-label">Length</span></div>
        <div class="stat"><span class="stat-value">${data.gap_open_used}</span><span class="stat-label">Gap Open</span></div>
        <div class="stat"><span class="stat-value">${data.gap_extend_used}</span><span class="stat-label">Gap Extend</span></div>
    `;

  // Color-coded alignment
  const lineWidth = 60;
  let html = "";
  let pos1 = 0,
    pos2 = 0;

  for (let start = 0; start < data.alignment_length; start += lineWidth) {
    const end = Math.min(start + lineWidth, data.alignment_length);
    const chunk1 = data.aligned_seq1.substring(start, end);
    const chunk2 = data.aligned_seq2.substring(start, end);
    const midline = data.midline.substring(start, end);

    const p1Start = pos1 + 1;
    const p2Start = pos2 + 1;

    // Seq1 line
    let seq1Html = `<span style="color:#95a5a6">Seq1 ${String(p1Start).padStart(5)}</span>  `;
    for (let j = 0; j < chunk1.length; j++) {
      const ch = chunk1[j];
      if (ch === "-") {
        seq1Html += `<span class="aln-gap">${ch}</span>`;
      } else {
        pos1++;
        const cls =
          midline[j] === "|"
            ? "aln-match"
            : midline[j] === "+"
              ? "aln-similar"
              : "aln-mismatch";
        seq1Html += `<span class="${cls}">${ch}</span>`;
      }
    }
    seq1Html += `  ${pos1}\n`;

    // Midline
    let midHtml = "              ";
    for (const ch of midline) {
      if (ch === "|") midHtml += `<span class="aln-match">${ch}</span>`;
      else if (ch === "+") midHtml += `<span class="aln-similar">${ch}</span>`;
      else midHtml += " ";
    }
    midHtml += "\n";

    // Seq2 line
    let seq2Html = `<span style="color:#95a5a6">Seq2 ${String(p2Start).padStart(5)}</span>  `;
    for (let j = 0; j < chunk2.length; j++) {
      const ch = chunk2[j];
      if (ch === "-") {
        seq2Html += `<span class="aln-gap">${ch}</span>`;
      } else {
        pos2++;
        const cls =
          midline[j] === "|"
            ? "aln-match"
            : midline[j] === "+"
              ? "aln-similar"
              : "aln-mismatch";
        seq2Html += `<span class="${cls}">${ch}</span>`;
      }
    }
    seq2Html += `  ${pos2}\n\n`;

    html += seq1Html + midHtml + seq2Html;
  }

  viewerDiv.innerHTML = html;
}

// ---- API Integration ----

async function apiCall(endpoint, method, body) {
  const options = {
    method,
    headers: { "Content-Type": "application/json" },
  };
  if (body) {
    options.body = JSON.stringify(body);
  }

  const response = await fetch(`${API_BASE}${endpoint}`, options);

  if (!response.ok) {
    const error = await response
      .json()
      .catch(() => ({ detail: "Request failed" }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

function showLoading(show) {
  document.getElementById("loading-overlay").classList.toggle("hidden", !show);
}

function showPanel(id) {
  document.getElementById(id).classList.remove("hidden");
}

// ---- Event Handlers ----

async function analyzeSequence() {
  const input = document.getElementById("sequence-input").value.trim();
  if (!input) return;

  // Parse FASTA if needed
  let sequence = input;
  if (input.startsWith(">")) {
    const parsed = parseFasta(input);
    sequence = parsed.sequence;
  }

  // Validate
  const validation = validateSequence(sequence);
  if (!validation.valid) {
    alert(
      `Invalid amino acid characters: ${validation.invalidChars.join(", ")}`,
    );
    return;
  }

  showLoading(true);

  try {
    const data = await apiCall("/analyze", "POST", {
      sequence: validation.cleaned,
    });

    // Display colored sequence
    const seqDisplay = document.getElementById("sequence-display");
    displayColoredSequence(data.sequence, seqDisplay);
    seqDisplay.classList.remove("hidden");

    // Properties
    document.getElementById("prop-length").textContent = `${data.length} aa`;
    document.getElementById("prop-mw").textContent =
      `${(data.molecular_weight / 1000).toFixed(2)} kDa`;
    document.getElementById("prop-pi").textContent =
      data.isoelectric_point.toFixed(2);

    const chargeEl = document.getElementById("prop-charge");
    chargeEl.textContent =
      data.charge_at_ph7 > 0
        ? `+${data.charge_at_ph7.toFixed(1)}`
        : data.charge_at_ph7.toFixed(1);
    chargeEl.className = `property-value ${data.charge_at_ph7 > 0 ? "positive" : "negative"}`;

    document.getElementById("prop-gravy").textContent = data.gravy.toFixed(3);
    document.getElementById("prop-aromaticity").textContent =
      `${(data.aromaticity * 100).toFixed(1)}%`;

    const instEl = document.getElementById("prop-instability");
    instEl.textContent = `${data.instability_index.toFixed(1)} (${data.is_stable ? "Stable" : "Unstable"})`;
    instEl.className = `property-value ${data.is_stable ? "positive" : "warning"}`;

    document.getElementById("prop-signal").textContent = data.has_signal_peptide
      ? `Yes (${data.signal_peptide_length} aa)`
      : "Not detected";

    document.getElementById("prop-disulfide").textContent =
      data.disulfide_bonds.length > 0
        ? data.disulfide_bonds.map((b) => `C${b[0]}-C${b[1]}`).join(", ")
        : "None predicted";

    showPanel("summary-panel");

    // Hydrophobicity chart
    drawHydrophobicityChart(
      data.hydrophobicity_profile,
      document.getElementById("hydrophobicity-chart"),
    );
    showPanel("hydrophobicity-panel");

    // Composition chart
    drawCompositionChart(
      data.amino_acid_composition,
      document.getElementById("composition-chart"),
    );
    showPanel("composition-panel");

    // Secondary structure
    displaySecondaryStructure(
      data.secondary_structure,
      data.secondary_structure_summary,
      document.getElementById("ss-visualization"),
    );
    showPanel("ss-panel");
  } catch (err) {
    alert(`Analysis failed: ${err.message}`);
  } finally {
    showLoading(false);
  }
}

async function alignSequences() {
  const seq1 = document.getElementById("align-seq1").value.trim();
  const seq2 = document.getElementById("align-seq2").value.trim();

  if (!seq1 || !seq2) {
    alert("Please enter both sequences for alignment.");
    return;
  }

  const gapOpen = parseFloat(document.getElementById("gap-open").value);
  const gapExtend = parseFloat(document.getElementById("gap-extend").value);
  const optimize = document.getElementById("optimize-gaps").checked;

  showLoading(true);

  try {
    const data = await apiCall("/align", "POST", {
      sequence1: seq1.replace(/\s/g, ""),
      sequence2: seq2.replace(/\s/g, ""),
      gap_open: gapOpen,
      gap_extend: gapExtend,
      optimize_gaps: optimize,
    });

    displayAlignment(data);
    document.getElementById("alignment-result").classList.remove("hidden");
  } catch (err) {
    alert(`Alignment failed: ${err.message}`);
  } finally {
    showLoading(false);
  }
}

async function loadExampleProtein(uniprotId) {
  showLoading(true);
  try {
    const data = await apiCall(`/protein/${uniprotId}`, "GET");
    document.getElementById("sequence-input").value =
      `>${data.name} | ${data.organism} | ${data.gene_name}\n${data.sequence}`;
  } catch (err) {
    alert(`Failed to load protein: ${err.message}`);
  } finally {
    showLoading(false);
  }
}

// ---- File Upload Handler ----

function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById("sequence-input").value = e.target.result;
  };
  reader.readAsText(file);
}

// ---- Initialize ----

document.addEventListener("DOMContentLoaded", () => {
  // Analyze button
  document
    .getElementById("analyze-btn")
    .addEventListener("click", analyzeSequence);

  // Clear button
  document.getElementById("clear-btn").addEventListener("click", () => {
    document.getElementById("sequence-input").value = "";
    document.getElementById("sequence-display").classList.add("hidden");
    document.getElementById("summary-panel").classList.add("hidden");
    document.getElementById("hydrophobicity-panel").classList.add("hidden");
    document.getElementById("composition-panel").classList.add("hidden");
    document.getElementById("ss-panel").classList.add("hidden");
  });

  // File upload
  document
    .getElementById("fasta-upload")
    .addEventListener("change", handleFileUpload);

  // Example protein buttons
  document.querySelectorAll("[data-uniprot]").forEach((btn) => {
    btn.addEventListener("click", () =>
      loadExampleProtein(btn.dataset.uniprot),
    );
  });

  // Align button
  document
    .getElementById("align-btn")
    .addEventListener("click", alignSequences);

  // Keyboard shortcut: Ctrl+Enter to analyze
  document.getElementById("sequence-input").addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.key === "Enter") {
      analyzeSequence();
    }
  });
});
