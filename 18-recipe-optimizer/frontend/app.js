/**
 * NutriOptimize Frontend Application
 *
 * Handles recipe input, API communication, and chart rendering.
 * Uses vanilla JS with Canvas API for charts (no external chart libraries).
 */

const API_BASE = window.location.origin;

// ============================================================
// Ingredient Management
// ============================================================

function addIngredient() {
  const list = document.getElementById("ingredients-list");
  const row = document.createElement("div");
  row.className = "ingredient-row";
  row.innerHTML = `
        <input type="text" class="ing-name" placeholder="Ingredient name" list="ingredient-suggestions">
        <input type="number" class="ing-qty" placeholder="Grams" min="1" step="1">
        <button class="btn-remove" onclick="removeIngredient(this)" title="Remove">&#10005;</button>
    `;
  list.appendChild(row);
  row.querySelector(".ing-name").focus();
}

function removeIngredient(button) {
  const list = document.getElementById("ingredients-list");
  if (list.children.length > 1) {
    button.closest(".ingredient-row").remove();
  }
}

function getIngredients() {
  const rows = document.querySelectorAll(".ingredient-row");
  const ingredients = [];
  rows.forEach((row) => {
    const name = row.querySelector(".ing-name").value.trim();
    const qty = parseFloat(row.querySelector(".ing-qty").value);
    if (name && qty > 0) {
      ingredients.push({ name, grams: qty });
    }
  });
  return ingredients;
}

function getGoal() {
  const checked = document.querySelector('input[name="goal"]:checked');
  return checked ? checked.value : "balanced";
}

function getRestrictions() {
  const checked = document.querySelectorAll(
    'input[name="restriction"]:checked',
  );
  return Array.from(checked).map((cb) => cb.value);
}

function getServings() {
  return parseInt(document.getElementById("servings").value) || 4;
}

// ============================================================
// Ingredient Search / Autocomplete
// ============================================================

let searchTimeout = null;

document.addEventListener("input", (e) => {
  if (e.target.classList.contains("ing-name")) {
    clearTimeout(searchTimeout);
    const query = e.target.value.trim();
    if (query.length >= 2) {
      searchTimeout = setTimeout(() => searchIngredients(query), 300);
    }
  }
});

async function searchIngredients(query) {
  try {
    const res = await fetch(
      `${API_BASE}/ingredients/search?q=${encodeURIComponent(query)}&limit=8`,
    );
    if (!res.ok) return;
    const data = await res.json();
    const datalist = document.getElementById("ingredient-suggestions");
    datalist.innerHTML = "";
    data.forEach((item) => {
      const option = document.createElement("option");
      option.value = item.name;
      option.textContent = `${item.name} (${item.calories_per_100g} kcal/100g)`;
      datalist.appendChild(option);
    });
  } catch {
    // Autocomplete failure is non-critical
  }
}

// ============================================================
// Goal selector interaction
// ============================================================

document.querySelectorAll(".goal-option").forEach((option) => {
  option.addEventListener("click", () => {
    document
      .querySelectorAll(".goal-option")
      .forEach((o) => o.classList.remove("selected"));
    option.classList.add("selected");
  });
});

// ============================================================
// API Calls
// ============================================================

function showLoading() {
  document.getElementById("loading").classList.remove("hidden");
  document.getElementById("results").classList.add("hidden");
}

function hideLoading() {
  document.getElementById("loading").classList.add("hidden");
}

function showError(message) {
  hideLoading();
  alert(message);
}

async function analyzeRecipe() {
  const ingredients = getIngredients();
  if (ingredients.length === 0) {
    showError("Please add at least one ingredient with a quantity.");
    return;
  }

  showLoading();

  try {
    const res = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ingredients,
        num_servings: getServings(),
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Analysis failed");
    }

    const data = await res.json();
    hideLoading();
    displayAnalysis(data);
  } catch (err) {
    showError(`Error: ${err.message}`);
  }
}

async function optimizeRecipe() {
  const ingredients = getIngredients();
  if (ingredients.length === 0) {
    showError("Please add at least one ingredient with a quantity.");
    return;
  }

  showLoading();

  try {
    const res = await fetch(`${API_BASE}/optimize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ingredients,
        num_servings: getServings(),
        goal: getGoal(),
        restrictions: getRestrictions(),
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Optimization failed");
    }

    const data = await res.json();

    // Also run analysis for the charts
    const analysisRes = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ingredients,
        num_servings: getServings(),
      }),
    });

    let analysisData = null;
    if (analysisRes.ok) {
      analysisData = await analysisRes.json();
    }

    hideLoading();
    if (analysisData) {
      displayAnalysis(analysisData);
    }
    displayOptimization(data);
  } catch (err) {
    showError(`Error: ${err.message}`);
  }
}

// ============================================================
// Display Functions
// ============================================================

function displayAnalysis(data) {
  document.getElementById("results").classList.remove("hidden");
  document.getElementById("analysis-results").classList.remove("hidden");

  const ps = data.per_serving;

  // Macro values
  document.getElementById("cal-value").textContent = Math.round(ps.calories);
  document.getElementById("protein-value").textContent =
    ps.protein_g.toFixed(1);
  document.getElementById("carbs-value").textContent = ps.carbs_g.toFixed(1);
  document.getElementById("fat-value").textContent = ps.fat_g.toFixed(1);
  document.getElementById("fiber-value").textContent = ps.fiber_g.toFixed(1);

  // Draw charts
  drawMacroPieChart(data.calorie_breakdown);
  drawMicroBarChart(data.micronutrient_scores);

  // Completeness score
  const score = data.completeness_score;
  document.getElementById("score-text").textContent = Math.round(score);
  const circle = document.getElementById("score-circle");
  const circumference = 2 * Math.PI * 50; // r=50
  const offset = circumference * (1 - score / 100);
  circle.style.strokeDasharray = circumference;
  circle.style.strokeDashoffset = offset;

  // Color based on score
  if (score >= 70) {
    circle.style.stroke = "#4caf50";
  } else if (score >= 40) {
    circle.style.stroke = "#f5a623";
  } else {
    circle.style.stroke = "#d32f2f";
  }

  // Allergens
  const allergenSection = document.getElementById("allergen-section");
  const allergenList = document.getElementById("allergen-list");
  if (data.allergens && data.allergens.length > 0) {
    allergenSection.classList.remove("hidden");
    allergenList.innerHTML = data.allergens
      .map(
        (a) =>
          `<span class="allergen-tag">${a.replace("_", " ").toUpperCase()}</span>`,
      )
      .join("");
  } else {
    allergenSection.classList.add("hidden");
  }

  // Warnings
  const warningsSection = document.getElementById("warnings-section");
  const warningsList = document.getElementById("warnings-list");
  if (data.warnings && data.warnings.length > 0) {
    warningsSection.classList.remove("hidden");
    warningsList.innerHTML = data.warnings.map((w) => `<li>${w}</li>`).join("");
  } else {
    warningsSection.classList.add("hidden");
  }

  // Bioavailability
  const bioSection = document.getElementById("bio-section");
  const bioList = document.getElementById("bio-list");
  if (data.bioavailability_notes && data.bioavailability_notes.length > 0) {
    bioSection.classList.remove("hidden");
    bioList.innerHTML = data.bioavailability_notes
      .map((n) => `<li>${n}</li>`)
      .join("");
  } else {
    bioSection.classList.add("hidden");
  }
}

function displayOptimization(data) {
  const section = document.getElementById("optimization-results");
  section.classList.remove("hidden");

  const servings = getServings();

  // Original vs Optimized comparison
  document.getElementById("orig-cal").textContent =
    Math.round(data.original.calories / servings) + " kcal";
  document.getElementById("orig-protein").textContent =
    (data.original.protein_g / servings).toFixed(1) + "g";
  document.getElementById("orig-fiber").textContent =
    (data.original.fiber_g / servings).toFixed(1) + "g";
  document.getElementById("orig-taste").textContent =
    (data.original_taste_score * 100).toFixed(0) + "%";
  document.getElementById("orig-comp").textContent =
    data.original_completeness.toFixed(1);

  document.getElementById("opt-cal").textContent =
    Math.round(data.optimized.calories / servings) + " kcal";
  document.getElementById("opt-protein").textContent =
    (data.optimized.protein_g / servings).toFixed(1) + "g";
  document.getElementById("opt-fiber").textContent =
    (data.optimized.fiber_g / servings).toFixed(1) + "g";
  document.getElementById("opt-taste").textContent =
    (data.optimized_taste_score * 100).toFixed(0) + "%";
  document.getElementById("opt-comp").textContent =
    data.optimized_completeness.toFixed(1);

  // Ingredient changes
  const changesList = document.getElementById("changes-list");
  changesList.innerHTML = "";
  const origIng = data.original_ingredients;
  const optIng = data.optimized_ingredients;

  for (const name of Object.keys(origIng)) {
    const origVal = origIng[name];
    const optVal = optIng[name] || 0;
    const diff = optVal - origVal;
    const pct = origVal > 0 ? ((diff / origVal) * 100).toFixed(0) : 0;

    let diffClass = "same";
    let diffText = "No change";
    if (Math.abs(diff) > 0.5) {
      diffClass = diff > 0 ? "increase" : "decrease";
      diffText = `${diff > 0 ? "+" : ""}${diff.toFixed(0)}g (${diff > 0 ? "+" : ""}${pct}%)`;
    }

    changesList.innerHTML += `
            <div class="change-item">
                <span class="change-name">${name}</span>
                <span>
                    ${origVal.toFixed(0)}g
                    <span class="change-diff ${diffClass}">${diffText}</span>
                    ${optVal.toFixed(0)}g
                </span>
            </div>
        `;
  }

  // Substitutions
  const subSection = document.getElementById("substitution-section");
  const subList = document.getElementById("substitution-list");
  if (data.substitutions && data.substitutions.length > 0) {
    subSection.classList.remove("hidden");
    subList.innerHTML = data.substitutions
      .map(
        (s) => `
            <div class="sub-item">
                <div class="sub-swap">${s.original} &#10148; ${s.substitute}</div>
                <div class="sub-reason">${s.reason}</div>
            </div>
        `,
      )
      .join("");
  } else {
    subSection.classList.add("hidden");
  }

  // Optimization notes
  const notesSection = document.getElementById("opt-notes-section");
  const notesList = document.getElementById("opt-notes-list");
  if (data.notes && data.notes.length > 0) {
    notesSection.classList.remove("hidden");
    notesList.innerHTML = data.notes.map((n) => `<li>${n}</li>`).join("");
  } else {
    notesSection.classList.add("hidden");
  }
}

// ============================================================
// Chart Drawing (Canvas API - no external dependencies)
// ============================================================

function drawMacroPieChart(breakdown) {
  const canvas = document.getElementById("macroChart");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;

  canvas.width = 280 * dpr;
  canvas.height = 280 * dpr;
  canvas.style.width = "280px";
  canvas.style.height = "280px";
  ctx.scale(dpr, dpr);

  const cx = 140;
  const cy = 130;
  const radius = 90;

  const segments = [
    { label: "Protein", pct: breakdown.protein_pct || 0, color: "#e74c3c" },
    { label: "Carbs", pct: breakdown.carbs_pct || 0, color: "#f39c12" },
    { label: "Fat", pct: breakdown.fat_pct || 0, color: "#3498db" },
  ];

  // Clear
  ctx.clearRect(0, 0, 280, 280);

  let startAngle = -Math.PI / 2;
  const total = segments.reduce((sum, s) => sum + s.pct, 0) || 1;

  segments.forEach((seg) => {
    const sliceAngle = (seg.pct / total) * 2 * Math.PI;

    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, radius, startAngle, startAngle + sliceAngle);
    ctx.closePath();
    ctx.fillStyle = seg.color;
    ctx.fill();

    // Label
    if (seg.pct > 5) {
      const midAngle = startAngle + sliceAngle / 2;
      const labelR = radius * 0.65;
      const lx = cx + Math.cos(midAngle) * labelR;
      const ly = cy + Math.sin(midAngle) * labelR;
      ctx.fillStyle = "white";
      ctx.font = "bold 13px Nunito";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(`${seg.pct.toFixed(0)}%`, lx, ly);
    }

    startAngle += sliceAngle;
  });

  // Center circle (donut)
  ctx.beginPath();
  ctx.arc(cx, cy, radius * 0.45, 0, 2 * Math.PI);
  ctx.fillStyle = "#ffffff";
  ctx.fill();

  // Center text
  ctx.fillStyle = "#2d1b0e";
  ctx.font = "bold 14px Nunito";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("Macros", cx, cy);

  // Legend
  const legendY = 260;
  segments.forEach((seg, i) => {
    const lx = 30 + i * 90;
    ctx.fillStyle = seg.color;
    ctx.fillRect(lx, legendY, 12, 12);
    ctx.fillStyle = "#5c4033";
    ctx.font = "11px Nunito";
    ctx.textAlign = "left";
    ctx.fillText(seg.label, lx + 16, legendY + 10);
  });
}

function drawMicroBarChart(scores) {
  const canvas = document.getElementById("microChart");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;

  canvas.width = 400 * dpr;
  canvas.height = 280 * dpr;
  canvas.style.width = "400px";
  canvas.style.height = "280px";
  ctx.scale(dpr, dpr);

  ctx.clearRect(0, 0, 400, 280);

  const nutrients = Object.entries(scores);
  const barWidth = 24;
  const gap = (380 - nutrients.length * barWidth) / (nutrients.length + 1);
  const maxH = 200;
  const baseY = 240;

  // 100% line
  const lineY = baseY - maxH * (100 / 200);
  ctx.strokeStyle = "#c9b99a";
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 3]);
  ctx.beginPath();
  ctx.moveTo(10, lineY);
  ctx.lineTo(390, lineY);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#8b7355";
  ctx.font = "10px Nunito";
  ctx.textAlign = "right";
  ctx.fillText("100% DRI", 390, lineY - 4);

  nutrients.forEach(([name, value], i) => {
    const x = gap + i * (barWidth + gap);
    const cappedVal = Math.min(value, 200);
    const barH = (cappedVal / 200) * maxH;

    // Bar gradient
    let color;
    if (value >= 100) {
      color = "#4caf50";
    } else if (value >= 50) {
      color = "#f5a623";
    } else if (value >= 25) {
      color = "#ff9800";
    } else {
      color = "#d32f2f";
    }

    // Bar
    ctx.fillStyle = color;
    const barRadius = 4;
    roundedRect(ctx, x, baseY - barH, barWidth, barH, barRadius);
    ctx.fill();

    // Value label
    ctx.fillStyle = "#2d1b0e";
    ctx.font = "bold 9px Nunito";
    ctx.textAlign = "center";
    ctx.fillText(`${Math.round(value)}%`, x + barWidth / 2, baseY - barH - 6);

    // Nutrient label (rotated)
    ctx.save();
    ctx.translate(x + barWidth / 2, baseY + 8);
    ctx.rotate(-Math.PI / 4);
    ctx.fillStyle = "#5c4033";
    ctx.font = "9px Nunito";
    ctx.textAlign = "right";
    // Shorten long names
    const shortName = name.replace("Vitamin ", "Vit ");
    ctx.fillText(shortName, 0, 0);
    ctx.restore();
  });
}

function roundedRect(ctx, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x, y + h);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

// ============================================================
// Initialize
// ============================================================

document.addEventListener("DOMContentLoaded", () => {
  // Pre-fill example recipe for better UX
  const rows = document.querySelectorAll(".ingredient-row");
  const examples = [
    { name: "chicken breast", qty: 200 },
    { name: "brown rice cooked", qty: 300 },
    { name: "broccoli", qty: 150 },
  ];

  rows.forEach((row, i) => {
    if (examples[i]) {
      row.querySelector(".ing-name").value = examples[i].name;
      row.querySelector(".ing-qty").value = examples[i].qty;
    }
  });
});
