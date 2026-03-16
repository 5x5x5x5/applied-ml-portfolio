/* ═══════════════════════════════════════════════════════
   BioTech Noir — Portfolio JavaScript
   DNA Helix Canvas Animation + UI Interactions
   ═══════════════════════════════════════════════════════ */

(function () {
  "use strict";

  // ─── DNA Helix Canvas Animation ──────────────────────
  const canvas = document.getElementById("dna-canvas");
  const ctx = canvas.getContext("2d");
  let width,
    height,
    particles,
    time = 0;
  let animFrame;

  const DNA_CONFIG = {
    particleCount: 80,
    helixRadius: 120,
    helixSpeed: 0.008,
    verticalSpread: 1.2,
    connectionOpacity: 0.12,
    particleSize: 2.2,
    glowSize: 6,
    colors: {
      strand1: { r: 0, g: 229, b: 255 }, // cyan
      strand2: { r: 57, g: 255, b: 20 }, // green
      bond: { r: 255, g: 255, b: 255 },
      glow1: "rgba(0, 229, 255, 0.15)",
      glow2: "rgba(57, 255, 20, 0.12)",
    },
  };

  function initCanvas() {
    width = canvas.width = canvas.offsetWidth;
    height = canvas.height = canvas.offsetHeight;
    initParticles();
  }

  function initParticles() {
    particles = [];
    const count = DNA_CONFIG.particleCount;
    for (let i = 0; i < count; i++) {
      particles.push({
        phase: (i / count) * Math.PI * 6,
        y: (i / count) * height * DNA_CONFIG.verticalSpread - height * 0.1,
        speed: 0.3 + Math.random() * 0.2,
        size: 1.5 + Math.random() * 1.5,
      });
    }
  }

  function drawHelix() {
    ctx.clearRect(0, 0, width, height);

    // Subtle background gradient
    const bgGrad = ctx.createRadialGradient(
      width * 0.7,
      height * 0.4,
      0,
      width * 0.7,
      height * 0.4,
      width * 0.6,
    );
    bgGrad.addColorStop(0, "rgba(0, 229, 255, 0.02)");
    bgGrad.addColorStop(1, "transparent");
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, width, height);

    const centerX = width * 0.72;
    const radius = Math.min(DNA_CONFIG.helixRadius, width * 0.15);

    // Draw connecting bonds first (behind strands)
    for (let i = 0; i < particles.length; i += 4) {
      const p = particles[i];
      const angle = p.phase + time * DNA_CONFIG.helixSpeed * p.speed;
      const x1 = centerX + Math.cos(angle) * radius;
      const x2 = centerX + Math.cos(angle + Math.PI) * radius;
      const y = ((p.y + time * 0.3) % (height * 1.4)) - height * 0.1;
      const depth1 = Math.sin(angle);
      const depth2 = Math.sin(angle + Math.PI);

      if (y > -20 && y < height + 20) {
        const bondAlpha =
          DNA_CONFIG.connectionOpacity *
          (0.3 + Math.abs(Math.cos(angle)) * 0.7);
        ctx.beginPath();
        ctx.moveTo(x1, y);
        ctx.lineTo(x2, y);
        ctx.strokeStyle = `rgba(255, 255, 255, ${bondAlpha})`;
        ctx.lineWidth = 0.5;
        ctx.stroke();

        // Bond midpoint marker
        const midX = (x1 + x2) / 2;
        ctx.beginPath();
        ctx.arc(midX, y, 1, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 255, 255, ${bondAlpha * 1.5})`;
        ctx.fill();
      }
    }

    // Draw both strands
    const sortedParticles = particles
      .map((p, i) => {
        const angle = p.phase + time * DNA_CONFIG.helixSpeed * p.speed;
        const y = ((p.y + time * 0.3) % (height * 1.4)) - height * 0.1;
        return [
          {
            x: centerX + Math.cos(angle) * radius,
            y,
            depth: Math.sin(angle),
            strand: 0,
            size: p.size,
            idx: i,
          },
          {
            x: centerX + Math.cos(angle + Math.PI) * radius,
            y,
            depth: Math.sin(angle + Math.PI),
            strand: 1,
            size: p.size,
            idx: i,
          },
        ];
      })
      .flat()
      .filter((p) => p.y > -20 && p.y < height + 20)
      .sort((a, b) => a.depth - b.depth);

    for (const p of sortedParticles) {
      const alpha = 0.3 + (p.depth + 1) * 0.35;
      const scale = 0.6 + (p.depth + 1) * 0.3;
      const c =
        p.strand === 0 ? DNA_CONFIG.colors.strand1 : DNA_CONFIG.colors.strand2;

      // Glow
      const glowGrad = ctx.createRadialGradient(
        p.x,
        p.y,
        0,
        p.x,
        p.y,
        DNA_CONFIG.glowSize * scale,
      );
      glowGrad.addColorStop(0, `rgba(${c.r}, ${c.g}, ${c.b}, ${alpha * 0.3})`);
      glowGrad.addColorStop(1, "transparent");
      ctx.fillStyle = glowGrad;
      ctx.fillRect(
        p.x - DNA_CONFIG.glowSize * scale,
        p.y - DNA_CONFIG.glowSize * scale,
        DNA_CONFIG.glowSize * 2 * scale,
        DNA_CONFIG.glowSize * 2 * scale,
      );

      // Particle
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size * scale, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${c.r}, ${c.g}, ${c.b}, ${alpha})`;
      ctx.fill();
    }

    // Floating ambient particles
    for (let i = 0; i < 30; i++) {
      const fx = ((Math.sin(time * 0.002 + i * 1.7) + 1) / 2) * width;
      const fy = ((Math.cos(time * 0.0015 + i * 2.3) + 1) / 2) * height;
      const fa = 0.03 + Math.sin(time * 0.003 + i) * 0.02;
      ctx.beginPath();
      ctx.arc(fx, fy, 1, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0, 229, 255, ${fa})`;
      ctx.fill();
    }

    time++;
    animFrame = requestAnimationFrame(drawHelix);
  }

  // ─── Navigation Scroll Effect ────────────────────────
  const nav = document.getElementById("nav");
  let lastScroll = 0;

  function handleNavScroll() {
    const scrollY = window.scrollY;
    nav.classList.toggle("scrolled", scrollY > 60);
    lastScroll = scrollY;
  }

  // ─── Counter Animation ───────────────────────────────
  function animateCounters() {
    const counters = document.querySelectorAll("[data-count]");
    counters.forEach((el) => {
      const target = parseInt(el.dataset.count, 10);
      const duration = 1500;
      const start = performance.now();

      function tick(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = Math.round(target * eased);
        if (progress < 1) requestAnimationFrame(tick);
      }

      requestAnimationFrame(tick);
    });
  }

  // ─── Project Filtering ───────────────────────────────
  const filterBtns = document.querySelectorAll(".filter-btn");
  const projectCards = document.querySelectorAll(".project-card");
  const projectGrid = document.querySelector(".project-grid");

  filterBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const filter = btn.dataset.filter;

      // Update active button
      filterBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      // Filter cards
      projectCards.forEach((card) => {
        const category = card.dataset.category;
        const show = filter === "all" || category === filter;

        if (show) {
          card.dataset.visible = "true";
          card.style.position = "";
          card.style.visibility = "";
        } else {
          card.dataset.visible = "false";
        }
      });
    });
  });

  // ─── Hamburger Menu ─────────────────────────────────
  const hamburger = document.getElementById("nav-hamburger");
  const navLinks = document.getElementById("nav-links");

  if (hamburger && navLinks) {
    hamburger.addEventListener("click", () => {
      hamburger.classList.toggle("active");
      navLinks.classList.toggle("open");
    });

    // Close menu when a link is clicked
    navLinks.querySelectorAll(".nav-link").forEach((link) => {
      link.addEventListener("click", () => {
        hamburger.classList.remove("active");
        navLinks.classList.remove("open");
      });
    });
  }

  // ─── Scroll Reveal ───────────────────────────────────
  function setupScrollReveal() {
    // Mark elements for reveal
    document
      .querySelectorAll(
        ".expertise-card, .about-detail-card, .tech-category, .skill-card",
      )
      .forEach((el) => {
        el.classList.add("reveal");
      });

    document
      .querySelectorAll(
        ".expertise-grid, .project-grid, .tech-grid, .about-details, .skills-grid",
      )
      .forEach((el) => {
        el.classList.add("stagger-children");
      });

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("visible");
            // Don't unobserve stagger-children so cards are always visible
          }
        });
      },
      { threshold: 0.1, rootMargin: "0px 0px -50px 0px" },
    );

    document.querySelectorAll(".reveal, .stagger-children").forEach((el) => {
      observer.observe(el);
    });
  }

  // ─── Smooth Scroll for Nav Links ─────────────────────
  document.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.addEventListener("click", (e) => {
      const target = document.querySelector(link.getAttribute("href"));
      if (target) {
        e.preventDefault();
        const offset = nav.offsetHeight + 20;
        const top =
          target.getBoundingClientRect().top + window.scrollY - offset;
        window.scrollTo({ top, behavior: "smooth" });
      }
    });
  });

  // ─── Card Hover Glow Follow ──────────────────────────
  document.querySelectorAll(".project-card-inner").forEach((card) => {
    card.addEventListener("mousemove", (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      card.style.setProperty("--mouse-x", `${x}px`);
      card.style.setProperty("--mouse-y", `${y}px`);
      card.style.background = `radial-gradient(circle 200px at ${x}px ${y}px, rgba(255,255,255,0.03), var(--bg-card-solid))`;
    });

    card.addEventListener("mouseleave", () => {
      card.style.background = "var(--bg-card-solid)";
    });
  });

  // ─── Parallax Subtle on Hero ─────────────────────────
  function handleParallax() {
    const scrollY = window.scrollY;
    const heroContent = document.querySelector(".hero-content");
    if (heroContent && scrollY < window.innerHeight) {
      heroContent.style.transform = `translateY(${scrollY * 0.15}px)`;
      heroContent.style.opacity = 1 - scrollY / (window.innerHeight * 0.8);
    }
  }

  // ─── Initialize ──────────────────────────────────────
  function init() {
    initCanvas();
    drawHelix();
    setupScrollReveal();

    // Delayed counter animation
    setTimeout(animateCounters, 1200);

    // Scroll handlers (throttled)
    let ticking = false;
    window.addEventListener(
      "scroll",
      () => {
        if (!ticking) {
          requestAnimationFrame(() => {
            handleNavScroll();
            handleParallax();
            ticking = false;
          });
          ticking = true;
        }
      },
      { passive: true },
    );

    // Resize handler
    let resizeTimer;
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        cancelAnimationFrame(animFrame);
        initCanvas();
        drawHelix();
      }, 200);
    });
  }

  // Wait for fonts to load
  if (document.fonts) {
    document.fonts.ready.then(init);
  } else {
    window.addEventListener("DOMContentLoaded", init);
  }
})();
