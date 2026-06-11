/**
 * Generates an academic-style "project brief" page for every portfolio
 * project into projects/<slug>.html, sharing projects/project.css.
 *
 * Run:  node build-project-pages.mjs
 *
 * Content is hand-authored per project (abstract, highlights, architecture)
 * and kept in sync with each project's README. Edit the PROJECTS array and
 * re-run to regenerate.
 */
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO = "https://github.com/5x5x5x5/applied-ml-portfolio/tree/main";

const PROJECTS = [
  {
    n: 1,
    slug: "01-pharma-sentinel",
    name: "PharmaSentinel",
    accent: "cyan",
    badge: "DE 1",
    area: "Cloud AI/ML",
    subtitle: "Drug Adverse Event Detection Pipeline",
    icon: `<path d="M20 7h-3a2 2 0 0 1-2-2V2"/><path d="M9 18a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h7l4 4v10a2 2 0 0 1-2 2H9z"/><path d="M3 7.6v12.8A1.6 1.6 0 0 0 4.6 22h9.8"/><path d="M12 10v4M10 12h4"/>`,
    abstract: `An end-to-end NLP pipeline that ingests <strong>FDA FAERS</strong> adverse event reports and classifies drug events by clinical severity. Built for cloud-native operation on ECS Fargate with a full CloudFormation footprint.`,
    highlights: [
      `<strong>FAERS-native ingestion</strong> — parses the authentic ICHICSR schema and pipe-delimited quarterly extracts (primaryid, drugname, reaction PT, outcome codes).`,
      `<strong>Severity classifier</strong> — TF-IDF features feeding a logistic-regression model with stratified cross-validation and feature-importance attribution.`,
      `<strong>Serverless data plane</strong> — S3 data lake, SQS event routing, and auto-scaling Fargate inference services.`,
      `<strong>Observability</strong> — Datadog APM tracing across the pipeline with infrastructure defined entirely in CloudFormation.`,
    ],
    arch: `FAERS extract → S3 raw → ingestion (SQS) → NLP classifier (FastAPI) → severity store\n                                  ↘ Datadog APM / CloudWatch`,
    tags: [
      "Python",
      "FastAPI",
      "AWS ECS",
      "CloudFormation",
      "Docker",
      "NLP",
      "DataDog",
      "S3",
      "SQS",
      "Lambda",
    ],
  },
  {
    n: 2,
    slug: "02-cloud-genomics",
    name: "CloudGenomics",
    accent: "cyan",
    badge: "DE 1",
    area: "Cloud AI/ML",
    subtitle: "Genomic Variant Classification Service",
    icon: `<circle cx="12" cy="12" r="3"/><path d="M12 2a7 7 0 0 1 7 7c0 3-2 5.5-4 7.5S12 22 12 22s-1-3-3-5.5S5 12 5 9a7 7 0 0 1 7-7z"/>`,
    abstract: `A HIPAA-aware ML service that classifies genetic variants (SNPs, indels) along the <strong>ACMG/AMP</strong> five-tier spectrum from benign to pathogenic, using population frequency, conservation, and in-silico functional scores.`,
    highlights: [
      `<strong>ACMG/AMP evidence framework</strong> — predictions are explained with the standard criteria codes (BA1/BS1 for common variants, PM2 for rarity, PVS1 for loss-of-function, PP3/BP4 for in-silico concordance).`,
      `<strong>Calibrated in-silico thresholds</strong> — SIFT < 0.05, PolyPhen-2 > 0.85, CADD > 25, REVEL > 0.7, matching published damaging cut-offs.`,
      `<strong>VCF processing</strong> — parses standard VCF, derives conservation composites (PhyloP, GERP), and flags frameshift vs in-frame indels.`,
      `<strong>Compliance posture</strong> — VPC endpoints, KMS encryption, WAF, and Step Functions orchestration.`,
    ],
    arch: `VCF → variant parser → feature vector (AF · conservation · in-silico)\n     → RandomForest classifier → ACMG tier + evidence codes`,
    tags: [
      "Python",
      "RandomForest",
      "BioPython",
      "VCF",
      "Step Functions",
      "CloudFormation",
      "KMS",
      "DataDog",
    ],
  },
  {
    n: 3,
    slug: "03-feature-forge",
    name: "FeatureForge",
    accent: "green",
    badge: "DE 2",
    area: "MLOps",
    subtitle: "ML Feature Store & Drift Detection",
    icon: `<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><path d="M17.5 14v7M14 17.5h7"/>`,
    abstract: `An enterprise feature store on <strong>Snowflake</strong> with point-in-time-correct retrieval, feature versioning, and lineage — paired with an automated drift-detection layer wired into SageMaker Model Monitor.`,
    highlights: [
      `<strong>Point-in-time correctness</strong> — training-serving skew avoided through as-of joins and versioned feature definitions.`,
      `<strong>Drift detection</strong> — textbook Population Stability Index (PSI), Kolmogorov–Smirnov, and chi-squared tests with standard severity bands (PSI 0.1 / 0.2 / 0.3).`,
      `<strong>Orchestration</strong> — Airflow DAGs schedule feature materialization and drift scans.`,
      `<strong>Model monitoring</strong> — SageMaker Model Monitor integration with automated retraining triggers.`,
    ],
    arch: `sources → Snowflake feature tables (versioned)\n        → point-in-time retrieval → training / serving\n        → drift detector (PSI · KS · chi²) → SageMaker Model Monitor`,
    tags: [
      "Snowflake",
      "SnowSQL",
      "Python",
      "Airflow DAG",
      "SageMaker",
      "Model Monitor",
      "PSI",
      "Feature Store",
    ],
  },
  {
    n: 4,
    slug: "04-drug-interaction-ml",
    name: "DrugInteractionML",
    accent: "green",
    badge: "DE 2",
    area: "MLOps",
    subtitle: "Drug-Drug Interaction Prediction",
    icon: `<circle cx="7" cy="12" r="3"/><circle cx="17" cy="12" r="3"/><path d="M10 12h4"/><path d="M7 9V4M17 9V4M7 15v5M17 15v5"/>`,
    abstract: `An XGBoost pipeline that predicts adverse drug-drug interactions and their severity from molecular fingerprints, co-prescription patterns, and Snowflake-derived patient features.`,
    highlights: [
      `<strong>Molecular features</strong> — RDKit descriptors (MolWt, LogP, TPSA, H-bond donors/acceptors, Morgan fingerprints) with Tanimoto/Dice similarity between drug pairs.`,
      `<strong>Severity modelling</strong> — multi-class interaction severity with SHAP explanations for each prediction.`,
      `<strong>Signal features</strong> — Snowflake SQL derives proportional reporting ratios and co-prescription overlap windows.`,
      `<strong>MLOps</strong> — MLflow experiment tracking, Step Functions orchestration, and drift-triggered retraining.`,
    ],
    arch: `drug pair → molecular FPs + patient features (Snowflake)\n          → XGBoost (interaction + severity) → SHAP attribution`,
    tags: [
      "XGBoost",
      "Snowflake",
      "Airflow",
      "Step Functions",
      "MLflow",
      "SageMaker",
      "SHAP",
      "SMILES",
    ],
  },
  {
    n: 5,
    slug: "05-rx-predict",
    name: "RxPredict",
    accent: "magenta",
    badge: "DE 3",
    area: "Full Stack / Low Latency",
    subtitle: "Real-time Drug Response Prediction",
    icon: `<path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>`,
    abstract: `A sub-100&nbsp;ms pharmacogenomic prediction API that infers patient drug response from genetic profile, demographics, and history — engineered for low-latency serving with performance gates in CI.`,
    highlights: [
      `<strong>Pharmacogenomics</strong> — real star-allele catalog (CYP2D6, CYP2C19, TPMT, DPYD&nbsp;*2A, UGT1A1&nbsp;*28, SLCO1B1&nbsp;*5, HLA-B*57:01) mapped to metabolizer phenotypes.`,
      `<strong>Latency engineering</strong> — optimized scikit-learn pipelines, Redis caching, and feature hashing to hold a &lt;100&nbsp;ms p99.`,
      `<strong>Resilience</strong> — circuit breakers and latency benchmarking baked into the request path.`,
      `<strong>Performance CI</strong> — GitHub Actions gates that fail the build on latency regressions.`,
    ],
    arch: `request → feature hashing → Redis cache → sklearn model\n        → response (<100ms p99)   ↘ Prometheus latency metrics`,
    tags: [
      "FastAPI",
      "Redis",
      "scikit-learn",
      "Prometheus",
      "GitHub Actions",
      "Docker",
      "<100ms p99",
      "CI/CD",
    ],
  },
  {
    n: 6,
    slug: "06-biomarker-dash",
    name: "BiomarkerDash",
    accent: "magenta",
    badge: "DE 3",
    area: "Full Stack / Low Latency",
    subtitle: "Real-time Biomarker Monitoring Dashboard",
    icon: `<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>`,
    abstract: `A full-stack clinical dashboard that streams patient biomarker data over WebSockets, detects anomalies with ML, and escalates clinical alerts against validated reference ranges.`,
    highlights: [
      `<strong>Clinically accurate ranges</strong> — reference intervals and critical thresholds verified against ICU conventions (K⁺ &gt; 6.5, glucose &lt; 40, SpO₂ &lt; 88, troponin &gt; 0.4).`,
      `<strong>Anomaly detection</strong> — Isolation Forest plus z-score and OLS-trend analysis on streaming vitals.`,
      `<strong>Real-time transport</strong> — WebSocket streaming with Canvas-rendered charts and Redis-backed state.`,
      `<strong>Alert escalation</strong> — tiered clinical alerting with sound clinical-value logic.`,
    ],
    arch: `device stream → WebSocket → anomaly detector (IForest · z-score)\n             → alert engine (critical thresholds) → live Canvas dashboard`,
    tags: [
      "FastAPI",
      "WebSocket",
      "Redis",
      "Isolation Forest",
      "Canvas API",
      "CI/CD",
      "Real-time",
    ],
  },
  {
    n: 7,
    slug: "07-pharma-data-vault",
    name: "PharmaDataVault",
    accent: "amber",
    badge: "DE 4",
    area: "Data / ETL",
    subtitle: "Pharmaceutical Data Warehouse",
    icon: `<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4.03 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/>`,
    abstract: `A <strong>Data Vault 2.0</strong> warehouse for clinical trial, manufacturing, and regulatory data, with hub/link/satellite modelling, hash-based change detection, and star-schema marts.`,
    highlights: [
      `<strong>Data Vault 2.0</strong> — MD5 hash keys over cleaned business keys, hashdiff change detection with NVL placeholders, and insert-only satellites.`,
      `<strong>Dimensional marts</strong> — star schemas built downstream for analytics consumption.`,
      `<strong>Data quality</strong> — completeness, orphan, and uniqueness checks with correct denominators; DEA schedule and NDC 5-4-2 validation.`,
      `<strong>Automation</strong> — PL/SQL ETL scheduled by Control-M with UNIX automation scripts.`,
    ],
    arch: `sources → staging (hash keys + hashdiff)\n        → hubs / links / satellites → star-schema marts`,
    tags: [
      "Data Vault 2.0",
      "PL/SQL",
      "Star Schema",
      "ETL",
      "Control-M",
      "UNIX",
      "Great Expectations",
    ],
  },
  {
    n: 8,
    slug: "08-reg-record",
    name: "RegRecord",
    accent: "amber",
    badge: "DE 4",
    area: "Data / ETL",
    subtitle: "Regulatory Record Keeping System",
    icon: `<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M12 18v-6"/><path d="M9 15h6"/>`,
    abstract: `A regulatory record system with a full audit trail, a submission-workflow state machine, and cryptographic pseudonymization with strict separation-of-duties for re-identification.`,
    highlights: [
      `<strong>Pseudonymization</strong> — 32-byte secret salts, SHA-256/512 and HMAC/BLAKE2B keyed hashing, with re-identification gated by separation-of-duties.`,
      `<strong>Audit trail</strong> — PL/SQL triggers capture immutable change history; purge logic protects records with re-id history.`,
      `<strong>Submission workflow</strong> — explicit state machine for regulatory submission lifecycle.`,
      `<strong>Compliance monitoring</strong> — Control-M automated checks in a UNIX environment.`,
    ],
    arch: `record → pseudo-ID (salt + keyed hash) → audit-logged store\n      → submission state machine → compliance monitor`,
    tags: [
      "PL/SQL",
      "Pseudo Records",
      "Audit Trail",
      "Control-M",
      "UNIX",
      "FastAPI",
      "ETL",
      "Compliance",
    ],
  },
  {
    n: 9,
    slug: "09-cell-vision",
    name: "CellVision",
    accent: "teal",
    badge: "BIO",
    area: "Bio & Vision",
    subtitle: "Microscopy Cell Type Classifier",
    icon: `<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="1"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4"/>`,
    abstract: `A PyTorch pipeline that classifies seven peripheral-blood cell types from microscopy images, with stain normalization and GradCAM interpretability behind a Streamlit interface.`,
    highlights: [
      `<strong>Seven cell types</strong> — RBC, neutrophil, lymphocyte, monocyte, eosinophil, basophil, and platelet, with morphology-grounded descriptions.`,
      `<strong>Two architectures</strong> — a custom CellNet CNN (4 conv blocks, global average pooling) and ResNet18 transfer learning.`,
      `<strong>Interpretability</strong> — GradCAM activation maps over the final convolutional layer.`,
      `<strong>Preprocessing</strong> — stain normalization and Otsu segmentation; logits feed CrossEntropyLoss with softmax at inference only.`,
    ],
    arch: `smear image → stain normalization → CNN / ResNet18\n           → softmax over 7 classes → GradCAM overlay`,
    tags: [
      "PyTorch",
      "Computer Vision",
      "ResNet18",
      "GradCAM",
      "Streamlit",
      "Microscopy",
    ],
  },
  {
    n: 10,
    slug: "10-molecule-gen",
    name: "MoleculeGen",
    accent: "teal",
    badge: "PHARMA",
    area: "Generative Chemistry",
    subtitle: "AI Drug Molecule Generator",
    icon: `<path d="M9 3h6v11a6 6 0 0 1-6 0V3z"/><path d="M12 14v-4"/><path d="M9 10h6"/><circle cx="12" cy="20" r="2"/>`,
    abstract: `A Variational Autoencoder over <strong>SMILES</strong> strings that learns a continuous latent space of drug-like molecules, enabling property-conditioned generation and latent-space optimization.`,
    highlights: [
      `<strong>SMILES VAE</strong> — bidirectional GRU encoder/decoder with the reparameterization trick over a 128-dim latent space.`,
      `<strong>Drug-likeness filters</strong> — Lipinski, Veber, Ghose, and lead-likeness rules with QED scoring and PAINS alerts.`,
      `<strong>Latent optimization</strong> — interpolation and property-conditioned sampling for lead generation.`,
      `<strong>Cheminformatics</strong> — RDKit descriptors, Morgan/MACCS fingerprints, and Tanimoto similarity.`,
    ],
    arch: `SMILES → tokenizer → GRU encoder → z ~ N(μ,σ)\n      → GRU decoder → SMILES  (ELBO = recon + β·KL)`,
    tags: [
      "PyTorch",
      "VAE",
      "SMILES",
      "RDKit",
      "Drug Discovery",
      "Generative AI",
    ],
  },
  {
    n: 11,
    slug: "11-plant-pathologist",
    name: "PlantPathologist",
    accent: "teal",
    badge: "BIO",
    area: "Applied Vision",
    subtitle: "Plant Disease Detection",
    icon: `<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><path d="M9 9h.01M15 9h.01"/><path d="M7 3.34C4.07 5.22 2 8.36 2 12"/><path d="M17 3.34C19.93 5.22 22 8.36 22 12"/>`,
    abstract: `A mobile-friendly disease detector that identifies 18 diseases across five crop species from leaf photos and pairs each diagnosis with pathogen-appropriate treatment guidance.`,
    highlights: [
      `<strong>23-class taxonomy</strong> — 18 diseases plus 5 healthy classes across tomato, potato, corn, apple, and grape.`,
      `<strong>Pathogen-aware advice</strong> — correctly types oomycete (late blight), bacterial, and viral diseases; recommends vector control rather than fungicide for viral cases.`,
      `<strong>Transfer learning</strong> — EfficientNet-B0 with temperature-scaled confidence calibration.`,
      `<strong>Field-ready</strong> — camera-enabled web interface with an accurate Latin-name disease knowledge base.`,
    ],
    arch: `leaf photo → EfficientNet-B0 → 23-class softmax\n          → disease database → pathogen type + treatment`,
    tags: [
      "PyTorch",
      "EfficientNet",
      "Transfer Learning",
      "Mobile Web",
      "Streamlit",
    ],
  },
  {
    n: 12,
    slug: "12-protein-explorer",
    name: "ProteinExplorer",
    accent: "teal",
    badge: "BIO",
    area: "Bioinformatics",
    subtitle: "Protein Structure Analysis Tool",
    icon: `<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10"/><path d="M16 4c1 2 2 4 2 8"/><path d="M12 2c2 2 4 5 4 10s-2 8-4 10"/><path d="M2 12h10"/><path d="M12 12h10"/>`,
    abstract: `An interactive protein sequence analysis toolkit covering hydrophobicity, secondary-structure prediction, pairwise alignment, and physicochemical property estimation with SVG dashboards.`,
    highlights: [
      `<strong>Sequence biochemistry</strong> — monoisotopic and average residue masses, Kyte–Doolittle hydropathy, side-chain pKa/pI via Henderson–Hasselbalch.`,
      `<strong>Alignment</strong> — Needleman–Wunsch with affine (Gotoh) gaps scored on the full BLOSUM62 matrix.`,
      `<strong>Structure prediction</strong> — Chou–Fasman secondary-structure propensities and disorder prediction.`,
      `<strong>Visualization</strong> — SVG-based profiles served through a FastAPI dashboard.`,
    ],
    arch: `sequence → hydropathy + composition\n        → Chou–Fasman SS · disorder\n        → Needleman–Wunsch (BLOSUM62) alignment → SVG`,
    tags: [
      "BioPython",
      "Sequence Analysis",
      "Alignment",
      "FastAPI",
      "SVG Charts",
      "Biochemistry",
    ],
  },
  {
    n: 13,
    slug: "13-llm-pharma-assistant",
    name: "PharmAssistAI",
    accent: "violet",
    badge: "GenAI",
    area: "Generative AI",
    subtitle: "LLM Pharmaceutical Knowledge Assistant",
    icon: `<path d="M12 2a4 4 0 0 1 4 4c0 1.1-.45 2.1-1.17 2.83L12 12l-2.83-3.17A4 4 0 0 1 12 2z"/><path d="M12 12l2.83 3.17A4 4 0 1 1 9.17 15.17L12 12z"/><path d="M2 12h4M18 12h4"/>`,
    abstract: `A retrieval-augmented assistant for pharmaceutical questions that grounds <strong>Claude</strong> responses in FDA drug labels and clinical guidelines, with citation attribution and medical-safety guardrails.`,
    highlights: [
      `<strong>Grounded RAG</strong> — semantic chunking of FDA label sections, hybrid search, and MMR re-ranking over ChromaDB.`,
      `<strong>Citations</strong> — answers attribute back to source label sections with cosine-similarity retrieval.`,
      `<strong>Safety</strong> — medical guardrails and refusal handling for unsafe or out-of-scope queries.`,
      `<strong>Streaming UX</strong> — token-streamed responses over WebSocket with rate limiting.`,
    ],
    arch: `query → embed → ChromaDB hybrid search → MMR re-rank\n      → Claude (grounded prompt) → streamed answer + citations`,
    tags: [
      "Claude API",
      "RAG",
      "ChromaDB",
      "LangChain",
      "FastAPI",
      "WebSocket",
      "Streaming",
    ],
  },
  {
    n: 14,
    slug: "14-streaming-pipeline",
    name: "StreamRx",
    accent: "cyan",
    badge: "DE 1",
    area: "Cloud AI/ML",
    subtitle: "Real-time Pharma Event Streaming",
    icon: `<path d="M2 12h4l3-9 6 18 3-9h4"/>`,
    abstract: `A high-throughput streaming pipeline that processes prescription and adverse-event data in real time over Kafka and Kinesis, with sliding-window pharmacovigilance signal detection.`,
    highlights: [
      `<strong>Disproportionality metrics</strong> — Proportional Reporting Ratio (PRR), Reporting Odds Ratio (ROR), and Yates-corrected chi-squared with the standard delta-method PRR confidence interval.`,
      `<strong>Signal thresholds</strong> — MHRA/EMA-style firing rules (PRR ≥ 2, χ² ≥ 3.84, ≥ 3 cases).`,
      `<strong>Stream processing</strong> — Faust agents over Kafka and AWS Kinesis with sliding windows.`,
      `<strong>Sink + infra</strong> — S3 Parquet sink and AWS MSK deployed via CloudFormation.`,
    ],
    arch: `events → Kafka / Kinesis → Faust (sliding window)\n      → PRR · ROR · χ² signal detector → S3 Parquet + alerts`,
    tags: [
      "Kafka",
      "Kinesis",
      "Faust",
      "AWS MSK",
      "S3 Parquet",
      "CloudFormation",
      "DataDog",
    ],
  },
  {
    n: 15,
    slug: "15-ab-testing-platform",
    name: "ModelLab",
    accent: "magenta",
    badge: "DE 3",
    area: "Full Stack / Low Latency",
    subtitle: "A/B Testing Platform for ML Models",
    icon: `<path d="M12 20V10"/><path d="M18 20V4"/><path d="M6 20v-4"/><rect x="3" y="3" width="18" height="18" rx="2"/>`,
    abstract: `A production experimentation platform for ML models with a rigorous statistics engine spanning frequentist and Bayesian analysis, consistent-hash routing, and champion/challenger promotion.`,
    highlights: [
      `<strong>Frequentist core</strong> — Welch's t-test with Satterthwaite df, two-proportion z-test (pooled test SE, unpooled CI), and observed power via the noncentral t.`,
      `<strong>Bayesian analysis</strong> — Beta-Binomial and Normal-Normal conjugate posteriors with P(B&gt;A), expected loss, and HDI.`,
      `<strong>Sequential testing</strong> — O'Brien–Fleming and Pocock alpha-spending, plus Bonferroni and Benjamini–Hochberg corrections.`,
      `<strong>Routing & variance reduction</strong> — consistent-hash traffic assignment, CUPED, SRM detection, and Wilson / Clopper–Pearson intervals.`,
    ],
    arch: `traffic → consistent-hash bucket → variant\n        → metrics → frequentist + Bayesian engine\n        → sequential monitoring → champion/challenger promotion`,
    tags: [
      "Bayesian A/B",
      "FastAPI",
      "SageMaker",
      "Redis",
      "PostgreSQL",
      "CI/CD",
      "Statistical Testing",
    ],
  },
  {
    n: 16,
    slug: "16-pharma-etl-informatica",
    name: "PharmaFlow",
    accent: "amber",
    badge: "DE 4",
    area: "Data / ETL",
    subtitle: "Informatica-Style ETL Framework",
    icon: `<path d="M4 6h16M4 12h16M4 18h16"/><path d="M8 6v12"/><path d="M16 6v12"/>`,
    abstract: `A Python ETL framework that mirrors <strong>Informatica PowerCenter</strong> semantics — Mappings, Sessions, Workflows, and eleven transformation types — applied to pharmaceutical data.`,
    highlights: [
      `<strong>PowerCenter model</strong> — Mappings / Sessions / Workflows with 11 transformations (Expression, Lookup, Router, Aggregator, and more).`,
      `<strong>Pharmacovigilance ETL</strong> — PRR, ROR, and Yates chi-squared with MedDRA term standardization (British spelling) and FAERS age-unit handling.`,
      `<strong>Slowly changing dimensions</strong> — SCD Type 2 PL/SQL procedures.`,
      `<strong>Operations</strong> — Control-M jobs, UNIX automation, and Great Expectations data validation.`,
    ],
    arch: `source → Mapping (Expression · Lookup · Router · Aggregator …)\n       → Session → Workflow → SCD2 target  (Control-M scheduled)`,
    tags: [
      "Informatica Patterns",
      "PL/SQL",
      "ETL",
      "SCD Type 2",
      "Control-M",
      "UNIX",
      "Great Expectations",
    ],
  },
  {
    n: 17,
    slug: "17-wildlife-classifier",
    name: "WildEye",
    accent: "teal",
    badge: "BIO",
    area: "Conservation ML",
    subtitle: "Wildlife Species Classifier",
    icon: `<path d="M20 21v-2a4 4 0 0 0-3-3.87"/><path d="M4 21v-2a4 4 0 0 1 3-3.87"/><circle cx="12" cy="7" r="4"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/><path d="M8 3.13a4 4 0 0 0 0 7.75"/>`,
    abstract: `A conservation toolkit that classifies camera-trap imagery with an efficient edge model and computes ecological biodiversity metrics for ecosystem monitoring.`,
    highlights: [
      `<strong>Biodiversity indices</strong> — Shannon H′, Gini–Simpson (1−D), Pielou evenness, species richness, and relative abundance index, all formula-verified.`,
      `<strong>Ecology analytics</strong> — naive occupancy, observed-vs-expected co-occurrence, and diel-activity classification (diurnal / nocturnal / crepuscular).`,
      `<strong>Edge model</strong> — MobileNetV3 with IR/night-vision handling, exported to ONNX.`,
      `<strong>Serverless</strong> — AWS Lambda classification backed by DynamoDB.`,
    ],
    arch: `camera trap → MobileNetV3 (ONNX) → species\n           → biodiversity engine (Shannon · Simpson · occupancy)`,
    tags: [
      "PyTorch",
      "MobileNetV3",
      "ONNX",
      "Lambda",
      "DynamoDB",
      "Conservation",
    ],
  },
  {
    n: 18,
    slug: "18-recipe-optimizer",
    name: "NutriOptimize",
    accent: "teal",
    badge: "FUN",
    area: "Optimization",
    subtitle: "AI Recipe & Nutrition Optimizer",
    icon: `<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path d="M15 9l-3 3-3-3"/><path d="M12 12v6"/>`,
    abstract: `A multi-objective optimizer that reformulates recipes for better nutrition while preserving taste, blending nutritional science and bioavailability heuristics with constrained optimization.`,
    highlights: [
      `<strong>Constrained optimization</strong> — SLSQP over a USDA-style nutrition database with dietary constraints and per-serving scaling.`,
      `<strong>Bioavailability-aware</strong> — accounts for heme vs non-heme iron, oxalate-bound calcium, and curcumin–piperine synergy.`,
      `<strong>Substitution engine</strong> — ingredient swaps that satisfy AHA/DRI targets (saturated fat, sodium).`,
      `<strong>Meal planning</strong> — extends single-recipe optimization to multi-meal plans via a FastAPI service.`,
    ],
    arch: `recipe → nutrition vectors → SLSQP (maximize nutrition\n       s.t. taste + dietary constraints) → substitutions`,
    tags: [
      "scipy.optimize",
      "Nutritional Science",
      "FastAPI",
      "Constraint Satisfaction",
      "Meal Planning",
    ],
  },
  {
    n: 19,
    slug: "19-pharma-forecaster",
    name: "PharmaForecast",
    accent: "green",
    badge: "DE 2",
    area: "MLOps / Forecasting",
    subtitle: "Time Series Forecasting for Pharma",
    icon: `<path d="M3 3v18h18"/><path d="M7 16l4-8 4 4 4-12"/>`,
    abstract: `An ensemble forecasting system combining ARIMA, Prophet, and ML models for drug-demand prediction, shortage early warning, and adverse-event trend analysis, orchestrated by Airflow.`,
    highlights: [
      `<strong>Ensemble</strong> — series characteristics (trend, seasonality, intermittency) drive inverse-error weighting across ARIMA, Prophet, and ML members.`,
      `<strong>Statistical rigor</strong> — ADF + KPSS stationarity logic, sMAPE backtesting, and Ljung–Box diagnostics.`,
      `<strong>Feature engineering</strong> — lag, rolling, calendar, Fourier-seasonal, and trend features with no look-ahead.`,
      `<strong>Operations</strong> — daily Airflow DAG with accuracy monitoring, drift detection, and retraining triggers.`,
    ],
    arch: `series → classify (trend · seasonality) → {ARIMA, Prophet, ML}\n      → weighted ensemble → backtest (sMAPE) → monitor / retrain`,
    tags: [
      "Prophet",
      "ARIMA",
      "Airflow",
      "CloudFormation",
      "Plotly",
      "Time Series",
      "Ensemble",
    ],
  },
  {
    n: 20,
    slug: "20-pharma-agents",
    name: "PharmaAgents",
    accent: "violet",
    badge: "GenAI",
    area: "Generative AI",
    subtitle: "Multi-Agent AI Research System",
    icon: `<circle cx="6" cy="6" r="3"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="18" r="3"/><path d="M9 6h6M6 9v6M18 9v6M9 18h6"/>`,
    abstract: `A multi-agent system on the <strong>Claude API</strong> where specialized agents collaborate on drug-research queries through task decomposition, tool use, and results synthesis.`,
    highlights: [
      `<strong>Specialized agents</strong> — Literature Review, Drug Safety, Medicinal Chemistry, and Regulatory Intelligence agents coordinate on a shared task.`,
      `<strong>Tool use</strong> — agents call structured tools and a coordinator decomposes queries and synthesizes results.`,
      `<strong>Cheminformatics grounding</strong> — Lipinski, Veber, and Ghose drug-likeness rules implemented to literature values.`,
      `<strong>Service layer</strong> — FastAPI with WebSocket streaming of agent progress.`,
    ],
    arch: `query → coordinator (decompose)\n      → [Literature · Safety · Chemistry · Regulatory] agents (tool use)\n      → synthesis → streamed answer`,
    tags: [
      "Claude API",
      "Multi-Agent",
      "Tool Use",
      "FastAPI",
      "WebSocket",
      "Orchestration",
    ],
  },
  {
    n: 21,
    slug: "21-clinical-trial-eda",
    name: "ClinicalTrialEDA",
    accent: "teal",
    badge: "PHARMA",
    area: "Biostatistics",
    subtitle: "Trial Analytics & Biomarker Discovery",
    icon: `<path d="M3 3v18h18"/><rect x="6" y="11" width="3" height="7"/><rect x="11" y="7" width="3" height="11"/><rect x="16" y="13" width="3" height="5"/>`,
    abstract: `A notebook-driven exploratory analysis of a synthetic <strong>Phase III</strong> anti-inflammatory trial (RX-7281 vs placebo), demonstrating the biostatistics workflow behind a trial readout.`,
    highlights: [
      `<strong>Inflammatory biomarker panel</strong> — CRP (mg/L), IL-6 (pg/mL), TNF-α (pg/mL), and ESR (mm/hr) with correct units throughout.`,
      `<strong>Hypothesis testing</strong> — chi-squared, Mann–Whitney U, response-rate comparison, and a baseline-balance "Table 1".`,
      `<strong>Biomarker discovery</strong> — scikit-learn predictors with SHAP attribution and survival curves via lifelines.`,
      `<strong>Reproducible data</strong> — a seeded synthetic generator (1,200 patients) keeps the whole narrative deterministic.`,
    ],
    arch: `synthetic generator → EDA → statistical testing\n                    → biomarker discovery (SHAP · survival)`,
    tags: ["Jupyter", "Biostatistics", "SciPy", "SHAP", "lifelines", "seaborn"],
    notebooks: [
      {
        file: "notebooks/01_data_generation.html",
        name: "01 · Data Generation",
        desc: "Synthetic RX-7281 Phase III dataset pipeline",
        live: true,
      },
      {
        file: "notebooks/02_exploratory_analysis.html",
        name: "02 · Exploratory Analysis",
        desc: "Demographics, biomarker distributions, correlations",
        live: true,
      },
      {
        file: null,
        name: "03 · Statistical Testing",
        desc: "Chi-squared, Mann–Whitney, response rates",
        live: false,
      },
      {
        file: null,
        name: "04 · Biomarker Discovery",
        desc: "Predictive biomarkers with SHAP + survival",
        live: false,
      },
    ],
  },
];

const ARROW = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg>`;
const BACK = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 12H5M12 19l-7-7 7-7"/></svg>`;
const GH = `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 1.5a10.5 10.5 0 0 0-3.32 20.46c.52.1.71-.23.71-.5v-1.75c-2.92.64-3.54-1.25-3.54-1.25-.48-1.21-1.16-1.53-1.16-1.53-.95-.65.07-.64.07-.64 1.05.07 1.6 1.08 1.6 1.08.94 1.6 2.46 1.14 3.06.87.1-.68.37-1.14.66-1.4-2.33-.27-4.78-1.17-4.78-5.2 0-1.15.41-2.09 1.08-2.83-.11-.27-.47-1.34.1-2.79 0 0 .88-.28 2.88 1.08a9.96 9.96 0 0 1 5.24 0c2-1.36 2.88-1.08 2.88-1.08.57 1.45.21 2.52.1 2.79.67.74 1.08 1.68 1.08 2.83 0 4.04-2.46 4.93-4.8 5.19.38.33.71.97.71 1.96v2.9c0 .28.19.61.72.5A10.5 10.5 0 0 0 12 1.5z"/></svg>`;

const nbIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="4" y="3" width="16" height="18" rx="2"/><path d="M8 3v18"/><path d="M11 8h6M11 12h6M11 16h3"/></svg>`;

function esc(s) {
  return s;
} // content is trusted, authored here

function notebooksSection(p) {
  if (!p.notebooks) return "";
  const items = p.notebooks
    .map((nb) => {
      if (nb.live) {
        return `        <a class="nb-item live" href="${nb.file}" target="_blank" rel="noopener">
          <span class="nb-meta"><span class="nb-icon">${nbIcon}</span><span><span class="nb-name">${nb.name}</span><span class="nb-desc">${nb.desc}</span></span></span>
          <span class="nb-status">View notebook ${ARROW}</span>
        </a>`;
      }
      return `        <div class="nb-item stub">
          <span class="nb-meta"><span class="nb-icon">${nbIcon}</span><span><span class="nb-name">${nb.name}</span><span class="nb-desc">${nb.desc}</span></span></span>
          <span class="nb-status muted">In progress</span>
        </div>`;
    })
    .join("\n");
  return `
      <section class="brief">
        <div class="section-eyebrow"><span class="num">04</span> Notebooks</div>
        <h2>Rendered Jupyter notebooks</h2>
        <p class="abstract" style="margin-bottom:1.4rem">The analysis narrative is captured in Jupyter notebooks, rendered to HTML for direct viewing.</p>
        <div class="nb-list">
${items}
        </div>
      </section>`;
}

function page(p) {
  const nbNumberShift = p.notebooks ? true : false;
  const stackNum = nbNumberShift ? "05" : "04";
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${p.name} — ${p.subtitle} | Danny</title>
  <meta name="description" content="${p.subtitle}. ${p.name}, project ${String(p.n).padStart(2, "0")} of the applied AI/ML portfolio.">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="project.css">
  <style>:root { --accent: var(--${p.accent}); }</style>
</head>
<body>
  <header class="topbar">
    <div class="topbar-inner">
      <a class="back-link" href="../index.html#projects">${BACK} Portfolio</a>
      <span class="topbar-id">PROJECT ${String(p.n).padStart(2, "0")} / 21</span>
    </div>
  </header>

  <main class="wrap">
    <header class="brief-header">
      <div class="eyebrow">Project ${String(p.n).padStart(2, "0")} · ${p.area}</div>
      <div class="brief-headline">
        <div class="brief-icon"><svg viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="1.5">${p.icon}</svg></div>
        <div>
          <span class="badge">${p.badge}</span>
          <h1 class="brief-title">${p.name}</h1>
          <p class="brief-subtitle">${p.subtitle}</p>
        </div>
      </div>
      <div class="meta-grid">
        <div class="meta-cell"><div class="meta-label">Domain</div><div class="meta-value">${p.area}</div></div>
        <div class="meta-cell"><div class="meta-label">Track</div><div class="meta-value">${p.badge}</div></div>
        <div class="meta-cell"><div class="meta-label">Source</div><div class="meta-value"><a href="${REPO}/${p.slug}">View on GitHub ↗</a></div></div>
      </div>
    </header>

    <section class="brief">
      <div class="section-eyebrow"><span class="num">01</span> Abstract</div>
      <h2>${p.subtitle}</h2>
      <p class="abstract">${p.abstract}</p>
    </section>

    <section class="brief">
      <div class="section-eyebrow"><span class="num">02</span> Highlights</div>
      <h2>Methods &amp; contributions</h2>
      <ul class="highlights">
${p.highlights.map((h) => `        <li>${h}</li>`).join("\n")}
      </ul>
    </section>

    <section class="brief">
      <div class="section-eyebrow"><span class="num">03</span> Architecture</div>
      <h2>Data flow</h2>
      <pre class="arch">${p.arch}</pre>
    </section>
${notebooksSection(p)}
    <section class="brief">
      <div class="section-eyebrow"><span class="num">${stackNum}</span> Stack</div>
      <h2>Technologies</h2>
      <div class="tags">
${p.tags.map((t) => `        <span class="tag">${t}</span>`).join("\n")}
      </div>
    </section>

    <footer class="brief-footer">
      <div class="cta-row">
        <a class="cta cta-primary" href="${REPO}/${p.slug}">${GH} Source code</a>
        <a class="cta cta-ghost" href="../index.html#projects">${BACK} All projects</a>
      </div>
      <div class="colophon">Applied AI/ML Portfolio · Project ${String(p.n).padStart(2, "0")} of 21 · ${p.name}</div>
    </footer>
  </main>
</body>
</html>
`;
}

let count = 0;
for (const p of PROJECTS) {
  const out = join(__dirname, "projects", `${p.slug}.html`);
  writeFileSync(out, page(p));
  count++;
}
console.log(`Generated ${count} project pages into projects/`);
