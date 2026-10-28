# Project Plan — Project ARGUS (Modular Monolith)

Persistent source of truth for Project ARGUS (Adversarially Robust Graph-based
Unified Screening for Synthetic Identity KYC Fraud Detection).

---

## Goal

Build an explainable, multi-source synthetic identity fraud detection platform
and modular monolith backend that uncovers coordinated fraud rings across documents,
relational graphs, biometrics, and ongoing behavioral events, validated with
reproducible adversarial robustness benchmarks.

---

## Division of Labor & Priority Queue

- **Agent (Backend & Platform Lead)**:
  1. Solidify modular monolith core, configuration, database schemas, and `ArtifactStore`.
  2. Implement PII pseudonymization (salted HMAC) and indexed relational graph persistence (`applicant_edges`).
  3. Build the zero-broker PostgreSQL lease worker with extensible `HandlerRegistry`.
  4. Implement KYC ingestion endpoints, 1-hop synchronous rules (M0), and 2-hop bounded ego-network extraction.
  5. Build analyst investigation APIs, case queues, and report exports.
  6. Deliver responsive Web Application interface for applicant submission and analyst graph exploration.

- **User / ML Lead (Model Training & Evaluation)**:
  1. Train and tune M1 (Tabular LightGBM) on applicant features.
  2. Train and tune M2 (Graph-enhanced LightGBM) on NetworkX topological features.
  3. Train and tune M3 (Multi-source Calibrated Fusion) with explicit missing-evidence indicators.
  4. Run benchmark evaluations on original Neo4j and BAF datasets.

---

## Current Phase

**Phase 2 — Relational Identity Graph & Bounded Ego-Network Engine**

---

## Status

- [x] Phase 1 — Foundation & Platform Hardening (Zero-Broker, ArtifactStore, PII Security)
- [x] Phase 2 — Relational Identity Graph & Bounded Ego-Network Engine
- [x] Phase 3 — Document Forensics & Evidence Adapters
- [ ] Phase 4 — Multi-Source Fusion & Machine Learning Model Contracts
- [ ] Phase 5 — Continuous Monitoring & Case Investigation API
- [ ] Phase 6 — Adversarial Robustness & Experimental Benchmark Suite
- [ ] Phase 7 — Unified Responsive Web Console & End-to-End Rehearsal

---

## Phase Breakdown 

### Phase 1 — Foundation & Platform Hardening
- [x] **1.1 Platform Core & Storage Engine**: Adapt `backend-platform-benchmark` core into a clean modular monolith; add `ArtifactStore` for secure binary document storage outside PostgreSQL; add PII pseudonymization (salted HMAC-SHA256); remove RabbitMQ/Kafka dependencies in favor of pure PostgreSQL lease queue.
- [x] **1.2 Identity & Role-Based Access**: Integrate secure user authentication, analyst roles, API key management, and session handling.
- [x] **1.3 Extensible Worker & Handler Registry**: Implement open/closed `HandlerRegistry` supporting cooperative timeout checks, bounded execution, and PostgreSQL `FOR UPDATE SKIP LOCKED` claims.

### Phase 2 — Relational Identity Graph & Bounded Ego-Network Engine
- [x] **2.1 Relational Graph Schema**: Implement indexed `applicant_nodes` and `applicant_edges` database tables in PostgreSQL.
- [x] **2.2 Bounded 2-Hop Ego-Network Extractor**: Implement targeted SQL queries that extract only the 2-hop neighborhood of an applicant into a transient NetworkX graph, avoiding global in-memory graph bloat.
- [x] **2.3 Fast Synchronous 1-Hop Rule Engine**: Query 1-hop identifier degree in <15ms during application ingestion; instantly flag reused phones/devices or confirmed fraud links under M0.
- [x] **2.4 Ring & Community Detection**: Implement connected component analysis and clique density algorithms on extracted ego-networks.

### Phase 3 — Document Forensics & Evidence Adapters
- [x] **3.1 Document Ingestion & Artifact Validation**: Multipart document and selfie upload endpoints with MIME validation, magic byte checks, and bounded file sizes (10MB).
- [x] **3.2 OCR & Field Contradiction Detector**: Extract text fields (name, DOB, ID number) and check internal consistency against claimed application data using Levenshtein distance and checksums.
- [x] **3.3 Image Tampering, AI-Signal & Face Consistency Adapters**:
  - **P0 Fast-plane signals (sync path, Pillow only):** image dimension anomaly (AI generator fixed sizes vs. camera photo), EXIF metadata anomaly (missing/AI-tool software tags), JPEG quality factor check (round quality values from generators vs. camera profiles). All return `EvidenceStatus.ANOMALOUS / PRESENT / UNAVAILABLE` — never a bare float.
  - **P1 Slow-plane signals (async worker jobs):** Error Level Analysis (ELA) for paste/edit forgery on JPEG documents; FFT spectral analysis for GAN frequency artifacts; InsightFace face swap detection + cosine embedding similarity between document face crop and selfie; TruFor document-forgery localization model (selected over UniversalFakeDetect — see `FORENSICS_DESIGN.md §2.7`).
  - **P3 Deferred:** C2PA manifest verification (near-absent in Indian KYC fraud tooling; > 99% NO_MANIFEST rate — zero discriminative value for this threat model). UniversalFakeDetect (severe distribution shift to identity document images; uncalibrated on this input domain).
  - Media forensics scores route to analyst evidence annotations only — NOT into M3 fusion — until calibration against labeled documents is complete (see `FORENSICS_DESIGN.md §4`).

### Phase 4 — Multi-Source Fusion & Machine Learning Model Contracts
- [x] **4.1 Baseline M0 (Deterministic Rules)**: Rule-based screening engine covering identifier reuse, field mismatch, and velocity anomalies.
- [ ] **4.2 Model Interfaces & Feature Serializer**: Standardized input/output contracts for M1, M2, and M3; feature extraction script for the user to train models without waiting for backend completion.
- [ ] **4.3 Proposed M3 (Multi-Source Calibrated Fusion)**: Calibrated fusion layer combining graph, document, and behavioral signals with missing-evidence indicators and uncertainty quantification.

### Phase 5 — Continuous Monitoring & Case Investigation API
- [ ] **5.1 Post-KYC Event Ingestion Pipeline**: Ingestion API for post-onboarding lifecycle events (`account_activated`, `device_changed`, `transaction_velocity_spike`).
- [ ] **5.2 Analyst Queue & Case Management API**: Endpoints for priority fraud review queue, case decisions (`APPROVE`, `FLAG_REVIEW`, `CONFIRM_FRAUD`), and audit history.
- [ ] **5.3 Exportable Compliance & Case Reports**: Generate structured JSON and Markdown case dossiers with evidence attribution and linked graph paths.

### Phase 6 — Adversarial Robustness & Experimental Benchmark Suite
- [ ] **6.1 Attack Matrix Simulation**: Controlled experimental harness for clean data, missing links (10%, 20%, 30%), and adversarial camouflage (synthetic links to legitimate profiles).
- [ ] **6.2 Evaluation & Ablation Suite**: Compute PR-AUC, Precision@K, Recall@Fixed-FPR, and Brier calibration scores across M0–M3 under all attack conditions.
- [ ] **6.3 Automated Benchmark CLI & Report Generator**: One-command evaluation script generating reproducible JSON metrics and Markdown comparison tables.

### Phase 7 — Unified Responsive Web Console & End-to-End Rehearsal
- [ ] **7.1 Applicant Self-Service Web Flow**: Mobile-responsive onboarding interface with document upload, webcam selfie capture, and instant screening feedback.
- [ ] **7.2 Fraud Analyst Investigation Console**: Interactive graph visualization (showing connected identity clusters), evidence score breakdown, and case resolution controls.
- [ ] **7.3 End-to-End Scenario Verification**: Rehearsal of the 3 canonical scenarios: Clean applicant, Hidden synthetic identity ring, and Adversarial link camouflage.

---

## Current Task

**Task ID:** 4.2

**Goal:** Implement Model Interfaces & Feature Serializer (`app/modules/fusion/features.py`, `app/modules/fusion/models.py`) providing standardized feature extraction contracts and serialized array vectors for M1 (Tabular LightGBM), M2 (Graph-enhanced LightGBM), and M3 (Calibrated Multi-Source Fusion), with explicit missing-evidence flags.

**Inputs:** `app/core/fusion.py`, `MODELS_TRAINING_SPEC.md`, `app/modules/fusion/schemas.py`.

**Output:** `app/modules/fusion/features.py`, `app/modules/fusion/models.py`, `tests/unit/test_fusion_models.py`.

**Constraints:**
- Standardized tabular feature vectors compatible with Scikit-learn / LightGBM.
- Zero-training pre-calibrated baseline weights for instant testing without mandatory retraining loops.
- Explicit binary missingness flags (`is_doc_missing`, `is_selfie_missing`, `is_graph_isolated`) per `MODELS_TRAINING_SPEC.md §6.2`.

**Verification:** Unit tests verifying feature extraction dimensions, missingness indicators, and inference contracts for M1, M2, and M3.

**Depends on:** 4.1

---

## Next Task

**Task ID:** 4.3 — Proposed M3 (Multi-Source Calibrated Fusion).

---

## Constraints

- Python 3.11+ compatible.
- All core logic type-hinted; no bare `Any` without concrete justification.
- Pure unit tests must run without requiring a live external PostgreSQL/Docker instance (using in-memory SQLite / mock stores).
- Integration tests isolate PostgreSQL-specific functionality.
- Dependencies: FastAPI, Pydantic v2, SQLAlchemy 2.0, LightGBM, NetworkX, Scikit-learn.

---

## Log

### 2026-10-28 — Task 4.1 complete
- Changed: Implemented Baseline M0 (Deterministic Rules Engine) in `app/modules/fusion/rules.py` conforming to `FraudModel` domain protocol in `ARCHITECTURE.md` §2. Defined `RiskTier`, `EvidenceAttribution`, `RiskAssessment`, `ApplicationEvidence`, and `FraudModel` in `app/core/fusion.py`. Created Pydantic response schemas in `app/modules/fusion/schemas.py`. In `DeterministicM0Model`: implemented multi-source rule evaluation covering confirmed fraud links (`CONFIRMED_FRAUD_LINK`), national ID collisions (`IDENTITY_REUSE_COLLISION`), tiered phone reuse (`R1_HIGH_PHONE_REUSE`, `PHONE_REUSE_MODERATE`), tiered device reuse (`R2_HIGH_DEVICE_REUSE`, `DEVICE_REUSE_MODERATE`), tampered Secure QR (`R7_SECURE_QR_TAMPERED`), invalid checksums (`R5_INVALID_ID_CHECKSUM`), OCR contradictions (`R3_NAME_CONTRADICTION`, `R4_DOB_CONTRADICTION`), face mismatch (`DOCUMENT_FACE_MISMATCH`), generative AI image signatures (`IMAGE_AI_SYNTHETIC`, `JPEG_QUALITY_SYNTHETIC`, `IMAGE_ELA_TAMPERED`, `IMAGE_FFT_GRID_ANOMALY`), and velocity bursts (`R6_VELOCITY_BURST`, `VELOCITY_MODERATE`). Added explicit missing signal tracking (`EvidenceStatus.UNAVAILABLE`) with zero score impact. Added benchmark dict helper `evaluate_m0_rules_dict` conforming to `MODELS_TRAINING_SPEC.md §3.3`.
- Files: `app/core/fusion.py`, `app/modules/fusion/schemas.py`, `app/modules/fusion/rules.py`, `app/modules/fusion/__init__.py`, `tests/unit/test_fusion_rules.py`, `PROJECT_PLAN.md`.
- Verified: All 14 tests in `tests/unit/test_fusion_rules.py` and all 96 repository tests passed via `PYTHONPATH=. python3 -m pytest tests/unit -q`; `python3 -m ruff check app tests` returned 0 errors; `py_compile` passed. Average execution time < 0.15ms per evaluation.
- Limitations: M0 is purely deterministic; statistical probability estimation and feature interactions are delegated to M1–M3 in Tasks 4.2–4.3.

### 2026-10-28 — Task 3.3 complete
- Changed: Implemented Image Tampering, AI-Signal & Face Consistency Adapters in `app/modules/forensics/image.py`. Fast-plane synchronous signals: `check_image_dimension_anomaly` (matches against standard generative AI resolutions: 1024x1024, 512x512, etc.), `check_exif_metadata_anomaly` (detects AI software signatures: Stable Diffusion, Midjourney, DALL-E, Firefly, with explicit `EvidenceStatus.UNAVAILABLE` when EXIF is stripped), `check_jpeg_quality_factor_anomaly` (inverts JPEG quantization tables to estimate original software quality factor in < 1ms), and `run_fast_plane_forensics` (< 20ms synchronous execution). Slow-plane asynchronous worker signals: `compute_error_level_analysis` (detects localized recompression differential and spliced copy-paste regions), `compute_fft_spectral_analysis` (2D FFT power spectrum for GAN periodic upsampling checkerboard grid artifacts), `compute_face_embedding_similarity` (cosine biometric similarity with explicit `EvidenceStatus.UNAVAILABLE` on missing crops), and `run_slow_plane_forensics`. Exported all functions in `app/modules/forensics/__init__.py`.
- Files: `app/modules/forensics/image.py`, `app/modules/forensics/__init__.py`, `tests/unit/test_forensics_image.py`, `LESSONS.md`.
- Verified: All 82 unit tests passed via `PYTHONPATH=. python3 -m pytest tests/unit -q`; `python3 -m ruff check app tests` returned 0 errors; `py_compile` passed.
- Limitations: Deep neural TruFor/InsightFace models run optionally in slow-plane worker if GPU/weights are loaded; fallback CPU adapters provide robust baseline coverage without external neural weights.

### 2026-10-28 — Task 3.2 complete
- Changed: Implemented OCR extraction, Verhoeff checksum algorithm, PAN format validation, Levenshtein fuzzy demographic matching, and offline Aadhaar Secure QR RSA verification in `app/modules/forensics/`. `app/core/forensics.py`: defined `ForensicExtractor` protocol, `EvidenceStatus`, `ExtractedDocument`, and `TamperingSignal`. `app/modules/forensics/ocr.py`: `validate_verhoeff`, `compute_verhoeff_check_digit`, `validate_pan_format`, `name_similarity_ratio`, `compare_dobs`, `OCRExtractor`, and `FieldContradictionDetector` returning `R3_NAME_CONTRADICTION`, `R4_DOB_CONTRADICTION`, and `R5_INVALID_ID_CHECKSUM`. `app/modules/forensics/qr.py`: `AadhaarSecureQRVerifier` with offline 2048-bit RSA PKCS#1 v1.5 SHA-256 signature verification over 0xFF-delimited payloads, detecting tampered records (`R7_SECURE_QR_TAMPERED`) without network calls. Integrated forensics pipeline into `kyc.document_forensics` background lease worker handler.
- Files: `app/core/forensics.py`, `app/modules/forensics/schemas.py`, `app/modules/forensics/ocr.py`, `app/modules/forensics/qr.py`, `app/modules/forensics/__init__.py`, `app/modules/kyc/service.py`, `tests/unit/test_forensics_ocr.py`, `LESSONS.md`.
- Verified: All 74 unit tests passed via `PYTHONPATH=. python3 -m pytest tests/unit -q`; `python3 -m ruff check app tests` returned 0 errors; `py_compile` passed.
- Limitations: Optical image tampering (ELA, dimensions, EXIF) will be implemented in Task 3.3.

### 2026-10-28 — Task 3.1 complete
- Changed: Implemented Document Ingestion & Artifact Validation module (`app/modules/kyc/`). Created `KYCApplication` ORM model storing applicant metadata, artifact storage URIs, and screening results; created `KYCApplicationCreate`, `KYCApplicationReceipt`, `KYCApplicationDetail` Pydantic schemas; implemented `KYCApplicationService` validating input formatting, enforcing 10MB bounds, validating magic bytes via `ArtifactStore`, computing salted HMAC-SHA256 digests for all PII, populating `ApplicantNode` and `ApplicantEdge` relational graph tables, executing synchronous M0 rule checks, and enqueuing background forensic extraction jobs; exposed multipart onboarding endpoints in `app/modules/kyc/router.py` (`POST /api/v1/kyc/applications`, `GET /api/v1/kyc/applications/{id}`, `GET /api/v1/kyc/applications`); registered router in `app/main.py`.
- Files: `app/modules/kyc/models.py`, `app/modules/kyc/schemas.py`, `app/modules/kyc/service.py`, `app/modules/kyc/router.py`, `app/modules/kyc/__init__.py`, `app/core/errors.py`, `app/main.py`, `tests/unit/test_kyc_ingestion.py`, `LESSONS.md`.
- Verified: All 64 unit tests passed via `PYTHONPATH=. python3 -m pytest tests/unit -q`; `python3 -m ruff check app tests` returned 0 errors; `py_compile` syntax validation passed.
- Limitations: Document and selfie images stored in artifact directory; deep OCR field extraction and offline Aadhaar Secure QR signature validation will be implemented in Task 3.2.

### 2026-10-28 — Task 2.4 complete
- Changed: Implemented Ring & Community Detection in `app/modules/graph/community.py`. `CommunityDetector` provides connected component analysis, bipartite one-mode projection, clique detection, cycle counting, and fraud contagion metrics for multi-applicant fraud syndicates. Exported classes in `app/modules/graph/__init__.py`. Added comprehensive unit tests in `tests/unit/test_graph_community.py`.
- Files: `app/modules/graph/community.py`, `app/modules/graph/__init__.py`, `tests/unit/test_graph_community.py`, `LESSONS.md`.
- Verified: All 55 unit tests passed via `PYTHONPATH=. python3 -m pytest tests/unit -q`; `python3 -m ruff check app tests` returned 0 errors; `py_compile` passed.
- Limitations: Community detector operates on transient subgraphs; offline batch scheduled job integration will run via Worker in continuous monitoring phase.

### 2026-10-28 — Task 2.3 complete
- Changed: Implemented Fast Synchronous 1-Hop Rule Engine (M0 Baseline). `app/modules/graph/rules.py`: `M0RuleEngine` evaluating 1-hop collision and link degree checks on indexed PostgreSQL/SQLite tables (`ix_applicant_edges_target_edge_type`), returning `RuleEvaluationResult` with structured reason codes (`IDENTITY_REUSE_COLLISION`, `R1_HIGH_PHONE_REUSE`, `PHONE_REUSE_MODERATE`, `R2_HIGH_DEVICE_REUSE`, `DEVICE_REUSE_MODERATE`, `CONFIRMED_FRAUD_LINK`, `R3_NAME_CONTRADICTION`, `R4_DOB_CONTRADICTION`, `R5_INVALID_ID_CHECKSUM`, `R6_VELOCITY_BURST`, `CLEAN`), calibrated risk tiers (`APPROVE`, `REVIEW`, `REJECT`), and execution time telemetry (< 20ms). `app/modules/graph/__init__.py`: exported rule engine classes.
- Files: `app/modules/graph/rules.py`, `app/modules/graph/__init__.py`, `tests/unit/test_graph_rules.py`.
- Verified: All 50 unit tests passed via `PYTHONPATH=. python3 -m pytest tests/unit -q`; `python3 -m ruff check app tests` returned 0 errors; `py_compile` passed.
- Limitations: Rule engine operates on 1-hop collision and local thresholds; multi-component community structure and dense ring clusters will be analyzed in Task 2.4.

### 2026-10-28 — Task 2.2 complete
- Changed: Implemented Bounded 2-Hop Ego-Network Extractor and topological fraud feature engine. `app/core/graph.py`: defined `IdentityGraphEngine` protocol, `GraphFeatureVector`, and `GraphSubnetwork`. `app/modules/graph/engine.py`: `RelationalIdentityGraphEngine` with bounded iterative BFS graph extraction, transient NetworkX graph generation, and calculation of the 8 topological features (`shared_phone_degree`, `shared_device_degree`, `shared_ssn_degree`, `ego_node_count`, `ego_edge_count`, `ego_density`, `max_neighbor_risk`, `cycle_count`). Added Unicode NFKD normalization in `app/core/security.py` to prevent homoglyph hashing evasion.
- Files: `app/core/graph.py`, `app/core/security.py`, `app/modules/graph/engine.py`, `app/modules/graph/__init__.py`, `tests/unit/test_graph_engine.py`, `LESSONS.md`.
- Verified: All 44 unit tests passed via `PYTHONPATH=. python3 -m pytest tests/unit -q`; `python3 -m ruff check app tests` returned 0 errors; `py_compile` passed.
- Limitations: Graph engine currently extracts localized ego-networks on demand; multi-component community detection across the full network will be implemented in Task 2.4.

### 2026-10-28 — Task 2.1 complete
- Changed: Implemented Relational Identity Graph Schema. `app/modules/graph/models.py`: `NodeType` and `EdgeType` StrEnums; `ApplicantNode` ORM model with pseudonymized identifier columns (`identity_hash`, `device_fingerprint_hash`, `phone_hash`, `email_hash`, `ip_address_hash`), screening status, prior risk, and strict `@validates` check rejecting unhashed raw PII; `ApplicantEdge` ORM model with composite indices on `(source_hash, edge_type)` and `(target_hash, edge_type)` for O(log n) lookups, unique constraint on `(source_hash, target_hash, edge_type)`, and raw PII validation. `app/modules/graph/__init__.py`: clean exports.
- Files: `app/modules/graph/models.py`, `app/modules/graph/__init__.py`, `tests/unit/test_graph_schema.py`.
- Verified: All 38 unit tests passed via `PYTHONPATH=. python3 -m pytest tests/unit -q`; `python3 -m ruff check app tests` returned 0 errors; `py_compile` passed.
- Limitations: Graph tables currently store relational node and edge rows; 2-hop extraction and transient NetworkX graph engine will be implemented in Task 2.2.

### 2026-10-28 — Task 1.1.3 complete
- Changed: Implemented Zero-Broker PostgreSQL Lease Worker system. `app/infrastructure/jobs/models.py`: `Job` ORM model (status, lease_token, lease_expires_at, deadline_at, attempt_count, cancel_requested_at, result_json). `app/infrastructure/jobs/registry.py`: `HandlerRegistry` with open/closed registration, `ExecutionContext` with cooperative cancellation checks, `PermanentJobError`, `RetryableJobError`, `JobCancelled`. `app/infrastructure/jobs/service.py`: `enqueue_job`, `claim_one` (FOR UPDATE SKIP LOCKED on PostgreSQL, graceful SQLite fallback), lease expiry recovery, `finish_job` with lease token fencing, `execute_job_lifecycle` with exponential backoff retry. `app/worker/worker.py`: `WorkerService` with bounded thread pool and clean SIGTERM/SIGINT shutdown.
- Files: `app/infrastructure/jobs/models.py`, `app/infrastructure/jobs/registry.py`, `app/infrastructure/jobs/service.py`, `app/worker/worker.py`, `app/worker/__init__.py`, `tests/unit/test_worker_jobs.py`.
- Verified: All 32 unit tests passed via `PYTHONPATH=. python3 -m pytest tests/unit -q`; `python3 -m ruff check app tests` returned 0 errors.
- Limitations: Worker currently uses a fixed poll interval; adaptive backoff on empty queue deferred to future scaling work.

### 2026-10-28 — Task 1.1.2 complete
- Changed: Implemented Identity & RBAC module (`app/modules/identity/`) with Argon2id password hashing, JWT HS256 access tokens, rotating opaque refresh tokens with 30-second replay grace period, admin bootstrap, RBAC dependencies (`current_user`, `require_role`, `require_admin`), and API key management. Integrated identity router and request context middleware into `app/main.py`.
- Files: `app/modules/identity/models.py`, `app/modules/identity/schemas.py`, `app/modules/identity/auth.py`, `app/modules/identity/router.py`, `app/modules/identity/__init__.py`, `app/main.py`, `tests/unit/test_identity.py`, `LESSONS.md`.
- Verified: All 24 unit tests passed via `PYTHONPATH=. python3 -m pytest tests/unit`; `ruff check` passed with 0 errors; `py_compile` passed.
- Limitations: Multi-factor authentication (MFA/TOTP) deferred to future stretch goals per plan.

### 2026-10-28 — Task 1.1.1 complete
- Changed: Implemented core configuration (`app/core/config.py`), cryptographic PII pseudonymizer (`app/core/security.py`), structured error handling (`app/core/errors.py`), database engine with dialect detection (`app/infrastructure/database.py`), and binary artifact storage with magic-byte validation and path jail (`app/infrastructure/storage/artifact_store.py`).
- Files: `app/core/config.py`, `app/core/security.py`, `app/core/errors.py`, `app/infrastructure/database.py`, `app/infrastructure/storage/artifact_store.py`, `app/infrastructure/storage/__init__.py`, `pytest.ini`, `tests/unit/test_security_pii.py`, `tests/unit/test_artifact_store.py`.
- Verified: All 18 unit tests passed via `pytest`; `ruff check` passed with 0 errors; `py_compile` syntax validation passed.
- Limitations: Storage currently uses local disk backend; S3/cloud storage adapter deferred to future scaling if needed.


