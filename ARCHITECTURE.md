# Architecture — Project ARGUS (Modular Monolith)

Stable architectural decisions for Project ARGUS (Adversarially Robust
Graph-based Unified Screening for Synthetic Identity KYC Fraud Detection).
Treat entries here as load-bearing: other modules and tasks assume they hold.

---

## 1. Module Map

```
app/
├── core/                  — Domain interfaces (Protocol/ABC), config, shared types, base errors.
│                            No framework or vendor-specific imports.
├── infrastructure/        — Concrete database sessions, cache/rate limiting, artifact/file storage.
│                            (No external message brokers; pure PostgreSQL lease queue).
├── modules/
│   ├── identity/          — User authentication, Argon2id, JWT sessions, API keys, role-based access.
│   ├── kyc/               — KYC application ingestion, applicant state machine, PII pseudonymization.
│   ├── graph/             — Relational identity graph, bounded 2-hop ego-networks (NetworkX), ring detection.
│   ├── forensics/         — Document OCR extraction, field consistency, tampering detection adapters.
│   ├── fusion/            — Multi-source evidence fusion models (M0 Rules, M1 Tabular, M2 Graph, M3 Fusion).
│   ├── investigation/     — Case management, fraud ring review queues, auditable decision history.
│   └── evaluation/        — Controlled corruption benchmarks (clean, missing links, camouflage), metrics reporting.
├── worker/                — Reliable lease-based job worker with explicit handler registry.
└── main.py                — Application bootstrap, router composition, exception handlers, telemetry.
```

---

## 2. Interfaces / Contracts

### `ForensicExtractor` (Protocol)
- `extract_fields(document_bytes: bytes, mime_type: str) -> ExtractedDocument`
- `assess_tampering(document_bytes: bytes, mime_type: str) -> TamperingSignal`
- Consumed by: `modules/kyc`, `modules/fusion`. Implemented by: `modules/forensics`.
- Adding a new document OCR or forgery model requires implementing this protocol; core KYC flow never imports vision libraries directly.

### `IdentityGraphEngine` (Protocol)
- `add_identity_node(applicant_id: str, attributes: dict[str, str]) -> None`
- `find_linked_clusters(applicant_id: str, depth: int = 2) -> GraphSubnetwork`
- `compute_graph_features(applicant_id: str) -> GraphFeatureVector`
- Consumed by: `modules/fusion`, `modules/investigation`. Implemented by: `modules/graph`.
- Guarantee: Only loads bounded 2-hop ego-networks from PostgreSQL; never holds an unconstrained global graph in memory.

### `FraudModel` (Protocol)
- `predict_risk(features: ApplicationEvidence) -> RiskAssessment`
- Guarantee: Returns calibrated `risk_score` (0.0 to 1.0), `risk_tier` (`APPROVE`, `REVIEW`, `REJECT`), and `evidence_attributions`.
- Consumed by: `modules/kyc`, `modules/investigation`. Implemented by: `modules/fusion` (`M0_Rules`, `M1_Tabular`, `M2_Graph`, `M3_Fusion`).

### `ArtifactStore` (Protocol)
- `save_artifact(applicant_id: str, artifact_type: str, data: bytes, mime_type: str) -> str`
- `load_artifact(storage_uri: str) -> bytes`
- Guarantee: Handles files outside database rows to prevent PostgreSQL table/WAL bloat.

---

## 3. Boundaries & Invariant Rules

1. **PII Minimization & Pseudonymization at Ingestion Boundary:**
   - Raw identity attributes (Aadhaar, SSN, phone numbers, email addresses, national IDs) must be pseudonymized via salted HMAC-SHA256 before being stored in the identity graph or exported in logs.
   - Core domain models reference pseudonymized identifiers (`identity_hash`, `device_fingerprint_hash`).
2. **Dual-Plane Separation:**
   - Synchronous Plane: `POST /api/v1/kyc/applications` executes instant validation, PII pseudonymization, deterministic rules (M0), and immediate 1-hop graph lookup in < 200ms.
   - Asynchronous Plane: Heavy forensic extraction (OCR, image tampering) and 2-hop graph ring recalculations execute in the background worker via PostgreSQL lease claims.
3. **No Dynamic Code or Unbounded Payloads:**
   - Worker handlers are strictly registered via an explicit Handler Registry. No `eval`, dynamic imports, or arbitrary callable execution.
   - Binary files (images/PDFs) must never be stored inside PostgreSQL `JSONB` columns. They are written to `ArtifactStore` and referenced by URI.
4. **Resilience & Missing-Signal Discipline:**
   - Missing evidence (e.g. absent biometric signal or unreadable document) must be explicitly recorded as `EvidenceStatus.UNAVAILABLE`, never silently defaulted to clean/passed.
   - The fusion model must explicitly support missing signals without crashing or returning false confidence.
5. **Biometric Scope Discipline (Critique Finding 1):**
   - Biometrics verify facial consistency between document and selfie, and local device-unlock possession. They do not constitute legal proof of genuine identity and are never used as a single deterministic decision factor.
6. **Bounded Transient Ego-Networks (Critique Finding 2):**
   - Never load a global graph into process memory. Subgraph operations in `NetworkX` must load only bounded 2-hop ego-networks queried from indexed PostgreSQL `applicant_edges` rows, compute metrics, and release memory immediately.
7. **Zero Broker Dependency (Critique Finding 4):**
   - RabbitMQ and Kafka adapters are disabled. PostgreSQL `FOR UPDATE SKIP LOCKED` row locking is the sole, authoritative queue mechanism for all asynchronous jobs, preventing broker timeout cascades and deployment complexity.
8. **Dataset Provenance & Simulation Transparency (Critique Finding 3):**
   - Real benchmark evaluations (Neo4j for graph, BAF for temporal behavior) remain on their separate respective datasets. The end-to-end integrated demo dataset is explicitly documented and labeled as a controlled simulation.

---

## 4. Decisions Log

### 2026-10-28 — Dual-Plane Modular Monolith Architecture
- **Decision:** Adopt a dual-plane architecture: Synchronous fast-path screening API for instant applicant feedback + Asynchronous worker pipeline for deep forensics and graph clustering, sharing a unified PostgreSQL database and domain model within a modular monolith.
- **Reasoning:** In automated KYC systems, applicants require immediate feedback, while fraud analysts require deep cross-applicant correlation and forensic verification.

### 2026-10-28 — Zero Broker Queue Architecture (PostgreSQL Only)
- **Decision:** Remove RabbitMQ and Kafka; execute 100% of background work via PostgreSQL `FOR UPDATE SKIP LOCKED` with per-claim lease tokens.
- **Reasoning:** Eliminates external broker container requirements, connection timeout cascades on API threads, and deployment failures during hackathon evaluation. PostgreSQL easily sustains hundreds of claims/sec.

### 2026-10-28 — Bounded Relational Graph vs Global In-Memory Graph
- **Decision:** Store typed identity edges in indexed PostgreSQL tables (`applicant_edges`). Load only bounded 2-hop ego-networks into transient `NetworkX` graphs on demand.
- **Reasoning:** Avoids memory bloat and GIL serialization under concurrent worker threads.

### 2026-10-28 — Handler Registry Pattern for Worker
- **Decision:** Replace the static class `CPUExecutionBackend` with a modular `HandlerRegistry` that allows modules to register bounded worker tasks (e.g. `kyc.document_forensics`, `graph.cluster_refresh`).
- **Reasoning:** Satisfies Open/Closed principle in `AGENTS.md`. New background tasks can be added without modifying the core worker engine.

### 2026-10-28 — KYC Domain Adaptation & Hard Override Fusion Architecture
- **Decision:** Decouple M3 Fusion from Kaggle e-commerce transaction schemas (`card1`, `TransactionAmt`). In our KYC domain, `m1_prob` is populated from applicant behavioral velocity and `m2_prob` is populated from the relational graph engine's ego-density and neighbor contagion risk. Enforce a two-tier gate:
  1. Deterministic Hard Overrides: If statutory checksum fails (Verhoeff $D_5$ / PAN format), C2PA marks media as AI-generated, or card is non-statutory $\implies$ immediate deterministic `REJECT`.
  2. Probabilistic Fusion: Valid inputs proceed to the calibrated `ArgusMLPFusion` neural network.
- **Reasoning:** Prevents ungrounded mock transaction values from injecting noise, and prevents neural network probability calibration from failing open on mathematically proven document forgery.

