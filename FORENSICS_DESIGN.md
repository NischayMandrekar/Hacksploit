# FORENSICS_DESIGN.md — Media Forensics Evidence Layer for ARGUS

Reference design document. Distilled from brainstorm and critic review sessions
(2026-10-28). Records what to build, in what order, and why — plus what was
explicitly deferred and the reasoning.

This is not an implementation file. Implementation lives in `app/modules/forensics/`.
The relevant plan tasks are 3.1, 3.2, and 3.3 in `PROJECT_PLAN.md`.

---

## 1. Three Distinct Problems (Do Not Conflate)

"AI detection" bundles three separate problems that require separate signals:

| Problem | Question answered | Relevant tool |
|---|---|---|
| **Provenance verification** | Does this file carry a verifiable record of its origin? | C2PA manifest (P3) |
| **AI-generated image detection** | Does the image statistically resemble synthetic generation? | UniversalFakeDetect (deferred) |
| **Document forgery detection** | Was a portrait, name, date, or number altered on a real document? | ELA, EXIF, TruFor (P0–P1) |

A synthetic photo can be harmless. A genuine photo can be used in a forged
document. A manipulated document may carry no AI watermark at all. Treating
these as one problem produces wrong signals.

---

## 2. Signal Inventory (Prioritized)

### P0 — Fast Plane Signals (< 10ms each, no model, no GPU)

These run synchronously during the < 200ms ingestion path. They require only
`Pillow`, which is already a dependency.

#### 2.1 Image Dimension Anomaly
- **What:** Check pixel dimensions against known AI generator output sizes.
- **How:** AI generators output at fixed resolutions (1024×1024, 512×512,
  768×512, 1024×768). Real phone camera photos are 4032×3024 or similar
  high-megapixel non-square dimensions.
- **Signal value:** Strong indicator of bulk-automated synthetic submission.
- **Robustness:** Defeated by cropping/resizing. Catches unsophisticated
  attackers; still valid as an additive signal.
- **Implementation:** `PIL.Image.open(data).size` — one line.

#### 2.2 EXIF Metadata Anomaly
- **What:** Check for missing or implausible camera EXIF data.
- **How:** Real camera photos carry Make, Model, DateTimeOriginal, and usually
  FocalLength, ISOSpeedRatings. AI-generated images have no EXIF or only
  minimal software-generated tags. Presence of AI-tool Software tags
  (e.g. "Stable Diffusion") is near-proof of synthetic origin.
- **Signal value:** Medium-high. Stripping EXIF is trivial, so absence alone
  is weak; positive software tag match is strong.
- **Implementation:** `PIL.Image._getexif()` or `piexif.load()`.

#### 2.3 JPEG Quality Factor Anomaly
- **What:** Check whether JPEG quantization tables indicate a suspiciously
  round quality factor.
- **How:** Optical cameras use proprietary compression profiles — never exactly
  quality=95 or quality=90. AI generators and image editors typically do.
  Quantization tables are readable from the JPEG header without decoding.
- **Signal value:** Low-medium on its own; useful as a tie-breaker.
- **Implementation:** `PIL.Image.quantization` dictionary compared against
  standard quality-level lookup tables.

**All three fast-plane signals must be recorded as EvidenceStatus.PRESENT /
EvidenceStatus.UNAVAILABLE / EvidenceStatus.ANOMALOUS — never as a bare float.
Missing EXIF is not the same as anomalous EXIF; the distinction matters for
the analyst.**

---

### P1 — Slow Plane Signals (async worker, CPU-feasible)

These run as registered HandlerRegistry jobs in the asynchronous plane.

#### 2.4 Error Level Analysis (ELA)
- **What:** JPEG recompression differential. Altered regions compress at a
  different error level than surrounding original content.
- **How:** Save at known quality (e.g. 95); compute absolute pixel diff against
  original; normalize; return heatmap + scalar ela_max_region_score.
- **Signal value:** High for copy-paste forgeries on JPEG documents. Degraded
  by multiple recompressions.
- **Implementation:** Pillow only. No neural model.

#### 2.5 FFT Spectral Analysis (GAN Frequency Fingerprint)
- **What:** GAN upsampling leaves characteristic grid artifacts in the
  frequency domain.
- **How:** 2D FFT on image → radial power spectrum → check for peaks at
  spatial frequencies associated with transposed-convolution upsampling.
- **Signal value:** Useful for GAN-generated faces. Less relevant for
  diffusion models which use a different architecture.
- **Implementation:** NumPy FFT. No model.

#### 2.6 InsightFace — Face Swap Detection + Embedding
- **What:** Combined face detection, face-swap probability, and facial
  embedding in one library. Replaces the need for a separate cosine
  similarity adapter planned in Task 3.3.
- **How:** Detect + align face from document crop and from selfie; compute
  embedding cosine distance; read swap-probability score.
- **Signal value:** High. Face swap on KYC selfie is a real, common attack.
- **Robustness:** High-quality diffusion inpainting can evade this. Explicit
  EvidenceStatus.UNAVAILABLE must be recorded if no face is detected.
- **Implementation:** `pip install insightface`. ~200–400ms CPU.
  Run in async plane only — never in the synchronous path.

#### 2.7 TruFor — Document Forgery Localization
- **What:** Image-forensics model producing manipulation detection and
  region-level localization heatmaps. Designed for document images.
- **Why TruFor over UniversalFakeDetect:** TruFor is trained on image forgery
  and produces localization. UniversalFakeDetect is trained on natural-scene
  AI generation with undocumented distribution shift to identity documents.
  TruFor was the baseline 20 of 26 DeepID challenge teams failed to beat.
- **Signal value:** High for Photoshop-style document manipulation.
- **Implementation:** grip-unina/TruFor (GitHub). PyTorch. CPU-feasible for
  demo. Must be validated on synthetic test documents before analyst display.

---

### P2 — Slow Plane, Lower Priority

#### 2.8 EXIF Software Tag Positive Match
- **What:** If EXIF Software tag contains "Stable Diffusion", "Midjourney",
  "DALL-E", "Adobe Firefly", or similar — flag immediately as strong signal.
- **Why separate from 2.2:** Absence of EXIF is weak; presence of an AI-tool
  tag is strong. Logically distinct.
- **Implementation:** String match against a maintained list. Trivial.

---

### P3 — Deferred (Do Not Implement Until Explicitly Needed)

#### 2.9 C2PA Manifest Verification
- **Deferred because:** The C2PA ecosystem is dominated by Adobe/OpenAI/Google
  tools. Indian KYC fraud overwhelmingly uses Photoshop, local deepfake tools,
  and mobile editing apps — none of which embed C2PA manifests. A manifest
  will be absent for > 99% of both legitimate and fraudulent submissions,
  producing a near-constant NO_MANIFEST result with zero discriminative value.
- **When to revisit:** If C2PA adoption reaches Indian mobile camera OEMs or
  if UIDAI mandates provenance metadata.

#### 2.10 UniversalFakeDetect / CLIP-based AI Image Classifier
- **Deferred because:** Trained on natural-scene photographic images. Identity
  documents have a severe distribution shift from the training set. Running on
  document images produces an uncalibrated score whose accuracy on this
  specific input distribution is unknown. Will corrupt the M3 feature space.
- **When to revisit:** When a labeled dataset of real vs. AI-generated Indian
  identity document images is available for validation.

---

## 3. How Signals Feed Into ARGUS

### Fast Plane (synchronous, < 200ms ingestion response)
```
Dimensions     → EvidenceStatus.ANOMALOUS / PRESENT
EXIF presence  → EvidenceStatus.ANOMALOUS / PRESENT
EXIF software  → EvidenceStatus.ANOMALOUS / PRESENT
JPEG quality   → EvidenceStatus.ANOMALOUS / PRESENT
```
Returned as part of the preliminary RiskAssessment from the synchronous path.
These are additive signals to M0, not gates. They do not auto-reject.

### Slow Plane (async worker jobs, registered in HandlerRegistry)
```
job type: forensics.ela_analysis      → ELA heatmap score
job type: forensics.fft_spectrum      → GAN artifact probability
job type: forensics.face_consistency  → InsightFace embedding + swap score
job type: forensics.trufor_integrity  → TruFor localization + integrity score
```
Results stored as ForensicEvidence rows referencing the application ID.
Displayed to analysts as annotated evidence — not as a combined fraud
probability — until M3 fusion has been calibrated against these features
with labeled data.

### M3 Fusion (Phase 4)
Media forensics features enter the M3 feature vector only after:
1. Synthetic test document validation confirms signal quality.
2. Calibration on the validation split is complete.
3. Each signal's PR-AUC contribution is documented.

---

## 4. Calibration and Evaluation Requirements

Before any signal influences automated decisions:

| Signal | Validation requirement |
|---|---|
| ELA score | Test on synthetic docs with known paste regions; confirm FPR on clean docs |
| TruFor integrity | Test on MIDV-500 or FantasyID; document precision/recall |
| InsightFace swap | Test on real vs. swapped face pairs; confirm cosine threshold |
| FFT spectrum | Test on ProGAN/StyleGAN faces vs. real photos; document generalization |

Do not combine uncalibrated scores into a single fraud probability.
Show them as independent evidence signals with provenance and limitations
visible to the analyst.

---

## 5. Test Document Sources (No Real PII)

| Source | Contents | License |
|---|---|---|
| MIDV-500 (arXiv:1807.05786) | 500 video clips, 17 document types | Academic (free) |
| FantasyID (DeepID ICCV 2025) | 362 synthetic IDs, 13 country templates | Research (free) |
| Synthdog (Clova AI) | Programmatic synthetic document generator | Apache 2.0 |
| scripts/generate_test_documents.py | Faker + Pillow synthetic Aadhaar-style cards | Build in-house |

Real identity documents of real people must never be used as test fixtures.
See AGENTS.md §15 Security.

---

## 6. Decisions Log

### 2026-10-28 — C2PA deprioritized to P3
C2PA is near-absent in Indian KYC fraud tooling. Absent for > 99% of
submissions. Zero discriminative value for this specific threat model.

### 2026-10-28 — UniversalFakeDetect deferred
Trained on natural-scene images; distribution shift to identity documents is
severe and unvalidated. Revisit only with labeled document-specific data.

### 2026-10-28 — TruFor selected over general AI classifiers for documents
TruFor is designed for image forgery localization. Outperformed 20 of 26 teams
on the DeepID identity-document manipulation challenge. Document-specific.

### 2026-10-28 — Three fast-plane signals added to Task 3.3 scope
Dimension check, EXIF metadata check, and JPEG quality factor check are
near-free (< 10ms, Pillow only, no model). Added to synchronous ingestion
path scope in Task 3.3. See PROJECT_PLAN.md.
