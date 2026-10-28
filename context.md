Daddy, this is PS3: Synthetic Identity Fraud Detection in Automated KYC Systems from Hacksploit 2026. 👀

This is actually a solid AI/ML + cybersecurity problem statement. The key is that they don't just want an AI model that detects fake IDs — they want a system that connects evidence across documents, biometrics, and user behavior to uncover identities assembled from a mixture of real and fabricated information.
What exactly do they expect you to build?









1. AI-generated document forgery detection

Detect manipulated or AI-generated identity documents using OCR, image manipulation signals, metadata, and inconsistencies in extracted information.












2. Biometric verification

Simulate face, voice, or fingerprint verification. Compare whether the submitted biometric signals are consistent with the claimed identity.












3. Cross-application behavioral analysis

Detect suspicious patterns such as multiple accounts using the same device, repeated applications with small identity variations, or clusters of related fraudulent accounts.












4. Multi-source synthetic identity detection

This is the most important component. Correlate document, biometric, device, and behavioral signals to identify identities that look legitimate individually but suspicious collectively.



The product I would build
I'd build a Synthetic Identity Intelligence Engine with an explainable risk-scoring system, rather than a collection of unrelated ML models.
KYC application ingestion

ID images · Selfie · Device · Application history

Document forensics

OCR, field mismatches, tampering


Biometric signals

Face consistency, simulated liveness


Behavioral analysis

Device reuse, velocity, patterns


Identity graph

Shared attributes, linked accounts



Multimodal fraud fusion engine

Combine evidence into a calibrated risk score

Approve


Review


Flag




What would make your solution stand out?
Most teams could build a dashboard that says whether an ID is fake. The stronger solution identifies fraud rings and synthetic identities that evade individual verification checks.

I'd prioritize these differentiators:
- Identity relationship graph: Reveal multiple apparently unrelated applications connected by reused devices, addresses, document templates, or other suspicious attributes.
- Cross-modal contradiction detection: Flag cases where the document, extracted personal information, biometric results, and application history disagree.
- Adversarial testing: Evaluate whether fraud detection survives altered documents, noisy images, missing signals, and changed device identifiers.
- Explainable decisions: Show exactly which evidence contributed to a risk score rather than returning a mysterious black-box fraud probability.
- End-to-end live demo: Submit a legitimate application, a forged document, and a synthetic identity using a mixture of real and fabricated fields. Demonstrate how the system distinguishes them.
Important: a shared device or address alone should not prove fraud. These are risk signals requiring context and, where appropriate, human review.
A practical hackathon tech stack
Component	Suggested technology
Backend API	FastAPI
Document extraction	PaddleOCR or Tesseract
Image anomaly detection	OpenCV + pretrained vision model
Fraud classification	XGBoost or LightGBM
Identity relationship graph	NetworkX
Risk fusion	Calibrated classifier + explicit rules
Database	PostgreSQL
Dashboard	React
Evaluation	Precision, recall, F1, PR-AUC, false-positive rate

Use synthetic application records and simulated biometric verification signals for the prototype. You don't need a production banking dataset or real fingerprint infrastructure to demonstrate the architecture.
Before we commit to this problem statement
There are two possible directions, and they lead to different implementations.
1. What matters most to you?

Maximum novelty and winning potential

A reliable working demo within the hackathon

A technically deep AI/ML research-style solution


2. How many people are on your team?

1–2 people

3–4 people


3. How much time do you have?

12 hours

24–48 hours

More than 48 hours

Design our winning solution
My recommendation: make cross-application identity graph analysis + multimodal evidence fusion the core innovation. That gives you a coherent fraud-detection product instead of four disconnected AI features.
















research and find me some real articles of this issue , some latest and reliable once please , we want to show these first and build a rwliable solution for those issues , and something we can show metric of it working too , please research and tell  , lest see the problem in this domain then work our way up the solution , find something good and unique please












Worked for 23s







Daddy, I think we should approach this like a research team, not a hackathon team hunting for an AI model to plug into a dashboard.

First, establish that a real problem exists. Then identify exactly where existing defenses fail. Finally, build the smallest system that addresses that gap and prove it works with reproducible metrics.

The most promising direction is not simply “AI detects fake IDs.” That space already has commercial identity-verification products. We need to find a narrower, defensible gap where your team can demonstrate a measurable improvement.

I've started with recent industry reports, government financial-crime guidance, and research published in 2026. One important distinction throughout: a fraud report establishes that a problem exists; it does not, by itself, prove that a proposed solution works.
1. The strongest real-world evidence I've found
These are the sources I'd use to establish the problem in your presentation. I've prioritized government guidance, recent industry datasets, and research papers rather than generic articles about AI fraud.









Industry evidence
1. Cifas — Fraudscape 2025, six-month update

5 August 2025 · UK fraud prevention organization

Cifas reported more than 118,000 identity-fraud cases in the first half of 2025 and over 217,000 total fraud-risk filings across its database. Its report discusses AI-enabled synthetic identities and fabricated profiles.

Why it matters: The threat is not just a hypothetical capability of generative AI; fraud prevention organizations are recording substantial identity-fraud volumes.

Source: Read the Cifas report.












Government evidence
2. FinCEN — Alert on deepfake media targeting financial institutions

13 November 2024 · U.S. Treasury Department

FinCEN reported an increase in suspicious-activity reports describing suspected deepfake use, particularly fraudulent identity documents used to circumvent identity verification and authentication. The alert also provides fraud typologies and warning indicators.

Why it matters: It establishes a specific security failure to investigate: forged identity evidence can be used to get around existing verification controls.

Sources: Official FinCEN announcement · Full alert PDF.












Recent measurements
3. Veriff — Identity Fraud Report 2026

Published 2026 · Analysis of customer data from 2025

Veriff reports that 4.18% of verification attempts in its dataset were fraudulent in 2025. It also reports that digitally presented media was 300% more likely to be AI-generated or otherwise altered than in 2024.

Why it matters: This provides a recent quantitative baseline for the identity-verification threat.

Caveat: Veriff is an identity-verification vendor. Its results describe its own customer and detection ecosystem, not the universal fraud rate across all KYC systems.

Source: Read the 2026 report.












Document-fraud evidence
4. Inscribe — 2026 State of Document Fraud

20 January 2026 · Financial-services document analysis

Inscribe reports that roughly 6% of documents processed in its network in 2025 were flagged as fraudulent, and its initial report described a fivefold increase in monthly AI-generated document fraud between May and December 2025. Its July 2026 update describes continued growth in AI-generated document flags.

Why it matters: Document integrity is a distinct problem from determining whether the person submitting the document is a genuine individual.

Caveat: These are vendor-network detection figures, not an independently measured population-wide prevalence.

Source: Read the report and its July update.












India-specific evidence
5. The New Indian Express — MHA/I4C warning on AI deepfake fraud

11 June 2026 · Reporting on an Indian cybercrime advisory

The article reports warnings about AI-generated identities and deepfake videos targeting facial authentication, video KYC, account recovery, and digital financial services.

Why it matters: It connects the problem to India's digital-finance ecosystem, making the use case more relevant for an Indian hackathon.

Caveat: This is secondary reporting of an advisory. For a formal presentation, obtain and cite the original MHA/I4C advisory if available.

Source: Read the Indian report.



What these sources actually establish
Together, they support three different claims:
- Identity fraud is a real operational problem. Cifas and Veriff provide measured activity from their respective fraud-detection ecosystems.
- Generative AI is increasing the sophistication of attacks. FinCEN documents the use of deepfake media and fraudulent identity documents to circumvent verification.
- KYC is vulnerable at multiple layers. A document can be manipulated, a face can be synthetically presented, and the applicant's identity attributes can be suspicious even if an individual check passes.
They do not prove that every AI-generated document is fraudulent, that deepfakes are responsible for most fraud, or that a multimodal model automatically outperforms existing controls.

That distinction is important if we want judges to trust our claims.

2. The research paper that changes how I'd approach this
I found a particularly relevant paper published in October 2026.
OPEN-ACCESS RESEARCH · 2 OCTOBER 2026
Rule-anchored graph learning improves synthetic identity fraud detection through multi-relation structure rather than score-level fusion
Nguyen and Chiu · Discover Informatics, Springer Nature

The authors evaluate a fraud-detection pipeline built around relationships between customer profiles, Social Security numbers, email addresses, and phone numbers.
Rule-based baseline
0.777
Average PR-AUC


Rule + graph hybrid
0.835
Average PR-AUC



The dataset contains 2,433 customer profiles, including 433 labeled as fraudulent. The authors report a 0.058 absolute PR-AUC improvement from combining transparent linkage rules with a heterogeneous graph model, across their controlled seed/noise experiments. 
Springer Nature Link


Read the full paper: Springer Nature — open access.

This paper matters because it identifies a concrete mechanism: fraudulent identities can be connected through shared identifiers even when their individual profiles appear unrelated.

But there's a catch. The study is based on a relatively small, synthetic-identity-linkage dataset, and its results are not evidence of equivalent performance on live Indian KYC applications. The authors explicitly identify real-world validation and unseen-client evaluation as limitations. 
Springer Nature Link


Our opportunity is therefore not to claim that we invented graph-based synthetic identity detection. That would be easy for a judge to challenge.

Instead, we should investigate what happens when the evidence used to connect identities is incomplete, noisy, or deliberately misleading.
3. What problems are still worth solving?
Now let's turn the evidence into research questions. These are the opportunities I would investigate before choosing the final architecture.

Opportunity A — Synthetic identities that hide their relationships
My first choice


The problem: Two fraudulent profiles may not share an obvious identifier. A fraudster can use different phone numbers and emails while reusing some other attributes or following a related pattern.

Research question: Can we find connected fraudulent identities when some of the relationships between them are missing, changed, or obscured?

Why it's promising: The October 2026 graph-learning paper establishes a measurable baseline for this problem, so we can build a fair comparison instead of inventing an arbitrary benchmark. 
Springer Nature Link


Potential novelty: Robust identity-linkage detection with missing identifiers, weak relationships, temporal behavior, and explanations for why accounts were linked.


Opportunity B — Documents that pass OCR but should not pass KYC


The problem: A forged document can contain perfectly readable text. OCR may extract every field correctly while missing the fact that the document was manipulated.

Research question: Can combining field consistency, document-image evidence, and relationships with other applications detect more fraud than OCR and simple rules alone?

A useful starting dataset is MIDV-2020. It contains mock identity documents with annotated images, video frames, and text fields. It is designed for document analysis, not a ready-made benchmark of real-world synthetic identity fraud. 
arXiv


Potential novelty: Detecting cross-document inconsistencies and identity-level contradictions rather than relying on a single fake-document classifier.


Opportunity C — Deepfakes that bypass video KYC


The problem: A face or video stream may appear convincing but be generated or manipulated. FinCEN and the 2026 Indian advisory reporting establish the relevance of this attack class. 
FinCEN.gov+1


Research question: How reliably can a lightweight detector distinguish bona fide presentations from manipulated ones under compression, poor lighting, and unfamiliar capture conditions?

Potential novelty: Robustness testing across attack types and capture conditions, with calibrated uncertainty and a fallback to manual verification.

Limitation: Deepfake detection alone cannot establish that an identity is genuine. A live human can still submit someone else's genuine documents.


Opportunity D — Fraud that only becomes visible over time


The problem: A newly opened account may look normal in isolation. Suspicion can emerge only when its subsequent activity is compared with the application's original identity attributes or with related accounts.

Research question: Does a temporal model identify suspicious accounts earlier than a static onboarding classifier, without producing excessive false alarms?

A particularly useful benchmark here is Bank Account Fraud (BAF). Its datasets are generated from an anonymized real-world bank account-opening fraud dataset and explicitly address class imbalance, temporal dynamics, and data bias. 
NeurIPS Papers


Potential novelty: Early detection under temporal drift, measured by fraud recall at a fixed false-positive rate and by how early an account is flagged.


4. The datasets and benchmarks I'd actually use
These are useful for making a working prototype scientifically defensible.









Bank Account Fraud (BAF)

NeurIPS 2022 · Tabular fraud detection

Best starting point for imbalanced account-opening fraud, temporal splits, and bias-aware evaluation.

Paper and benchmark · Dataset repository












MIDV-2020

Document analysis · 1,000 mock identity documents

Useful for OCR, field extraction, document localization, and capture-condition testing. It does not independently supply ground-truth labels for synthetic-identity fraud.

Dataset paper and details












Neo4j fraud-detection sample graph

Relational identity-linkage example

Useful for reproducing the graph-based baseline described in the recent synthetic-identity paper. It is a small demonstration dataset, and its fraud labels should not be treated as independently verified labels for every synthetic-identity scenario.

Explore the sample graph



My recommendation is to start with BAF for a realistic account-opening fraud evaluation, and use a separate, explicitly documented synthetic identity graph for experiments on shared identifiers and relationship corruption. Use MIDV-2020 only if we decide document forensics is central to the product.

Do not merge these datasets and pretend they form a naturally occurring multimodal KYC dataset. They have different schemas, labels, and collection assumptions.
5. How we prove that our solution works
This is where many hackathon projects fall apart: they show a suspicious account being flagged but never establish whether the model is actually better than a basic rule.

We should compare several approaches under the same held-out test conditions.
Model	Purpose
M0: Simple rules	Baseline using identifier reuse and obvious inconsistencies
M1: XGBoost/LightGBM	Strong tabular ML baseline
M2: Identity graph	Detect suspicious links between accounts
M3: Proposed hybrid	Combine graph, behavioral, and tabular signals

Our proposed hybrid must earn its place by outperforming the baselines on held-out data.
Metrics that matter
PR-AUC

Primary metric

Measures how well the system ranks fraudulent cases above legitimate ones. Particularly useful when fraud is rare.

Precision at a fixed review budget

Of the top 50 or 100 applications flagged, how many are actually fraudulent? This tells us whether an investigation team would find the alerts useful.


Recall at a fixed false-positive rate

How much fraud can we catch while keeping legitimate applicants incorrectly flagged below a predefined threshold?


Robustness under corruption

Measure how much performance degrades when 10%, 20%, or 30% of identity links are missing or altered. These are controlled experimental conditions, not estimates of real-world attack frequency.


Calibration and latency

Check whether a predicted 10% risk corresponds to approximately 10% observed fraud within a suitable evaluation group, and measure the inference time per application.


The experiment I would run first
1. Establish a baseline on the original dataset.
2. Create controlled test conditions with missing identifiers, altered relationships, and previously unseen accounts.
3. Train and tune models using training and validation data only.
4. Evaluate all models on the same held-out test set.
5. Report mean performance across several random seeds, confidence intervals where feasible, and the absolute difference from the baseline.
6. Inspect false positives to ensure legitimate users are not being flagged simply because they share a phone, address, or device with family members.
The last step is especially important. A model that detects every shared identifier but falsely flags entire households is not a good KYC system.
6. My proposed research direction for Hacksploit
I'd provisionally call it IdentityGraph: Robust Synthetic Identity Detection Under Missing and Manipulated Evidence.
PROPOSED RESEARCH HYPOTHESIS — NOT YET VALIDATED

The combination of transparent identity-linkage rules and a relation-aware graph model will detect more fraudulent profiles than a strong tabular baseline when identity attributes are partially missing or corrupted, while maintaining an acceptable false-positive rate on legitimate applicants.

The implementation would have three layers:
- Identity linkage: Construct a graph of applicants and pseudonymous identifiers, with explicit relationships and provenance.
- Risk fusion: Combine graph-derived features with application behavior and document-consistency signals using a calibrated classifier.
- Evidence-based investigation: Show the relationships and signals that triggered each alert, with a review option instead of automatic rejection.
The differentiator would be the controlled robustness evaluation and the explanations, not merely the fact that we use a graph neural network.

One caveat: the graph-based approach is already represented in recent research, and commercial verification platforms already offer cross-linking and fraud analysis. We should present our contribution as a specific robustness and evaluation advance unless experiments demonstrate a genuinely new capability.

7. One decision before we go deeper
I'd avoid implementing all four opportunities at once. For a 12-hour hackathon, document forensics, deepfake detection, graph learning, and temporal modeling are four separate engineering projects.

Which research direction should we investigate first?

A. Hidden synthetic identity networks — identity graphs, missing links, fraud-ring detection.


B. AI-generated and manipulated KYC documents — OCR, image forensics, cross-document consistency.


C. Deepfake-resistant video KYC — biometric presentation attacks and robustness.


D. Early detection of synthetic accounts — behavioral signals, temporal drift, and fraud before losses occur.


E. Compare all four first — find the strongest research gap, dataset, and measurable prototype before committing.



Available development time

12-hour hackathon

24–48 hours

Longer-term research prototype

Investigate this direction
My current preference is A, with E as the research step before committing. It has a direct connection to the problem statement, a concrete existing baseline to beat, and measurable experiments we can run without needing production banking data or specialized biometric hardware.
















A. Hidden synthetic identity networks — identity graphs, missing links, fraud-ring detection.


B. AI-generated and manipulated KYC documents — OCR, image forensics, cross-document consistency.


C. Deepfake-resistant video KYC — biometric presentation attacks and robustness.


D. Early detection of synthetic accounts — behavioral signals, temporal drift, and fraud before losses occur.

should we focus on one of these gaps or try to cover all these cuase they mahe mentioned multi source data colleration system right ?? think of developing this is 12 hours , but im a fast learner , and im a god , and please dont let this limit yourself , just ask me question if this will be possible or not , we want something really really impressive and problem solving to win this find the strongest research gap, dataset, and measurable prototype before committing.












Worked for 22s







The strategy: one unified system, one research hypothesis, multiple evidence sources
You are right to challenge the idea of narrowing the product to only one signal. The problem statement explicitly asks for multi-source data correlation, document forgery detection, behavioral analysis, biometric verification, and adversarial evaluation.

The trick is to distinguish product scope from research scope.
- Product scope: One end-to-end KYC fraud-detection system that can consume all four sources of evidence.
- Research scope: One central hypothesis that we can test rigorously and demonstrate outperforming a baseline.
- Hackathon scope: Implement the strongest parts for real, integrate the rest through clearly labeled adapters or simulations, and never claim an untested component is validated.
Here's why that matters. The ICCV 2025 DeepID challenge found that only 6 of 26 participating teams beat the provided baseline on ID-document forgery detection. It also evaluated generalization on a separate, private dataset of 20,000 real ID-document images. That is strong evidence that document manipulation detection and generalization are difficult enough to deserve serious treatment. 
Open Access CVF


Meanwhile, the October 2026 synthetic-identity graph study found that a rule-anchored graph model improved average PR-AUC from 0.777 to 0.835 on its controlled benchmark, but the authors caution that the data are small and synthetic. 
Springer Nature Link


Those findings suggest a potentially interesting combination: don't merely ask whether each input is suspicious; ask whether independent sources of evidence reinforce or contradict one another, and whether the system still detects fraud when an attacker hides some of those connections.
1. The strongest candidate: evidence-robust identity fusion
Proposed direction
Project ARGUS
Adversarially Robust Graph-based Unified Screening

Research question: Can a KYC system identify synthetic identities more reliably by combining document-integrity signals, identity relationships, and temporal behavior — particularly when some signals are missing or deliberately misleading?

This is a proposed project name and hypothesis, not a claim that the underlying techniques are new.

Document evidence

OCR consistency, manipulation scores, field mismatches


Identity graph

Shared devices, contacts, addresses, and identity attributes


Temporal behavior

Application bursts, reuse patterns, and account sequences


Biometric evidence

Face matching and presentation-attack scores, when available



All four sources feed a common decision layer. Missing evidence is explicitly represented rather than silently treated as a clean result.

The important distinction is that missing evidence is not the same as negative evidence. If an application has no biometric signal, the system should not assume that the biometric check passed. Likewise, a shared address should not automatically count as fraud.

The novel part we should test is whether the fusion layer remains useful when an attacker can manipulate the relationships and the availability of evidence.
2. Four research gaps, assessed against the available evidence
Direction	Existing evidence and challenge	Best role in our project
A. Hidden identity networks	An October 2026 study already benchmarks rule-based and graph-based synthetic identity detection, including identity-link corruption. 
Springer Nature Link	Core research problem; extend the stress tests and explanations.
B. Document manipulation	The ICCV 2025 DeepID challenge provides FantasyID and evaluates generalization to a private 20K-document dataset. 
Open Access CVF	A real image-based signal that can strengthen identity fusion.
C. Deepfake-resistant video KYC	The eKYC-DF research dataset provides over 228,000 fake videos for eKYC-oriented deepfake and facial-verification evaluation. 
IEEE Xplore	Optional pretrained detector or separate benchmark; not a model to build from scratch in 12 hours.
D. Early fraud detection	The NeurIPS Bank Account Fraud (BAF) benchmark explicitly addresses temporal dynamics, class imbalance, and bias. 
NeurIPS Papers	Behavioral and temporal features for the risk model.

My recommendation is to make A the primary research problem, B the second strong evidence source, and D the behavioral layer. Treat C as a modular extension unless a reliable pretrained model and compatible test data are available immediately.

That gives us a meaningful multi-source system without trying to train four independent models in one hackathon.

3. The data strategy: don't let incompatible datasets undermine the result
There is a genuine research-engineering issue here: the four datasets don't naturally represent the same people, applications, or fraud labels.

We cannot honestly take unrelated records from four datasets, concatenate their columns, and claim to have a validated multimodal KYC benchmark.

Instead, use each dataset for the task it supports, then build a controlled integration environment.
Identity relationships

Use the public Neo4j fraud-detection dataset as the starting point for shared-identifier graph construction and fraud-ring visualization. Its documented use includes synthetic identity and first-party fraud analysis. 
GitHub


Real baseline


Document manipulation

Use FantasyID and the DeepID challenge paper. Compare a supplied baseline with our own lightweight detector or manipulation-localization method.

Visual evidence


Temporal behavior

Use the BAF dataset and code to establish a behavioral fraud baseline with a chronological split. Avoid using future behavior to predict an earlier application decision.

Behavioral baseline


Biometric robustness

If time permits, evaluate a pretrained detector against the eKYC-DF dataset and toolkit. Keep this result separate unless we can construct a sound, documented mapping into the integrated evaluation.

Optional


For the integrated demonstration, generate a separate synthetic KYC population with known ground-truth fraud rings, application timestamps, shared attributes, missing evidence, and document-manipulation indicators. Link the records to document examples where possible, while clearly marking which evidence is real benchmark output and which is simulated.

This provides an end-to-end demo with known labels without pretending that our synthetic integration data are real bank customers.
4. The experiment that could make this genuinely impressive
A working dashboard is necessary, but the strongest demonstration is a controlled experiment showing exactly when the combined system succeeds and fails.

We should run an ablation study: keep the data splits fixed, then remove one evidence source at a time.
Experiment	Signals enabled	Question answered
E0	Simple rules	How strong is the basic baseline?
E1	Rules + tabular ML	Does ML improve on the rules?
E2	Rules + identity graph	Do relationships uncover hidden fraud rings?
E3	Graph + document signals	Does document evidence add independent value?
E4	Graph + document + behavior	Does multi-source fusion improve detection?
E5	Full model with corrupted evidence	Does the improvement survive adversarial conditions?

The biometric adapter can be added as E6 if the integration and evaluation are ready.
Metrics we should show live
1. Detection quality
- PR-AUC, precision, recall and F1.
- Recall at a fixed false-positive rate.
- Precision among the top 50 flagged applications.


2. Robustness
Test with 0%, 10%, 20%, and 30% of selected identity links missing or corrupted. Report the change in PR-AUC and recall relative to the clean condition.

3. Incremental value
Compare E4 with E2. If adding document and behavioral evidence does not improve held-out performance, we should say so. The ablation is what establishes whether fusion is useful.

4. Operational usefulness
Measure inference latency, the number of applications needing manual review, and whether the system can explain the evidence behind each alert.

What would count as success?
I'd set provisional targets before running the experiments:
- A measurable PR-AUC improvement over the strongest baseline on held-out data, with uncertainty reported.
- Better fraud recall at the same false-positive rate, rather than simply flagging more applicants.
- Graceful degradation when identity relationships are removed or corrupted.
- A traceable explanation for every flagged synthetic fraud ring.
- No data leakage between training, validation, and test sets.
I would not promise a particular accuracy percentage in advance. With a small synthetic dataset, near-perfect accuracy can be a sign of an overly easy experiment rather than a strong model.

One additional test could make this more interesting: selectively removing the most informative evidence source. For example, test whether the system can still identify a fraud ring when phone-number relationships are unavailable but document and temporal signals remain. This is a specific, measurable form of missing-evidence robustness.
5. Can we actually build this in 12 hours?
Yes, we can attempt a credible prototype in 12 hours, provided we distinguish a completed experimental system from a production-ready KYC product.

Hours 0–2: Data and baselines

Load the graph dataset, inspect labels and relationships, create a leakage-safe split, and train a rule-based and tabular baseline.



Hours 2–5: Identity graph and attack simulation

Construct typed identity links, implement suspicious-link features, inject controlled missing links and camouflage, and measure graph performance.



Hours 5–7: Document evidence

Integrate an available pretrained or supplied document-forensics baseline. Record its scores separately and connect the outputs to the synthetic KYC test environment.



Hours 7–9: Fusion and evaluation

Train a calibrated fusion model using out-of-fold or otherwise leakage-safe training predictions. Run ablations and robustness tests.



Hours 9–12: Demo and results

Build the application-review dashboard, show an example fraud ring, display metrics, test failure cases, and prepare the research narrative.



This schedule assumes datasets and pretrained weights can be downloaded promptly. If a large dataset, gated model, or dependency fails, the fallback is to complete the graph and fusion experiments first, then demonstrate the document subsystem separately.

I would not attempt to train a deepfake detector or a large graph neural network from scratch in those 12 hours. We should use pretrained components and spend our limited time on integration, evaluation, and the research contribution.
6. Before we lock the architecture, I need three answers from you
I don't want to underestimate your capabilities, but I also don't want to prescribe an architecture based on assumptions about your team's setup.
1. How many people will actually code during the 12 hours?

Just me

2 developers

3–4 developers


2. What compute can you use immediately?

CPU only on laptops

Free cloud GPU / Colab / Kaggle

Local machines plus cloud GPU


3. What must be ready for judging?

A working end-to-end product demo

A strong experiment with defensible metrics

Both the working product and quantitative results


4. Which technical work can your team do confidently right now?

Python, ML, data processing and model evaluation

Full-stack development and APIs

Computer vision and pretrained vision models

All of the above

Finalize the research plan
My position is clear: don't abandon multi-source correlation, but don't confuse supporting four signal types with having four research contributions. Build one coherent system around identity-relationship robustness, use document and behavioral evidence to test whether fusion adds real value, and make the adversarial evaluation the centerpiece of the demo.

That's a much stronger story than claiming to detect every possible form of AI fraud.
















Finalize the research and engineering plan for Project ARGUS, an evidence-robust multi-source synthetic identity KYC fraud detection prototype. Team: 2. Compute: cloud. Judging deliverables: both. Current strengths: all. Available time is 12 hours. Select the best dataset combination, exact baseline models, a novel but defensible experiment, measurable success criteria, and task assignments. Prioritize a working product plus reproducible metrics; include fallback paths for data or model download failures.













Project ARGUS — final 12-hour research and engineering plan
Hacksploit 2026 · 2 developers · Cloud compute · Working product + reproducible metrics

My recommendation is to build a complete, evidence-correlating KYC prototype, with robust synthetic-identity detection as the central research contribution. We will not attempt to train four separate AI models from scratch.

The research is motivated by two important findings:
- A 2026 study on synthetic-identity detection reports that a rule-anchored graph hybrid achieved mean PR-AUC of 0.835 versus 0.777 for rule-based scoring across its controlled experiments. 
  Springer Nature Link
- The ICCV 2025 DeepID challenge found that only six of 26 participating teams beat its supplied baseline for ID-document forgery detection. It also evaluated generalization to a separate private dataset of 20,000 real document images. 
  Open Access CVF
These findings give us a compelling starting point: detect connected fraud that survives individual checks, then measure how well the system handles incomplete or contradictory evidence.
1. What we will actually deliver

Core: identity relationship intelligence

Build a typed graph connecting applicants through phone numbers, emails, identity attributes, and shared devices. Detect suspicious clusters, calculate risk features, and explain why applicants are connected.




Second source: document integrity

Use a pretrained document-forgery detector if its weights and inference code are available. Extract OCR fields and detect contradictions. If integration fails, use a clearly labeled simulated document-risk signal for the end-to-end demonstration and keep it out of real-model performance claims.




Third source: application behavior

Derive application velocity, device reuse, identifier novelty, and other available signals. Evaluate the behavioral classifier separately on a temporal fraud benchmark.




Fourth source: biometric evidence

Include a biometric adapter for face matching or a pretrained presentation-attack detector if readily available. Otherwise demonstrate the adapter with simulated scores and explicit provenance labels. No fabricated claim of biometric model accuracy.




Final product: explainable KYC risk console

Submit an application, inspect linked accounts, see evidence-level scores, investigate fraud clusters, and view reproducible model metrics and robustness curves.



This architecture satisfies the multi-source product objective without confusing a working integration with a fully validated multimodal model.
2. Dataset combination: three sources, three distinct purposes
Use these in priority order.









Dataset 1 — identity graph benchmark
Critical path
The Neo4j fraud-detection sample graph used in the October 2026 study

The paper describes 2,433 profiles, including 433 fraud-labeled clients, with client, SSN, email, and phone relationships. This is a small graph suitable for fast experiments. Its fraud labels are not independently verified ground truth for every formal definition of synthetic identity fraud. 
Springer Nature Link


Dataset and graph example · 2026 research paper and reported baselines












Dataset 2 — FantasyID
Document evaluation
A public dataset of bona fide and manipulated fantasy identity cards. The DeepID research describes face-swapping and text-inpainting attacks, and reports substantial challenges in generalizing document-forgery detection to unseen attacks. 
arXiv+1


FantasyID dataset · Official DeepID challenge resources












Dataset 3 — Bank Account Fraud (BAF)
Behavioral evaluation
Feedzai's public benchmark was generated from an anonymized real-world account-opening fraud dataset and addresses temporal dynamics, class imbalance, and bias. It is appropriate for evaluating application-level behavioral signals, not as a substitute for identity-graph ground truth. 
GitHub


Official repository and download instructions · NeurIPS 2022 benchmark paper



Critical design decision: Do not merge unrelated dataset records and pretend they belong to the same customers.

We will train and evaluate the graph, document, and behavioral components on their respective datasets. For the live integrated demo, we will generate a fourth, controlled KYC dataset with known fraud-ring labels and explicitly identified simulated evidence. This is the only way to test the integrated pipeline without inventing cross-dataset identity matches.
3. Exact model choices
No GNN is necessary for the first version. On a small graph, carefully engineered graph features and a strong tabular model give us a faster, easier-to-debug baseline. We can add a heterogeneous graph neural network only if the core experiment is already working.
ID	Model	Implementation
M0	Rule baseline	Shared-identifier counts, suspicious cluster size, simple velocity rules
M1	Tabular baseline	LightGBM on applicant-level features
M2	Graph-enhanced model	NetworkX-derived degree, connected components, typed-neighbor counts, and risk-neighbor features + LightGBM
M3	Proposed fusion	Calibrated logistic regression or LightGBM using graph, document, and behavioral scores plus evidence-availability indicators

All models must use the same held-out application split for a fair comparison.
The critical implementation details
- Build graph features using only information available at the time of the KYC decision.
- Compute connected-component and neighborhood features carefully to avoid label leakage. Do not propagate test labels into model features.
- Train the fusion layer on out-of-fold predictions or a separate training split, not on the in-sample predictions of component models.
- Calibrate the final risk scores using validation data, and freeze the decision threshold before testing.
- Keep the document detector's raw output separate from OCR consistency rules. Otherwise, the same evidence could be counted twice without justification.
For the graph baseline, begin with deterministic rules and LightGBM. A GNN is an optional experiment, not a dependency for delivering the prototype.
4. The research contribution: evidence degradation and adversarial camouflage
We need something more precise than “we use multiple data sources.”

Our proposed research hypothesis is:
A risk-fusion system that explicitly represents evidence availability and cross-source contradictions can maintain useful fraud-detection performance under missing identity relationships and unreliable individual signals, compared with a rule-based baseline and a standard fusion model.

This is a hypothesis to test, not an established result.

The central experiment will compare M0–M3 under a controlled attack matrix.
Attack matrix
Each attack is applied to the held-out test data with fixed random seeds. Report clean and degraded performance separately.
Condition	What changes
Clean	Original test records and relationships.
Missing links	Remove 10%, 20%, and 30% of eligible identity edges.
Camouflage	Add controlled links between fraudulent and legitimate profiles to weaken obvious cluster patterns.
Missing source	Remove document or behavioral evidence for selected applications and compare degradation.
Contradictory signals	Introduce controlled disagreements between evidence-source scores, while preserving the ground-truth labels.


One important qualification: randomly deleting graph edges tests robustness to missing relationships, but it does not fully reproduce an intelligent attack. We should call these controlled corruption tests, and describe camouflage experiments separately.
What would make the result novel?
The individual ingredients are established. The potential contribution is the joint, controlled evaluation of missing graph relationships, cross-source evidence loss, and fusion behavior, using one fixed protocol and an auditable comparison.

Before claiming novelty, search prior work for the exact combination. The recent graph paper already studies identity-noise levels and hybrid graph models, so simply adding random edge deletion would not be enough. 
Springer Nature Link


Our stronger differentiator is to measure whether the fusion layer adds value when different sources fail independently, and whether it knows when to abstain and send an uncertain case for manual review.
5. Measurable success criteria
These are engineering targets for the hackathon, not predicted results.
Metric	Acceptance target
Graph baseline	Reproduce a meaningful result with M0 and M1 before attempting M3
Fusion improvement	M3 improves held-out PR-AUC over the strongest baseline; report the absolute delta and uncertainty
Robustness	Measure PR-AUC and recall degradation at each corruption level
False-positive control	Compare models at the same validation-selected false-positive rate
Explainability	Every flagged application displays its supporting evidence and linked-account paths
Reproducibility	Fixed seeds, saved test split, model configurations, and one-command evaluation
Product demo	Submit an application, inspect its evidence, view the risk decision, and open the linked fraud cluster

Do not set “99% accuracy” as the target. In imbalanced fraud detection, accuracy can look excellent even when a model misses most fraudulent applications.

Use these metrics in the results report:
- PR-AUC: primary ranking metric for the imbalanced fraud task.
- Precision and recall: with the decision threshold selected on validation data.
- Recall at a fixed false-positive rate: useful for measuring detection under operational constraints.
- Brier score or calibration error: assesses risk-score reliability.
- Inference latency: median and p95 on a documented machine.
- Risk coverage: fraud detection performance when the system automatically handles only high-confidence cases and defers uncertain cases to review.
For a small graph dataset, confidence intervals may be wide. We should report that honestly and avoid presenting synthetic corruption experiments as evidence of production performance.
6. Divide the work between two developers

Developer 1 — ML, graph and experiments
Primary owner of research validity
- Load and validate the graph dataset.
- Implement M0–M3, leakage-safe splits and controlled perturbations.
- Build the ablation and robustness evaluation scripts.
- Generate metrics, plots and a machine-readable results report.
- Keep a record of failed experiments as well as successful ones.




Developer 2 — Data integration, API and product
Primary owner of the live demonstration
- Create the application schema and synthetic KYC scenario generator.
- Implement evidence-source adapters and the risk API.
- Build the applicant-review dashboard and graph visualization.
- Integrate the document detector if dependencies and weights are ready.
- Package the demo and verify that the entire flow works end to end.



Shared responsibility

Agree on the data schema and API contract during the first 20 minutes. Both developers run the same integration test before the final hour. Developer 1 supplies a stable model interface; Developer 2 must never invent model outputs to make the dashboard look convincing.


Suggested architecture
Use a small, modular Python application rather than introducing distributed infrastructure into a 12-hour project.
- Data and ML: Python, pandas, scikit-learn, LightGBM, NetworkX.
- Document processing: OpenCV and an OCR library; pretrained forensics model only if readily deployable.
- API: FastAPI with Pydantic schemas.
- Dashboard: Streamlit for speed, with Plotly for graph and metric visualizations.
- Persistence: SQLite or Parquet for the prototype.
- Reproducibility: pinned dependencies, fixed seeds, JSON metrics and a documented evaluation command.
PostgreSQL, Neo4j, Docker orchestration, and complex GNN infrastructure are optional. Add them only if the basic pipeline is already complete.
7. The 12-hour execution schedule
00:00–00:30

Freeze the scope

Agree on schema, repository structure, dataset access and the core hypothesis.


00:30–02:00

Establish the graph baseline

Load the client records, validate labels, build graph features, and get M0 running.


02:00–04:00

Train and evaluate M1/M2

Train LightGBM, build typed graph features, establish a common test split and save predictions.


04:00–06:00

Build robustness experiments

Implement missing-link and camouflage perturbations, then produce initial ablation metrics.


06:00–08:00

Integrate the risk engine

Implement the API and fusion layer. Connect real document scores if the detector is available.


08:00–10:00

Complete the KYC console

Show evidence, connected profiles, decisions, and a controlled fraud-ring scenario.


10:00–11:00

Freeze and rerun evaluation

Run the full experiment, save the metrics, verify thresholds and check for leakage.


11:00–12:00

Demo rehearsal

Test the happy path and failure paths, prepare the evidence slide, and rehearse the demonstration.



Non-negotiable milestone: By hour 4, M0 and at least one ML baseline must produce reproducible metrics. By hour 8, the product must support an end-to-end application review. If either milestone slips, reduce scope instead of spending the final hours debugging a complex model.
8. Failure paths: protect the demo
Failure	Fallback
Graph dataset download fails	Use a small, documented synthetic graph with deterministic labels; clearly label the results as synthetic and do not claim reproduction of published benchmarks.
BAF access or download fails	Skip the independent behavioral benchmark. Use generated behavioral features for the demo, without reporting them as BAF results.
FantasyID download or model weights fail	Use OCR and explicit consistency rules; show the document-forensics adapter as unavailable rather than pretending to have a trained detector.
GPU unavailable	Use CPU LightGBM and NetworkX. The core graph benchmark is small enough to attempt on CPU.
Fusion model performs worse	Keep the best baseline as the operational model and report the negative ablation. The dashboard can still demonstrate multi-source evidence and uncertainty.
Integration runs late	Serve a saved, versioned model and prediction artifacts through the same API. Make it clear when a result comes from a cached benchmark versus a live inference.

For the document subsystem, the published DeepID benchmark is especially instructive: its test protocol included unseen attacks and an out-of-domain dataset, so a model that works only on familiar manipulation examples should not be presented as robust. 
Open Access CVF

9. The live demo judges should see









Scene 1 — Legitimate applicant

A clean application with no suspicious identity links. Show the available evidence, risk score and decision.












Scene 2 — Hidden synthetic identity ring

Individually plausible profiles form a suspicious network through shared identifiers. Show the links that caused the alert and compare M0 with M2.












Scene 3 — Adversarial stress test

Remove some links or introduce controlled camouflage. Plot baseline and proposed-model performance as the evidence deteriorates.












Scene 4 — Evidence fusion and uncertainty

Show how document, graph and behavioral signals contribute to the integrated decision. Missing signals are marked unavailable, and uncertain cases are sent for review.



Final decision
Build ARGUS as one end-to-end product with a rigorous graph-centered experiment, not four disconnected detectors.

The strongest defensible claim will be something like:
We developed an explainable, multi-source KYC fraud-detection prototype and evaluated how identity-linkage corruption and missing evidence affect its performance against transparent rule-based and tabular ML baselines.

If the results support it, we can add a quantified statement about the improvement. If they do not, the experiment still tells us where fusion helps, where it fails, and which signals matter.

That is how we turn a compelling demo into a credible engineering and research result.
















i have some queries , should we make a mobile application ? so we can get the fingerprint sensor to work ?? is it possible ?? , then also use google authenticator app to get 2 step authentication , then for the user its just kyc ap right ?? also not while just doing the KYC can we catch them like verifing and all ?? cause there are many types of uses there right ?? not just at the start ? correct me if im wrong ? how will this be used by a real life user ?? what all interfaces will be there ?? we also need to show the real working of the system then also what are some reporting or even catching the source of the frauds ??












Worked for 12s
Planned the KYC architecture


Searched 4 websites













Man, yes — a mobile app could make ARGUS a much stronger live demo, but we need to separate three things: authenticating the user, verifying their identity, and detecting fraud.

Your instinct about continuous detection is also correct. KYC is not just a one-time check at signup. A real system can screen someone when they open an account, monitor later activity, and trigger another verification when suspicious changes occur.

The key is to build a product that demonstrates this lifecycle without spending your 12 hours building a full commercial banking app.
1. Should we build a mobile KYC app with fingerprint support?
Yes, but use the fingerprint sensor for local biometric authentication—not as proof that the applicant is the person on their identity documents.









Fingerprint / device authentication

The app invokes Android's biometric prompt. The operating system verifies the enrolled fingerprint and tells the app whether authentication succeeded. Ordinary apps do not receive the raw fingerprint image or fingerprint template.

Useful for: confirming that someone can unlock the device or approve a sensitive action.












Document and face verification

The app captures an ID document and a selfie or short video. ARGUS evaluates document manipulation, face consistency and other available signals.

Useful for: identifying suspicious or inconsistent identity evidence.












Identity-graph investigation

ARGUS checks whether the applicant is connected to other accounts through reused identifiers or suspicious relationships.

Useful for: discovering coordinated synthetic identities that might each look legitimate when examined separately.



These signals answer different questions. A fraudster can pass fingerprint authentication on their own phone while submitting a fabricated identity. Conversely, a legitimate family might share an address or device. ARGUS must combine evidence rather than automatically treating a shared attribute as proof of fraud.
2. Should we add Google Authenticator for two-step authentication?
Technically, yes. But I would not make Google Authenticator mandatory for ordinary applicants completing KYC.

Google Authenticator can generate time-based one-time passwords (TOTP). Your backend can validate those codes if you implement the corresponding enrollment and verification flow. However, this proves control of an authenticator secret; it does not prove that the applicant's identity documents are genuine.

Use authentication where it makes the most sense:
User or action	Recommended authentication
Applicant signing in	Phone/email verification, or a passkey
Applicant approving a sensitive action	Device biometric prompt, with fallback
Fraud analyst accessing the dashboard	Password plus TOTP or another strong second factor
Administrator changing system settings	Strong MFA, with audit logging
Risky account recovery	Step-up verification and additional fraud checks

For the hackathon, we can simulate or implement analyst MFA if time permits. It's a security feature, not the core fraud-detection innovation, so it comes after the graph engine and working investigation dashboard.

Android provides a system biometric prompt for this kind of app-level authentication. 
Android Developers

3. Can ARGUS detect fraud after KYC, too?
Absolutely. This is one of the most important corrections to the product design.

KYC establishes and verifies a customer's identity at particular points in a relationship. Ongoing due diligence can also involve monitoring whether activity remains consistent with the customer's known profile and risk. The RBI's KYC Direction explicitly covers ongoing due diligence and risk-aligned monitoring. 
Master Directions+1


Here is the lifecycle we should demonstrate:

1. Account application

Applicant submits identity details and documents. ARGUS checks for manipulated documents, inconsistent fields, reused identifiers, and suspicious links to existing profiles.
Onboarding screening




2. Account linking and activation

The new applicant is compared with the existing identity graph. If several supposedly unrelated applicants share a suspicious set of identifiers, ARGUS can flag a possible coordinated ring.




3. Ongoing activity

A connected financial institution could send events such as unusual transaction velocity, rapid changes in account details, or activity inconsistent with the profile. ARGUS can update risk and create alerts.




4. Risk-triggered re-verification

A high-risk event can trigger a request for additional evidence, a manual review, or a step-up authentication challenge. A risk score alone should not automatically establish that fraud occurred.




5. Investigation and reporting

The analyst sees the evidence, connected accounts, timeline of alerts, and reasons for the risk score, then records a decision and exports a case report.



One important distinction: we can build the event pipeline and demonstrate continuous monitoring using simulated events, but our hackathon prototype won't be monitoring real bank accounts unless a legitimate institution provides an integration and permission to access that data.
4. What interfaces should the real product have?
ARGUS should not be just a KYC app. The applicant app collects evidence; the fraud analyst console is where the core product's value becomes visible.

I'd design three interfaces, plus an optional integration view.









Interface A — Applicant app

For the customer
- Start a new application and enter identity details.
- Capture or upload an ID document and selfie.
- Complete a simulated biometric/device-authentication step.
- View the application status and respond to requests for additional evidence.
This should feel like a normal, trustworthy KYC experience—not a cybersecurity tool.












Interface B — Fraud analyst console

The primary product interface
- Live queue of flagged applications and accounts.
- Applicant risk score and its contributing signals.
- Interactive identity graph showing connected applicants.
- Document findings and behavioral alerts.
- Case timeline, evidence details and analyst decisions.
- Exportable investigation reports.
This is where we demonstrate why ARGUS catches a fraud ring that individual document checks could miss.












Interface C — Research and evaluation dashboard

For judges and technical reviewers
- Rule-based vs tabular ML vs graph-enhanced model.
- Precision, recall, PR-AUC and false-positive rate.
- Clean data vs missing links vs camouflage stress tests.
- Evidence-source ablations and latency.
- Dataset provenance, test configuration and experiment results.
This demonstrates that ARGUS is not merely a convincing UI; it has measurable experimental results.












Optional Interface D — Integration and event simulator

A control panel for sending fake account-opening, profile-change and transaction events into ARGUS. It shows which event was received, which checks ran, and whether a new alert or case was created.

This gives us a repeatable live demo without relying on a real bank or real customer data.



5. Can we actually catch the source of a fraud ring?
Yes—with an important distinction: ARGUS can identify suspicious connections and explain why they matter. It cannot automatically prove who committed the fraud or who controls every linked account.

Consider this demo scenario:
Illustrative identity-ring investigation
Entirely simulated example

High risk






What the investigator should see
1. Connection evidence: multiple applications share a phone number and device identifier.
2. Temporal evidence: the applications arrived in a short time window, if the event data supports that observation.
3. Document evidence: one or more IDs show suspicious manipulation signals, if the document detector flags them.
4. Model explanation: the graph model and other signals contributed to the alert. The analyst can inspect each contributing feature instead of seeing only a black-box score.
5. Investigation outcome: the analyst marks the case as confirmed fraud, benign, or unresolved only after reviewing the available evidence.

The system can report which identifiers, relationships and events led to the alert. If a device identifier is linked to multiple accounts, we can flag that relationship. But we must not claim it reveals the real-world operator's name or physical location unless we have reliable, authorized evidence supporting that conclusion.
What reports should ARGUS generate?
Report	Contents
Applicant risk report	Risk score, reasons, evidence sources and missing evidence
Identity-ring report	Connected profiles, shared identifiers, relationship paths and time window
Document review report	Manipulation signals, field inconsistencies and review status
Event timeline	Application events, profile changes, alerts and analyst actions
Model evaluation report	Precision, recall, PR-AUC, false positives and robustness results
Audit report	Who reviewed a case, what action they took and when

For a hackathon, export these as JSON or CSV, and ideally a printable case summary. We can demonstrate a regulator-style reporting workflow, but we should label it as a prototype rather than claim that it files official fraud reports or connects to law-enforcement systems.
6. How should we implement this in 12 hours?
My recommendation is a mobile-friendly applicant interface plus a web-based analyst console. Don't build two full native applications.
Build priorities
Must build

Applicant submission flow, API, identity graph, risk scoring, analyst investigation screen, seeded fraud-ring scenario and a reproducible metrics page.


Must demonstrate

Submit a new applicant → detect suspicious relationships → create an alert → inspect the graph → record a decision → export the report.


Stretch features

Real Android fingerprint prompt, TOTP for analysts, document manipulation model, and a live behavioral-event simulator. Implement these only after the core flow works.



If someone on your team already knows Android development, we can add a small native Android client and use the real system biometric prompt. If not, a responsive web app with a simulated biometric step is a safer use of the time. We must label simulated results honestly.

The backend can expose event types such as application_submitted, document_uploaded, account_activated, profile_changed, and transaction_flagged. Each event can trigger appropriate checks and append an auditable result to the applicant's timeline.

One design rule matters: do not turn every risk alert into an automatic rejection. Use risk tiers such as approve, request additional evidence, and manual review. A suspicious shared address or device alone is not proof of fraud.
My final recommendation
Build ARGUS as an end-to-end, graph-centered KYC fraud investigation platform, not as a fingerprint app with fraud detection bolted on.
- The mobile-facing flow makes the product tangible.
- The identity graph is the key differentiator.
- Continuous event screening shows the product works beyond onboarding.
- The investigation console explains why an applicant was flagged and which other profiles are connected.
- The evaluation dashboard proves what the model can and cannot detect.
That gives the judges a real workflow to interact with, a fraud-ring scenario they can watch unfold, and quantitative evidence to evaluate—not just a polished frontend.