# CLAUDE.md — Financial LLM Governance Repo

This file is the master context document for all Claude Code sessions working in this repository.

---

## Who This Is For

**Petitioner:** Arpan Parikh, Senior ML Engineer at Prudential Financial
**Purpose:** EB-2 National Interest Waiver (NIW) petition under the Dhanasar (2016) 3-prong test
**Current focus:** Prong 1 (Substantial Merit & National Importance) — building a verifiable open-source trail

---

## Proposed Endeavor (PE) — One-Sentence Definition

> Architect and deploy sovereign AI governance infrastructure for the U.S. financial services sector — specifically, production-grade LLM orchestration and adaptive anomaly detection systems that operationalize the 230 control objectives established by the U.S. Treasury's February 2026 Financial Services AI Risk Management Framework (FS AI RMF).

**Always lead with this sentence in READMEs, blog posts, and papers.**

---

## Four PE Goals

| Goal | Description | Key Tech |
|------|-------------|----------|
| **Goal 1** | Sovereign AI Gateways for Regulatory Compliance | Kong API Gateway, JWT/OAuth 2.0, PII-scrubbing middleware, AWS API Gateway |
| **Goal 2** | AI Risk Management & Observability Pipelines | Eval harnesses, hallucination detection, model drift monitoring |
| **Goal 3** | AI-Powered Anomaly Detection for AML/KYC | Hybrid supervised/unsupervised ML, false-positive reduction (target: <10% vs 90–95% baseline) |
| **Goal 4** | Agentic AI for Regulatory Document Automation | XAI (SHAP, LIME), STR generation, FINRA/SEC audit trails |

---

## Regulatory Anchors — Cite in Every Public Artifact

- **U.S. Treasury FS AI RMF (Feb 2026)** — 230 control objectives, identifies uncontrolled AI as "systemic risk"
- **Executive Order 14110** — Safe, Secure, and Trustworthy AI (Oct 2023)
- **NIST AI RMF 1.0 (Jan 2023)** — measure, manage, govern functions
- **FSOC 2023 Annual Report** — AI operational risk, model opacity
- **FinCEN Strategic Plan 2022–2025** — AML modernization via advanced analytics
- **OWASP LLM Application Security Top 10** — PII leakage vulnerabilities

---

## Key Metrics to Substantiate

- AML false-positive rate: **90–95%** — Saaradeey et al. (2019), *Disrupting status quo in AML compliance*, Oracle White Paper, as cited in Coelho, De Simoni & Prenio (2019), FSI Insights No. 18, BIS, p. 3
- Annual AML compliance cost to U.S. institutions: **USD 25.3 billion** — LexisNexis Risk Solutions (2018), *2018 True Cost of Compliance Study*, as cited in Coelho, De Simoni & Prenio (2019), FSI Insights No. 18, BIS, p. 3
- Primary authoritative source: Coelho, R., De Simoni, M. & Prenio, J. (2019). *Suptech applications for anti-money laundering*. FSI Insights on policy implementation, No. 18. Bank for International Settlements, August 2019. https://www.bis.org/fsi/publ/insights18.pdf
- Target: **<10% false-positive rate** via the hybrid pipeline in this repo

---

## Production Stack (Prudential Financial)

- **Kong API Gateway** — on-prem / DMZ layer
- **AWS API Gateway** — cloud boundary
- **Azure OpenAI (GPT-4o)** — LLM inference endpoint

This is the real stack. All architecture in this repo reflects it.

---

## Architecture — 9-Layer Design

The full architecture diagram is at `docs/financial_llm_governance_architecture.png`.
The source code is at `docs/architecture.py`.
The legend table is embedded in the diagram itself (rows 1–14).

### Layer Summary

| Layer | Name | FS AI RMF Pillar |
|-------|------|-----------------|
| 0 | Input Classification & Adversarial Defense | Pillar 2 |
| 1 | Kong API Gateway (On-Prem / DMZ) | Pillar 1 · Pillar 3 |
| 2 | Semantic Cache (Redis + Embeddings) | Pillar 3 |
| 3 | AWS API Gateway (Cloud Boundary) | Pillar 2 · Pillar 4 |
| 4a | RAG Layer — Regulatory Q&A Path | Pillar 1 |
| 4b | AML / KYC Pipeline | Pillar 2 |
| 5 | Azure OpenAI (GPT-4o) | — |
| 6 | Output Validation & Guardrails | Pillar 3 |
| 7 | Observability, XAI & Audit | Pillar 4 · Pillar 1 |
| 8 | Model Governance & Feedback Loop | Pillar 1 |

### Two Workload Paths

- **Path A (AML/KYC):** Transaction narrative → Kong → AWS → Hybrid AML Model → FP Reduction → Azure OpenAI → Output Validation → SHAP/LIME FINRA audit JSON
- **Path B (Reg Q&A):** Policy query → Kong → AWS → pgvector RAG (FS AI RMF / FINRA / SEC / FinCEN) → Cross-Encoder Re-ranker → Azure OpenAI → Output Validation → Langfuse trace

### Numbered Arrow Flow (matches diagram legend 1–14)

1. Clients → Workload Router: tag by type, assign risk tier
2. Workload Router → Injection Detector: scan for OWASP LLM Top 10 #1
3. Injection Detector → Kong/JWT/PII: auth enforcement, strip SSN/account/ABA
4. Kong layer → Semantic Cache: check vector cache, bypass LLM on hit
5. Cache miss → AWS layer: WAF filters, rate limiter logs
6. Rate Limiter → pgvector: Reg Q&A retrieval
7. Rate Limiter → Hybrid AML Model: AML/KYC routing
8. pgvector → Cross-Encoder Re-ranker: re-rank for relevance
9. AML Model → FP Reduction Engine: filter false alerts, annotate high-risk
10. Re-ranker / FP Engine → Azure OpenAI: inference with grounded context
11. Azure OpenAI → [ECLIPSE / PII Re-check / NeMo]: parallel output validation
12. Output Validators → Observability: LLM trace, FINRA audit JSON, metrics, SIEM
13. Observability → Governance: model registry, review queue, eval trigger
14. Eval Trigger → ECLIPSE (dashed, red): feedback loop re-eval

---

## Diagram Technical Details

**Library:** `diagrams` (mingrammer) v0.25.1
**Renderer:** Graphviz 15.1.0 (installed at `C:\Program Files\Graphviz\bin`)
**Font:** `Cascadia Code NF SemiBold` (installed on this machine)
  - Note: Circled Unicode numbers (①②③) are NOT in this font's charset — use plain numbers (1, 2, 3)
**Direction:** `TB` (top-to-bottom)
**Key layout rule:** `source >> [a, b, c]` forces a, b, c to the same graphviz rank (side-by-side)
**Legend:** Embedded as a raw graphviz HTML-label node via `diag.dot.node()` with `rank=sink` subgraph
**Feedback edge:** `constraint="False"` on the 14→ECLIPSE back-edge prevents rank disruption

### Run the diagram

```bash
export PATH="$PATH:/c/Program Files/Graphviz/bin"
cd financial-llm-governance/docs
conda run -n base python architecture.py
# Output: docs/financial_llm_governance_architecture.png
```

### Install dependencies

```bash
conda run -n base pip install diagrams pillow
winget install Graphviz.Graphviz --source winget
```

---

## Repo Structure (planned)

```
financial-llm-governance/
├── README.md                    ← NEXT TASK: write this
├── CLAUDE.md                    ← this file
├── docs/
│   ├── architecture.py          ← DONE: diagram source
│   └── financial_llm_governance_architecture.png  ← DONE: generated diagram
├── gateway/                     ← Goal 1: Kong + AWS API GW implementation
├── observability/               ← Goal 2: hallucination detection, eval harnesses
├── aml-detection/               ← Goal 3: hybrid AML pipeline
└── xai-compliance/              ← Goal 4: SHAP/LIME → FINRA audit JSON
```

---

## What Has Been Done (Session History)

- [x] Created and cloned `financial-llm-governance` repo
- [x] Designed 9-layer architecture with 14 numbered data flow steps
- [x] Generated architecture diagram (`docs/architecture.py` + PNG)
- [x] Embedded legend table (14 rows) directly in the diagram
- [x] Font: Cascadia Code NF SemiBold throughout
- [x] All rows render horizontally (using the list rank-forcing trick)
- [x] Legend anchored below diagram with `rank=sink` subgraph
- [x] Feedback edge uses `constraint=False` to prevent layout disruption

## What Is Next

- [ ] **Write README.md** — this is the immediate next task
- [ ] Contribute to `Portkey-AI/gateway` — FS AI RMF guardrail plugin, U.S. financial PII patterns, FINRA/SEC audit log schema
- [ ] Contribute to `langfuse/langfuse` — financial hallucination eval suite
- [ ] Extend Feedzai AML-Elliptic repo with U.S. benchmark (PaySim dataset)
- [ ] Implement ECLIPSE hallucination detector (`pip install financial-llm-eval`)
- [ ] Build SHAP/LIME XAI module → FINRA-compliant audit JSON
- [ ] Post arXiv preprint of AML false-positive reduction methodology
- [ ] Submit public comment on FS AI RMF at regulations.gov

---

## README Requirements (for next session)

The README must:
1. Open with the one-sentence PE definition
2. Include the architecture diagram (`docs/financial_llm_governance_architecture.png`)
3. Map each component to at least two FS AI RMF regulatory anchors by name
4. Cite OWASP LLM Top 10, NIST AI RMF, and Treasury FS AI RMF inline
5. Call out the AML false-positive reduction metric (<10% target vs 90–95% baseline)
6. Describe the two workload paths (Reg Q&A and AML/KYC)
7. Include a "Getting Started" section with install and diagram generation commands
8. Link to open-source repos being contributed to (LiteLLM, Langfuse, Feedzai)
9. List all four PE goals as repo sections
10. Keep authorship framing in first person: "I designed X", not "we will build X"

---

## NIW Filing Rules (apply to all artifacts)

1. Tie every artifact back to the PE one-sentence definition
2. Cite at least two regulatory anchors per document
3. The false-positive reduction claim (<10% vs 90–95%) is the strongest Cake Framework metric — include in every AML artifact
4. Keep authorship **solo or first-author** — adjudicators scrutinize multi-author work
5. Timestamp everything: commit early and often before filing date
6. Frame as "reference architecture" not "Prudential system" — IP boundary

---

## Open-Source Resources Reference

| Resource | URL | Contribution Plan |
|----------|-----|-------------------|
| Portkey-AI Gateway (12.1k+ stars) | https://github.com/Portkey-AI/gateway | FS AI RMF guardrail plugin, U.S. financial PII patterns (ABA/ACH/ITIN/EIN), FINRA/SEC audit log schema (Rule 4370 / 17a-4), FinCEN SAR/STR output validator |
| Langfuse | https://github.com/langfuse/langfuse | Financial hallucination eval suite, FS AI RMF audit trail module |
| Feedzai AML-Elliptic | https://github.com/feedzai/research-aml-elliptic | Extend with U.S. regulatory benchmark, measure FP reduction |
| AMLGentex | https://github.com/aidotse/AMLGentex | Add U.S. SAR/BSA alert pattern module |
| ECLIPSE paper | https://arxiv.org/abs/2512.03107 | Implement in Python as `financial-llm-eval` package |
| XAI-Comply paper | https://link.springer.com/chapter/10.1007/978-981-92-0126-6_1 | Build Python OSS equivalent |
| SHAP library | https://github.com/shap/shap | Core dependency for Goal 4 XAI module |
