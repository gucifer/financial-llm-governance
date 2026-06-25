# CLAUDE.md — Financial LLM Governance Repo

This file is the master context document for all Claude Code sessions working in this repository.

---

## Who This Is For

**Petitioner:** Arpan Parikh, Senior ML Engineer at Prudential Financial
**Purpose:** EB-2 National Interest Waiver (NIW) petition under the Dhanasar (2016) 3-prong test
**Current focus:** Prong 1 (Substantial Merit & National Importance) — building a verifiable open-source trail
**Roadmap file:** `c:\Users\apari\OneDrive\Desktop\eb2_niw\NIW_OpenSource_Roadmap.md` — read this for the full strategy

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

## Key Metrics (verified, cite these exactly)

- AML false-positive rate: **90–95%** — Saaradeey et al. (2019), *Disrupting status quo in AML compliance*, Oracle White Paper, as cited in Coelho, De Simoni & Prenio (2019), FSI Insights No. 18, BIS, p. 3
- Annual AML compliance cost to U.S. institutions: **USD 25.3 billion** — LexisNexis Risk Solutions (2018), *2018 True Cost of Compliance Study*, as cited in Coelho, De Simoni & Prenio (2019), FSI Insights No. 18, BIS, p. 3
- Primary source: Coelho, R., De Simoni, M. & Prenio, J. (2019). *Suptech applications for anti-money laundering.* FSI Insights No. 18, BIS, August 2019. https://www.bis.org/fsi/publ/insights18.pdf
- **Measured study result:** FPR 0.0057 → 0.0003 (−94.7% relative) on Elliptic test set, recall 0.6726 (floor ≥ 0.65 satisfied), F1 0.8024

---

## Production Stack (Prudential Financial)

- **Kong API Gateway** — on-prem / DMZ layer
- **AWS API Gateway** — cloud boundary
- **Azure OpenAI (GPT-4o)** — LLM inference endpoint

This is the real stack. All architecture in this repo reflects it. Frame as "reference architecture" not "Prudential system" — IP boundary.

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

---

## Repo Structure (current state)

```
financial-llm-governance/
├── README.md                    ← exists (comprehensive, written in prior session)
├── CLAUDE.md                    ← this file
├── docs/
│   ├── architecture.py          ← DONE
│   └── financial_llm_governance_architecture.png  ← DONE
├── gateway/                     ← Goal 1: stub only
├── observability/               ← Goal 2: stub only
├── aml-detection/               ← Goal 3: COMPLETE (see below)
│   ├── README.md                ← populated with real results
│   ├── requirements.txt
│   ├── data/elliptic/dataset/   ← Elliptic CSVs here (not committed, excluded by .gitignore)
│   ├── src/
│   │   ├── data_loader.py       ← loads Elliptic, temporal split
│   │   ├── baseline.py          ← Feedzai XGBoost baseline + FPR metrics
│   │   ├── hybrid_pipeline.py   ← cost-sensitive XGB + CV threshold opt + IF blend
│   │   ├── shap_audit.py        ← SHAP → FINRA Rule 4370 / SEC Rule 17a-4 audit JSON
│   │   └── eval_harness.py      ← KS drift + Brier/ECE calibration + temporal FPR
│   ├── notebooks/
│   │   └── aml_fp_reduction_study.ipynb  ← end-to-end executable (DONE, clean run)
│   └── results/                 ← 12 artifacts: plots, audit NDJSON, CSVs, full_results.json
└── xai-compliance/              ← Goal 4: stub only (SHAP work folded into aml-detection)
```

---

## AML Detection Study — Key Implementation Details

**Dataset:** Elliptic Bitcoin dataset (Weber et al., 2019), 203,769 txns / 49 time steps / ~21% labeled  
**Temporal split:** train time steps 1–34, test 35–49 (matches Feedzai section 4.1)  
**Conda env:** `c:\Users\apari\OneDrive\Desktop\eb2_niw\.conda\python.exe` (Python 3.13.13)  
**Run Python from this env, not the system Python 3.14 which lacks the ML packages**  
**Installed:** scikit-learn, xgboost, shap, imbalanced-learn, scipy, matplotlib, seaborn + **torch 2.12.1+cpu, torch-geometric 2.8.0** (for the GNN baseline)

### Three contributions over Feedzai baseline

1. `scale_pos_weight = n_licit / n_illicit` (~7.6:1 on train) — cost-sensitive XGBoost
2. Cross-validated precision-recall threshold optimisation — recall floor 0.65 (used in notebook; 0.70 is the code default but causes threshold transfer failure due to 99.4% feature drift between train/test periods)
3. Isolation Forest trained on licit-class only, anomaly score blended at weight=0.15; IF blending is applied **inside CV folds** during threshold search (not just at test time) — critical for threshold consistency

### Measured results (5-run average, Elliptic test set)

| Metric | Baseline | Hybrid | Delta |
|--------|----------|--------|-------|
| FPR | 0.0057 | 0.0003 | −94.7% |
| Recall | 0.7239 | 0.6726 | −0.05 (floor ≥ 0.65 satisfied) |
| Precision | 0.8981 | 0.9945 | +0.10 |
| F1 | 0.8016 | 0.8024 | +0.001 |
| AUC-ROC | 0.9432 | 0.8933 | −0.05 |
| Threshold | 0.5 | 0.8324 | CV-optimised |

### Model-family comparison (graph baselines — `src/gnn_baseline.py`)

GCN (Kipf & Welling 2017) and GAT (Veličković et al. 2018) on the actual transaction graph (203,769 nodes, ~470k undirected edges, same 165 features), evaluated under the **same** temporal split + illicit-class/FPR metrics. Cost-sensitive (class-weighted loss); recall-floor threshold tuned on a leakage-free training-period hold-out.

| Model | FPR | F1 (illicit) | AUC |
|-------|-----|------|-----|
| XGBoost baseline | 0.0057 | 0.8016 | 0.9432 |
| Hybrid FP-optimised | 0.0003 | 0.8024 | 0.8933 |
| GCN | 0.0206 | 0.6403 | 0.8787 |
| GAT | 0.1526 | 0.3807 | 0.8851 |

**Finding:** gradient-boosted trees beat vanilla GNNs on Elliptic's engineered features (consistent with Weber 2019 / Feedzai); graph structure alone does NOT reduce FPR — the cost-sensitive threshold pipeline does. (GCN runtime ~13 min, GAT ~7 min on CPU.)

### Evaluation-pitfalls control (Stage 5 of the notebook)

Same XGBoost, identical hyperparameters, only the split differs:

| Split | F1 (illicit) | AUC | FPR |
|-------|------|-----|-----|
| Random 70/30 (leaky) | 0.9418 | 0.9961 | 0.0007 |
| Temporal (honest) | 0.8016 | 0.9432 | 0.0057 |

Random splitting leaks future time steps and inflates F1 from 0.80 → 0.94 — matching the headline numbers in public Elliptic notebooks (e.g. the reference RF notebook's 0.938). This converts a common flaw into a methodological contribution and reinforces the honesty narrative. The 3 third-party reference notebooks that prompted this live in `aml-detection/notebooks/reference/` and are **git-ignored** (provenance protection).

### Scope caveat (always include)

The study demonstrates FPR reduction methodology on public data. It illustrates and aligns with the industry-wide 90–95% FPR figure but does NOT independently prove it. That figure rests on the BIS/LexisNexis citation above and refers to rule-based production AML systems, not an ML benchmark on Bitcoin data.

---

## Git State

**Remote:** `origin/main` → https://github.com/gucifer/financial-llm-governance.git  
**Latest commits (PUSHED to GitHub 2026-06-25):**
- `1be1c49` — "Add GNN baselines (GCN/GAT) and evaluation-pitfalls analysis"
- `dd7e724` — "Add AML false-positive reduction study (Tier 1 flagship)"

**Status:** ✅ Both commits pushed; flagship is timestamped on GitHub. The repo (Artifact 1) is now public and verifiable.

**README honesty pass (2026-06-25 — committed locally, push pending):**
- `README.md` — rewritten to stop overclaiming: Goal 3 now shows real measured results + honesty caveat; Goals 1/2/4 marked "reference design (scaffolded)"; PaySim→Elliptic fixed; AMLGentex removed (CUT); contributions table reframed (shipped/planned legend); Portkey PR #1691 kept as a real shipped contribution (user-confirmed).
- `CLAUDE.md` — this file (status updates).
- `aml-detection/README.md` — BibTeX URL fixed (aparikhdev→gucifer).
- Untracked `aml-detection/data/` — Elliptic CSVs, **not for committing** (not currently matched by `.gitignore`).

**Next actions: `git push origin main` to publish the honesty pass, then the arXiv preprint** (Artifact 2).

```bash
cd "c:/Users/apari/OneDrive/Desktop/eb2_niw/financial-llm-governance"
git push origin main
```

---

## What Has Been Done

- [x] Created and cloned `financial-llm-governance` repo on GitHub
- [x] Designed 9-layer architecture with 14 numbered data flow steps
- [x] Generated architecture diagram (`docs/architecture.py` + PNG)
- [x] Written comprehensive repo README.md
- [x] **Tier 1 flagship AML study — complete:**
  - [x] `aml-detection/src/` — 6 modules written and tested (incl. `gnn_baseline.py`)
  - [x] `aml-detection/notebooks/aml_fp_reduction_study.ipynb` — 6 stages, runs end-to-end clean (29 cells, 0 errors)
  - [x] `aml-detection/results/` — 15+ artifacts (plots, NDJSON, CSVs, JSON)
  - [x] `aml-detection/README.md` — real results + model-family comparison + evaluation-pitfalls tables
  - [x] GNN baselines (GCN/GAT) and evaluation-pitfalls control added
  - [x] Reference notebooks isolated to `notebooks/reference/` (git-ignored) for provenance
  - [x] Committed locally as `dd7e724` + `1be1c49`

## What Is Next (in priority order)

1. ~~**Push to GitHub**~~ — ✅ DONE 2026-06-25 (`dd7e724` + `1be1c49` on origin/main)
2. **Commit + push the uncommitted README honesty pass** (see Git State above) — do this first
3. **Write arXiv preprint** ← THE MOST URGENT WRITE-UP ACTION — 4–6 pages; content is already in the README and notebook; needs reformatting into paper structure; arXiv endorsement for cs.LG may be needed (start early)
4. **Technical blog post** — Medium or Towards Data Science; cover AML study + Goal 1 gateway architecture pattern (no Prudential code); link repo and preprint
5. **Evidence log** — start a spreadsheet tracking stars, forks, citations from day of push
6. **Workshop paper submission** — same preprint reformatted; FinNLP@EMNLP or IEEE S&P financial-AI track; do not gate filing on acceptance
7. **Fix petition body** — remove `[ACTION ITEM]` brackets; fix LexisNexis citation year (petition says "2023", correct source is 2018/BIS)

> **Scope note:** the flagship is now feature-complete (tabular baseline + hybrid + SHAP audit + observability + GNN baselines + pitfalls control). The GNN was the first and last addition beyond the roadmap's "one dataset, one baseline, one improvement" line — justified by reviewer expectation. **Resist further model additions**; remaining effort goes into write-up artifacts (preprint, blog), not more code.

---

## NIW Filing Rules (apply to all artifacts)

1. Tie every artifact back to the PE one-sentence definition
2. Cite at least two regulatory anchors per document
3. The false-positive reduction claim is the strongest Cake Framework metric — include in every AML artifact; frame as "illustrates and aligns with" the 90–95% figure, not "proves" it
4. Keep authorship **solo or first-author** everywhere — adjudicators scrutinize multi-author work
5. Timestamp everything: commit early and often; push before filing
6. Frame as "reference architecture" not "Prudential system" — IP boundary

---

## Open-Source Resources Reference (trimmed to what's used)

| Resource | URL | Used in |
|----------|-----|---------|
| Feedzai AML-Elliptic | https://github.com/feedzai/research-aml-elliptic | Tier 1 flagship baseline |
| SHAP library | https://github.com/shap/shap | aml-detection/src/shap_audit.py |
| PyTorch Geometric | https://github.com/pyg-team/pytorch_geometric | aml-detection/src/gnn_baseline.py (GCN/GAT) |
| Portkey-AI Gateway (12.1k+ stars) | https://github.com/Portkey-AI/gateway | Tier 3 optional PR |
| Langfuse | https://github.com/langfuse/langfuse | Tier 3 optional PR |
| regulations.gov | https://www.regulations.gov | Tier 2 public comment |
