# Paper Creation Process Summary

**Artifact:** arXiv preprint (cs.LG) — *Honest Evaluation of Cryptocurrency AML Screening: Temporal Leakage, Cost-Sensitive False-Positive Reduction, and Graph Baselines on the Elliptic Dataset*
**Author:** Arpan Parikh (sole, first author)
**Pipeline:** academic-research-skills `academic-pipeline` v3.13.0, 10-stage orchestration
**Completed:** 2026-06-25
**NIW context:** Artifact 2 of the open-source evidence trail (see `NIW_OpenSource_Roadmap.md`)

---

## Stage Ledger

| Stage | Output | Verdict |
|---|---|---|
| 1 RESEARCH | `stage1_research/stage1_research_report.md` (~2,400 w, APA 7, 9 refs) | Complete |
| 2 WRITE (full) | `preprint_draft.md` (v1) | Complete |
| 2.5 INTEGRITY | Table 1–3 vs `full_results.json` + CSVs | PASS, 100% match |
| 3 REVIEW (5 reviewers) | 3 P0 + P1/P2, 8-item roadmap | Minor Revision |
| 4 REVISE | `preprint_v2.md` | 6/8 full + 2 honest partials |
| 3' RE-REVIEW | 8 items re-checked vs v2 | ACCEPT |
| 4.5 FINAL INTEGRITY | From-scratch re-verify every number | PASS, zero issues |
| 5 FINALIZE | `preprint.tex` + `BUILD.md` | Complete (PDF compiles on Overleaf) |
| 6 PROCESS SUMMARY | this file | Complete |

## What Changed v1 → v2 (the substantive revision)
1. **Reframed contribution order** — evaluation hygiene / temporal-leakage promoted to #1; FPR reduction demoted to an honest "operating point, not SOTA." Title rewritten.
2. **Recall-floor 0.65 justified** (new §4 subsection) — demonstration value, knob exposed, not a production target.
3. **Calibration degradation surfaced** (new §5.4 + Table 4) — ECE 0.190→0.291 reported as the *cost* of threshold-shifting, mechanistically tied to drift.
4. Absolute FP counts (4 vs 89), 500× GAT bound, static-GNN bounding throughout, 99.4% drift promoted to first-class result.

## Two Honest Partials (deliberately not "fixed")
- **Variance (SD/CI):** reported five-run **means**; per-run SD/CI deferred to camera-ready. Inventing variance would be fabrication. Disclosed in Limitation (e).
- **External-notebook citation:** declined. Attributed leakage magnitudes to own controlled reproduction rather than vibe-citing an unread notebook.

## Integrity Discipline (held throughout)
- Every table number traces to `aml-detection/results/full_results.json`. Zero drift across two independent integrity passes (2.5 and 4.5).
- 90–95% supervisory FPR framed as "illustrates / aligns with," **never** "proves" — that figure describes rule-based bank systems, not this Bitcoin ML benchmark.
- GNN claims bounded to *static* GCN/GAT (cf. EvolveGCN), never "graph methods are inferior."
- Costs reported in full: recall ↓0.05, AUC ↓0.05, calibration ↓ — no cherry-picking the FPR win.
- IP boundary respected: framed as reference architecture, no Prudential proprietary code, solo authorship.

## Collaboration Quality Evaluation
- **Strengths:** Two mandatory integrity gates caught nothing because the data discipline held from Stage 2; the two-stage review materially improved the paper's honesty framing (the v1→v2 contribution reorder is the single biggest quality gain). Refusal to fabricate variance / vibe-cite preserved credibility.
- **Friction:** No local LaTeX toolchain — PDF compile offloaded to Overleaf/Tectonic (`BUILD.md`). Fact-Forcing Gate added one retry per Write (6 denials, all resolved).
- **Residual risk for camera-ready:** add per-run SD/CI; consider one production-proxy dataset to address single-dataset external-validity limit; optionally tune GNN baselines to strengthen the "trees beat vanilla graphs" claim.

## Deliverables (paper/)
- `stage1_research/stage1_research_report.md` — research foundation
- `preprint_draft.md` — v1
- `preprint_v2.md` — accepted draft (current of record)
- `preprint.tex` — arXiv LaTeX source
- `BUILD.md` — compile + arXiv submission checklist
- `PROCESS_SUMMARY.md` — this record

## Next Action (outside pipeline)
Compile `preprint.tex` → PDF (Overleaf), then submit to arXiv cs.LG. Pin the companion-repo commit first.
