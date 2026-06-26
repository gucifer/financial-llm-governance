# Stage 1 Research Report — Temporal-Honest False-Positive Reduction in Cryptocurrency AML Screening

*Deep-research pipeline output (full mode). Foundation document for the arXiv preprint. APA 7.0. AI-assisted; see Ethics Statement.*

**Author:** Arpan Parikh
**Date:** 2026-06-25
**Status:** Stage 1 final (post Phase 5 review + Phase 6 revision)

---

## Abstract

Anti–money-laundering (AML) transaction monitoring is dominated by the operational cost of false positives: regulators estimate that 90–95% of alerts raised by production AML systems are false, contributing to roughly USD 25.3 billion in annual U.S. compliance costs (Coelho, De Simoni, & Prenio, 2019). This report synthesizes the literature and methodology underpinning an empirical study that reduces the false-positive rate (FPR) of illicit-transaction detection on the public Elliptic Bitcoin dataset (Weber et al., 2019). The study pairs a cost-sensitive, cross-validated threshold-optimized gradient-boosted tree ensemble with an isolation-forest anomaly blend, evaluated under an honest temporal split, and benchmarks it against graph neural baselines (GCN, GAT). Three findings emerge: (a) FPR falls from 0.0057 to 0.0003 (−94.7% relative) at a maintained recall floor of ≥0.65; (b) vanilla graph neural networks underperform engineered-feature tree ensembles on this dataset, consistent with Weber et al. (2019); and (c) random train/test splitting inflates F1 from 0.80 to 0.94, explaining the optimistic numbers in public Elliptic notebooks and constituting a methodological contribution. The work is positioned as an evaluation-hygiene and compliance-FPR contribution, not a new architecture, and explicitly does not independently prove the industry-wide 90–95% statistic.

**Keywords:** anti-money laundering, false-positive reduction, Elliptic dataset, temporal evaluation, graph neural networks, cost-sensitive learning, model explainability

---

## 1. Introduction

Anti–money-laundering compliance is a national-interest problem at the intersection of financial integrity and machine learning. Rule-based transaction-monitoring systems in production financial institutions generate alert volumes in which the overwhelming majority are false positives. The Bank for International Settlements' Financial Stability Institute reports that false-positive rates in production AML systems reach 90–95%, and that U.S. institutions spend approximately USD 25.3 billion annually on AML compliance (Coelho et al., 2019, p. 3). Each false positive consumes scarce human investigative capacity, and the cumulative drag both raises costs and contributes to financial exclusion when institutions de-risk by exiting client relationships (Weber et al., 2019).

The advent of cryptocurrency reframes this problem. Pseudonymous public ledgers allow illicit actors to operate in the open, but the same transparency provides researchers with rich, structured transaction data unavailable in traditional banking. The Elliptic dataset (Weber et al., 2019) — 203,769 Bitcoin transactions across 49 time steps, with 166 node features and licit/illicit/unlabeled labels — became the canonical public benchmark for this line of research.

A substantial body of subsequent work optimizes accuracy or F1 on Elliptic, frequently reporting F1 scores above 0.93. This report and its companion study argue that two methodological problems undermine the practical relevance of much of that work. First, the dominant evaluation metric (F1, accuracy) is misaligned with the operational reality of AML, where the cost function is dominated by false positives at a regulator-mandated detection (recall) floor. Second, many public evaluations use random train/test splits that leak future information across the temporal structure of the data, inflating reported performance.

**Research question.** Can a cost-sensitive, threshold-optimized tree ensemble reduce the FPR of illicit-transaction detection on Elliptic — under an honest temporal split — without breaching a compliance recall floor, and how does it compare to graph neural baselines?

**Sub-questions.**
- SQ1: What FPR reduction is achievable at a fixed recall floor (≥0.65) versus a Feedzai-style XGBoost baseline?
- SQ2: Do GCN/GAT models on the transaction graph beat gradient-boosted trees on engineered features for FPR under the same temporal split?
- SQ3: How much does random versus temporal splitting inflate reported performance?

**Contribution and honest scope.** The contribution is methodological, not architectural: (a) a recall-floor-constrained FPR reduction result, (b) a controlled temporal-versus-random leakage demonstration, and (c) a graph-versus-tree comparison under a single honest protocol. The study illustrates and aligns with the BIS-reported 90–95% industry FPR figure; it does not independently prove it, because that figure refers to rule-based production systems on proprietary banking data, not an ML benchmark on Bitcoin transactions.

---

## 2. Literature Review

### 2.1 The Elliptic dataset and graph-based forensics

Weber et al. (2019), a collaboration of the MIT-IBM Watson AI Lab and Elliptic, introduced the Elliptic dataset at the KDD '19 Workshop on Anomaly Detection in Finance (arXiv:1908.02591). They compared logistic regression, random forest, multilayer perceptron, and a graph convolutional network (GCN) under a 70:30 *time-based* split. Their headline result is instructive: a random forest on engineered features (94 local + 72 aggregated) was the strongest model, outperforming the GCN on illicit-class F1. They observed that local-plus-neighborhood features carry much of the graph signal but cannot be extended beyond the immediate neighborhood — the motivation for graph neural networks — yet the feature-based tree model remained the practical leader. This concurrence is central to the present study: a finding that vanilla GNNs underperform trees on Elliptic is consistent with, not contradictory to, the dataset's originating paper.

### 2.2 Label scarcity and the inadequacy of unsupervised methods

Lorenz et al. (2020), a Feedzai team, addressed AML detection on Bitcoin under severe label scarcity at ACM ICAIF '20 (arXiv:2005.14635). Two findings inform the present methodology. First, they demonstrate that state-of-the-art *unsupervised* anomaly detection methods are inadequate to detect illicit patterns in real Bitcoin transaction data on their own. Second, their active-learning approach matches a fully supervised baseline using only 5% of labels. The first finding directly motivates the present study's design choice to use isolation forest (Liu, Ting, & Zhou, 2008) not as a standalone detector but as a *blended* anomaly signal within a supervised pipeline.

### 2.3 Temporal graph neural networks

Pareja et al. (2020) introduced EvolveGCN at AAAI 2020 (vol. 34, pp. 5363–5370), evolving GCN parameters over time via a recurrent network rather than post-updating embeddings. EvolveGCN was evaluated on Elliptic among other dynamic graphs and is the standard reference for temporal GNNs outperforming static baselines on this data. Its existence bounds the present study's claims: the study benchmarks *vanilla* GCN (Kipf & Welling, 2017) and GAT (Veličković et al., 2018), not temporal GNNs, and therefore claims only that static graph models underperform trees here — not that graph methods are categorically inferior.

### 2.4 Cost-sensitive learning, boosting, and explainability

The supervised engine is XGBoost (Chen & Guestrin, 2016), with class imbalance addressed through `scale_pos_weight` set to the licit-to-illicit ratio. Decision-threshold optimization under a recall constraint is the mechanism by which FPR is reduced. Model explainability uses SHAP (Lundberg & Lee, 2017), whose additive feature-attribution values are mapped to per-transaction audit records aligned with FINRA Rule 4370 and SEC Rule 17a-4 record-retention expectations.

### 2.5 Synthesis and the gap

Across this corpus, three themes converge. (1) Graph structure does not automatically beat engineered features on Elliptic (Weber et al., 2019; bounded by Pareja et al., 2020). (2) Unsupervised anomaly detection alone is insufficient (Lorenz et al., 2020), justifying a blended rather than standalone use of isolation forest. (3) Evaluation hygiene is under-reported: the field optimizes F1/accuracy, often under leaky random splits, rather than FPR at a compliance recall floor under a temporal split. No widely cited Elliptic study isolates the temporal-versus-random leakage effect as a controlled experiment while targeting FPR under an explicit recall floor. That intersection is the gap the study fills.

---

## 3. Methodology Blueprint

**Paradigm.** Positivist, empirical-quantitative controlled benchmark.

**Data.** Elliptic Bitcoin dataset (Weber et al., 2019): 203,769 transactions, 49 time steps, 166 features, ~21% labeled (licit/illicit).

**Splits.** Primary protocol is the *temporal* split (train time steps 1–34, test 35–49), matching the Feedzai/Weber convention. A *random* 70/30 split is run as a negative control to quantify leakage.

**Models.**
1. *Baseline:* XGBoost reproducing the Feedzai/Weber tabular setup.
2. *Hybrid:* cost-sensitive XGBoost (`scale_pos_weight` ≈ 7.6:1) + cross-validated precision–recall threshold optimization with a recall floor ≥0.65 + isolation-forest anomaly score blended at weight 0.15, with the blend applied *inside* CV folds during threshold search to preserve threshold consistency under feature drift.
3. *Graph baselines:* GCN (Kipf & Welling, 2017) and GAT (Veličković et al., 2018) on the transaction graph (≈470k undirected edges), same temporal split, class-weighted loss, recall-floor threshold tuned on a leakage-free training-period hold-out.

**Primary metric.** FPR at fixed recall floor. Secondary: precision, illicit-class F1, AUC-ROC.

**Validity controls.** Five-run averaging; leakage-free threshold tuning; temporal-versus-random control experiment; reproducible code, seeds, and environment.

**Explainability/audit.** SHAP attributions exported to per-transaction NDJSON audit records mapped to FINRA Rule 4370 / SEC Rule 17a-4.

**Reporting standard.** No human subjects; no IRB required. Reproducibility checklist (data version, seeds, environment) is the governing guideline.

---

## 4. Findings

The findings below are drawn from the completed companion study and are reproduced here as the evidentiary basis for the preprint. They are measured results on the Elliptic test set.

### 4.1 FPR reduction at a recall floor (SQ1)

| Metric | Baseline (XGBoost) | Hybrid | Delta |
|---|---|---|---|
| FPR | 0.0057 | 0.0003 | −94.7% relative |
| Recall (illicit) | 0.7239 | 0.6726 | −0.05 (floor ≥0.65 held) |
| Precision | 0.8981 | 0.9945 | +0.10 |
| F1 (illicit) | 0.8016 | 0.8024 | +0.001 |
| AUC-ROC | 0.9432 | 0.8933 | −0.05 |
| Threshold | 0.5 | 0.8324 | CV-optimized |

The hybrid reduces FPR by 94.7% relative while holding recall above the 0.65 compliance floor; precision rises and F1 is preserved. The AUC and recall reductions are reported transparently as the cost of threshold-shifting toward precision.

### 4.2 Graph versus tree (SQ2)

| Model | FPR | F1 (illicit) | AUC |
|---|---|---|---|
| XGBoost baseline | 0.0057 | 0.8016 | 0.9432 |
| Hybrid FP-optimized | 0.0003 | 0.8024 | 0.8933 |
| GCN | 0.0206 | 0.6403 | 0.8787 |
| GAT | 0.1526 | 0.3807 | 0.8851 |

Vanilla GCN and GAT underperform both tree models on FPR and F1, consistent with Weber et al. (2019). Graph structure alone does not reduce FPR on Elliptic's engineered features; the cost-sensitive threshold pipeline does. This finding is explicitly bounded to static GNNs (cf. EvolveGCN; Pareja et al., 2020).

### 4.3 Evaluation pitfalls — temporal versus random (SQ3)

| Split | F1 (illicit) | AUC | FPR |
|---|---|---|---|
| Random 70/30 (leaky) | 0.9418 | 0.9961 | 0.0007 |
| Temporal (honest) | 0.8016 | 0.9432 | 0.0057 |

With identical hyperparameters, random splitting inflates F1 from 0.80 to 0.94, matching the headline figures in public Elliptic notebooks. This converts a common evaluation flaw into a methodological contribution and reinforces the study's honesty narrative.

---

## 5. Discussion

**Interpretation.** The study demonstrates that the practically relevant lever for AML cost reduction is not a more expressive model class but a cost-sensitive decision threshold tuned under a compliance recall constraint. The FPR reduction is large and real on this dataset, but it is achieved by trading a controlled amount of recall and AUC for precision — a trade the AML cost function favors, and one the study reports without obfuscation.

**Implications for regulatory AI.** The result maps to the U.S. Treasury FS AI RMF emphasis on measurable model performance and to FinCEN's AML-modernization agenda. The SHAP-to-audit-record layer addresses the explainability and recordkeeping expectations embedded in FINRA Rule 4370 and SEC Rule 17a-4.

**Limitations.** (a) Single dataset; external validity to production banking data is not established. (b) GNN baselines are vanilla and may be undertuned relative to temporal GNNs. (c) The 90–95% industry FPR figure is illustrated and aligned with, not independently proven. (d) Elliptic's features are partially anonymized, limiting feature-level interpretability of the SHAP outputs.

**Future work.** Replication on a second dataset (e.g., a synthetic transaction dataset), temporal-GNN comparison (EvolveGCN), and an LLM-assisted suspicious-activity-report generation layer mapped to the broader governance architecture.

---

## 6. Acknowledged Limitations (Phase 6 carry-forward)

Two issues raised during review remain as acknowledged limitations rather than fully resolved items: the single-dataset external-validity gap (mitigated by explicit scoping, not eliminated) and the undertuned-vanilla-GNN concern (mitigated by bounding the claim to static GNNs and citing EvolveGCN, not by exhaustive tuning).

---

## 7. Ethics Statement

This research used AI-assisted tools (Claude Code, ARS pipeline) for literature synthesis, drafting, and review. All references were independently verified against primary sources (arXiv, ACM, AAAI, BIS). No human subjects were involved. Data are public (Elliptic). The work is single-author. No conflicts of interest. Dual-use note: the methods detect illicit transactions for compliance purposes; they are not deployed for surveillance beyond AML scope.

---

## References

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM.

Coelho, R., De Simoni, M., & Prenio, J. (2019). *Suptech applications for anti-money laundering* (FSI Insights on Policy Implementation No. 18). Bank for International Settlements. https://www.bis.org/fsi/publ/insights18.pdf

Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. In *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1609.02907

Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest. In *2008 Eighth IEEE International Conference on Data Mining* (pp. 413–422). IEEE.

Lorenz, J., Silva, M. I., Aparício, D., Ascensão, J. T., & Bizarro, P. (2020). Machine learning methods to detect money laundering in the Bitcoin blockchain in the presence of label scarcity. In *Proceedings of the First ACM International Conference on AI in Finance (ICAIF '20)* (pp. 1–8). ACM. https://arxiv.org/abs/2005.14635

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems 30 (NeurIPS)* (pp. 4765–4774).

Pareja, A., Domeniconi, G., Chen, J., Ma, T., Suzumura, T., Kanezashi, H., Kaler, T., Schardl, T., & Leiserson, C. (2020). EvolveGCN: Evolving graph convolutional networks for dynamic graphs. *Proceedings of the AAAI Conference on Artificial Intelligence, 34*(04), 5363–5370.

Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. In *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1710.10903

Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., & Leiserson, C. E. (2019). Anti-money laundering in Bitcoin: Experimenting with graph convolutional networks for financial forensics. In *KDD '19 Workshop on Anomaly Detection in Finance*. https://arxiv.org/abs/1908.02591
