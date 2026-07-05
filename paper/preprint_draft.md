# Temporal-Honest False-Positive Reduction in Cryptocurrency AML Screening: A Cost-Sensitive Benchmark with Graph and Evaluation-Pitfall Controls

**Arpan Parikh**
Independent Research
`washrmn@gmail.com`

*Preprint. Companion code: https://github.com/gucifer/financial-llm-governance*

---

## Abstract

False positives dominate the operational cost of anti–money-laundering (AML) transaction monitoring. Supervisory estimates place the false-positive rate (FPR) of production AML systems at 90–95% and U.S. AML compliance spending at approximately USD 25.3 billion per year (Coelho et al., 2019). Yet the academic benchmark literature on the public Elliptic Bitcoin dataset (Weber et al., 2019) overwhelmingly optimizes F1 or accuracy, frequently under random train/test splits that leak temporal information. We present a controlled study that targets the operationally relevant objective directly: minimizing FPR subject to a fixed recall floor, under an honest temporal split. A cost-sensitive, cross-validated threshold-optimized gradient-boosted tree ensemble with a blended isolation-forest anomaly signal reduces FPR from 0.0057 to 0.0003 (−94.7% relative) while holding illicit-class recall at 0.67 (floor ≥ 0.65) and raising precision from 0.90 to 0.99. We add two controls absent from most prior work: (i) vanilla graph neural baselines (GCN, GAT) evaluated under the identical temporal protocol, which underperform the tree ensemble and confirm that graph structure alone does not reduce FPR on this dataset; and (ii) a temporal-versus-random split experiment showing that random splitting inflates illicit-class F1 from 0.80 to 0.94, explaining the optimistic numbers reported in public Elliptic notebooks. We frame the work as an evaluation-hygiene and compliance-FPR contribution rather than a new architecture, and we explicitly decline to claim independent proof of the industry-wide 90–95% statistic. Per-prediction SHAP attributions are exported to audit records aligned with FINRA Rule 4511 and SEC Rule 17a-4.

---

## 1. Introduction

Anti–money-laundering compliance carries a cost structure dominated by false positives. Rule-based transaction-monitoring systems in production financial institutions raise alert volumes in which the great majority are benign, and each alert consumes scarce human investigative capacity. The Bank for International Settlements' Financial Stability Institute reports production AML false-positive rates of 90–95% and annual U.S. AML compliance costs near USD 25.3 billion (Coelho et al., 2019, p. 3). Beyond direct expense, the burden contributes to financial exclusion when institutions de-risk by exiting client relationships (Weber et al., 2019).

The Elliptic dataset (Weber et al., 2019) made this problem tractable for open research: 203,769 Bitcoin transactions across 49 time steps, 166 node features, and licit/illicit/unlabeled labels, released as the largest labeled public cryptocurrency transaction graph at the time. A large secondary literature followed, much of it reporting illicit-class F1 above 0.93.

We argue that two methodological habits limit the practical relevance of those numbers. **First, the optimization target is misaligned.** AML operations do not minimize 1 − F1; they minimize investigative load (false positives) subject to a regulator-driven detection floor (recall). A model that improves F1 by raising recall at the cost of precision can *increase* false positives. **Second, evaluation frequently leaks time.** Elliptic is a temporal sequence; a random train/test split places future transactions in the training set, inflating measured performance relative to the deployment setting, where a model trained on the past must score the future.

We address both. Our research question: *Can a cost-sensitive, threshold-optimized tree ensemble reduce the FPR of illicit-transaction detection on Elliptic — under an honest temporal split — without breaching a compliance recall floor, and how does it compare to graph neural baselines?* We decompose it into three sub-questions: (SQ1) the achievable FPR reduction at a fixed recall floor versus a strong tabular baseline; (SQ2) whether graph neural networks beat trees on FPR under the same protocol; and (SQ3) the magnitude of inflation introduced by random versus temporal splitting.

**Contributions.**
1. A recall-floor-constrained FPR reduction on Elliptic: −94.7% relative FPR at recall ≥ 0.65, with the recall/AUC trade-off reported transparently.
2. A graph-versus-tree comparison under one honest temporal protocol, showing vanilla GCN/GAT underperform trees here.
3. A controlled temporal-versus-random split experiment quantifying leakage-driven inflation, which we offer as an evaluation-hygiene contribution.
4. A reproducible pipeline with SHAP-to-audit-record export for regulatory recordkeeping.

We are explicit about scope. The contribution is methodological, not a new model. The study *illustrates and aligns with* the 90–95% supervisory FPR figure; it does not independently prove it, because that figure describes rule-based production systems on proprietary banking data, not an ML benchmark on Bitcoin.

---

## 2. Related Work

**Elliptic and graph-based forensics.** Weber et al. (2019) introduced the dataset and compared logistic regression, random forest, a multilayer perceptron, and a GCN under a 70:30 time-based split. Notably, the random forest on engineered features (94 local + 72 aggregated) was their strongest model, outperforming the GCN on illicit-class F1. They noted that local-plus-neighborhood features capture much of the graph signal but cannot be extended beyond the immediate neighborhood, motivating graph neural networks — while the feature-based tree model remained the practical leader. Our finding that vanilla GNNs trail trees is therefore consistent with the originating paper.

**Label scarcity.** Lorenz et al. (2020) studied AML on Bitcoin under severe label scarcity, showing that unsupervised anomaly-detection methods alone are inadequate on real transaction data, and that active learning matches a supervised baseline with only 5% of labels. The first result motivates our use of isolation forest (Liu et al., 2008) as a *blended* signal inside a supervised pipeline rather than as a standalone detector.

**Temporal graph networks.** Pareja et al. (2020) introduced EvolveGCN, evolving GCN weights over time via a recurrent network, and demonstrated gains on dynamic graphs including Elliptic. EvolveGCN bounds our claims: we benchmark *static* GCN (Kipf & Welling, 2017) and GAT (Veličković et al., 2018), not temporal GNNs, so we claim only that static graph models underperform trees here, not that graph methods are categorically inferior.

**Boosting and explainability.** Our supervised engine is XGBoost (Chen & Guestrin, 2016). Threshold optimization under a recall constraint is the FPR-reduction mechanism. Explainability uses SHAP (Lundberg & Lee, 2017), whose additive attributions we map to per-transaction audit records.

**Gap.** No widely cited Elliptic study isolates the temporal-versus-random leakage effect as a controlled experiment while simultaneously targeting FPR under an explicit recall floor. That intersection is our focus.

---

## 3. Data

The Elliptic dataset (Weber et al., 2019) comprises 203,769 Bitcoin transactions over 49 time steps. Each transaction (node) carries 166 features: 94 local and 72 aggregated neighborhood features. Roughly 21% of nodes are labeled (≈2% illicit and ≈21% licit of the total, the remainder unlabeled); we use the labeled subset for supervised training and evaluation. Edges encode Bitcoin flows, yielding a graph of ≈470k undirected edges used by the graph models. We adopt the temporal split convention: train on time steps 1–34, test on 35–49. This matches the deployment setting in which a model trained on historical data scores subsequent transactions.

---

## 4. Method

**Baseline.** An XGBoost classifier reproducing the tabular setup of the Elliptic/Feedzai line, scored on the temporal test set. This is our FPR reference point.

**Hybrid pipeline.** Three components extend the baseline:

1. *Cost-sensitive boosting.* We set `scale_pos_weight` to the licit-to-illicit ratio on the training period (≈ 7.6:1), reweighting the loss toward the minority illicit class.

2. *Recall-floor threshold optimization.* Rather than the default 0.5 decision threshold, we search the precision–recall curve under cross-validation for the threshold that minimizes FPR subject to recall ≥ 0.65, a stand-in for a regulator-mandated detection floor. The selected operating threshold is 0.83.

3. *Blended anomaly signal.* An isolation forest (Liu et al., 2008) trained on the licit class produces an anomaly score blended into the decision score at weight 0.15. Critically, the blend is applied *inside* the cross-validation folds during threshold search, not only at test time, which preserves threshold consistency under the substantial feature drift between training and test periods.

**Graph baselines.** We train GCN (Kipf & Welling, 2017) and GAT (Veličković et al., 2018) on the transaction graph under the identical temporal split, with class-weighted loss and a recall-floor threshold tuned on a leakage-free training-period hold-out. These isolate whether graph structure, on its own, reduces FPR.

**Evaluation.** Primary metric is FPR at the recall floor; secondary metrics are precision, illicit-class F1, and AUC-ROC. Results are averaged over five runs. To quantify evaluation leakage, we re-run the identical XGBoost configuration under a random 70/30 split and compare.

**Explainability and audit.** For flagged transactions we compute SHAP values (Lundberg & Lee, 2017) and export per-transaction records to newline-delimited JSON, structured for FINRA Rule 4511 and SEC Rule 17a-4 record-retention expectations.

---

## 5. Results

### 5.1 FPR reduction at a recall floor (SQ1)

Table 1 reports the baseline and hybrid on the Elliptic temporal test set (five-run average).

**Table 1. Hybrid vs. baseline, temporal split.**

| Metric | Baseline | Hybrid | Delta |
|---|---|---|---|
| FPR | 0.0057 | 0.0003 | −94.7% rel. |
| Recall (illicit) | 0.7239 | 0.6726 | −0.05 |
| Precision | 0.8981 | 0.9945 | +0.10 |
| F1 (illicit) | 0.8016 | 0.8024 | +0.001 |
| AUC-ROC | 0.9432 | 0.8933 | −0.05 |
| Threshold | 0.5 | 0.8324 | CV-opt. |

The hybrid cuts FPR by 94.7% relative while holding recall above the 0.65 floor and lifting precision to 0.99. F1 is essentially unchanged. The recall and AUC reductions are the transparent cost of shifting the operating point toward precision — exactly the trade the AML cost function favors.

### 5.2 Graph versus tree (SQ2)

**Table 2. Model-family comparison, temporal split.**

| Model | FPR | F1 (illicit) | AUC |
|---|---|---|---|
| XGBoost baseline | 0.0057 | 0.8016 | 0.9432 |
| Hybrid FP-optimized | 0.0003 | 0.8024 | 0.8933 |
| GCN | 0.0206 | 0.6403 | 0.8787 |
| GAT | 0.1526 | 0.3807 | 0.8851 |

Vanilla GCN and GAT underperform both tree models on FPR and F1. GAT in particular produces an FPR two orders of magnitude worse than the hybrid. Graph structure alone does not reduce FPR on Elliptic's engineered features; the cost-sensitive threshold pipeline does. We bound this to static GNNs (cf. EvolveGCN; Pareja et al., 2020).

### 5.3 Evaluation pitfalls — temporal versus random (SQ3)

**Table 3. Identical XGBoost, split varied.**

| Split | F1 (illicit) | AUC | FPR |
|---|---|---|---|
| Random 70/30 (leaky) | 0.9418 | 0.9961 | 0.0007 |
| Temporal (honest) | 0.8016 | 0.9432 | 0.0057 |

With identical hyperparameters, random splitting inflates illicit-class F1 from 0.80 to 0.94 and AUC to 0.996. These leaky numbers match the headline figures commonly reported in public Elliptic notebooks, indicating that a substantial share of optimistic published performance is attributable to temporal leakage rather than model quality.

---

## 6. Discussion

The practically relevant lever for AML cost reduction here is not a more expressive model class but a cost-sensitive decision threshold tuned under a compliance recall constraint. The FPR reduction is large and real on this dataset, achieved by trading a controlled amount of recall and AUC for precision. We report that trade rather than obscure it.

The graph-versus-tree result is a caution against assuming architectural sophistication transfers to operational metrics. On Elliptic's engineered features, static GNNs do not beat well-tuned trees on FPR; the literature's temporal-GNN gains (Pareja et al., 2020) require modeling the time dimension explicitly.

The split experiment has the broadest implication. If random-split numbers circulate as state of the art, practitioners may select models and set expectations on inflated evidence. Reporting FPR under a temporal split at a stated recall floor is a small change in protocol with a large change in honesty.

For regulatory AI, the results map to the U.S. Treasury Financial Services AI Risk Management Framework emphasis on measurable performance and to FinCEN's AML-modernization agenda; the SHAP-to-audit-record layer addresses explainability and recordkeeping expectations embedded in FINRA Rule 4511 and SEC Rule 17a-4.

---

## 7. Limitations

(a) **Single dataset.** External validity to production banking data is not established; Elliptic is Bitcoin-specific and partially anonymized. (b) **Vanilla GNN baselines** may be undertuned relative to temporal GNNs; we bound claims accordingly. (c) **The 90–95% industry FPR figure** is illustrated and aligned with, not independently proven. (d) **Feature anonymization** limits the semantic interpretability of SHAP outputs.

---

## 8. Conclusion

Targeting the operationally correct objective — FPR at a fixed recall floor, under an honest temporal split — yields a 94.7% relative FPR reduction on Elliptic from a cost-sensitive, threshold-optimized tree ensemble, while vanilla graph baselines underperform and random-split evaluation inflates reported F1 by 14 points. The takeaway is methodological: in AML benchmarking, the choice of objective and split governs the credibility of the result at least as much as the choice of model.

**Reproducibility.** Code, seeds, and the full results set are available at the companion repository.

---

## References

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

Coelho, R., De Simoni, M., & Prenio, J. (2019). *Suptech applications for anti-money laundering* (FSI Insights on Policy Implementation No. 18). Bank for International Settlements. https://www.bis.org/fsi/publ/insights18.pdf

Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *International Conference on Learning Representations*. https://arxiv.org/abs/1609.02907

Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest. *2008 Eighth IEEE International Conference on Data Mining*, 413–422.

Lorenz, J., Silva, M. I., Aparício, D., Ascensão, J. T., & Bizarro, P. (2020). Machine learning methods to detect money laundering in the Bitcoin blockchain in the presence of label scarcity. *Proceedings of the First ACM International Conference on AI in Finance*, 1–8. https://arxiv.org/abs/2005.14635

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems 30*, 4765–4774.

Pareja, A., Domeniconi, G., Chen, J., Ma, T., Suzumura, T., Kanezashi, H., Kaler, T., Schardl, T., & Leiserson, C. (2020). EvolveGCN: Evolving graph convolutional networks for dynamic graphs. *Proceedings of the AAAI Conference on Artificial Intelligence, 34*(04), 5363–5370.

Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. *International Conference on Learning Representations*. https://arxiv.org/abs/1710.10903

Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., & Leiserson, C. E. (2019). Anti-money laundering in Bitcoin: Experimenting with graph convolutional networks for financial forensics. *KDD '19 Workshop on Anomaly Detection in Finance*. https://arxiv.org/abs/1908.02591
