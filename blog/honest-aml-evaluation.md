# Your AML model's 0.94 F1 is a lie. Here's the 0.80 that isn't.

Take the most-cited public benchmark for machine-learning anti–money-laundering — the Elliptic Bitcoin dataset — train a gradient-boosted tree on it, and you will very likely report an illicit-class F1 around 0.94 and an AUC near 0.996. Those numbers circulate in tutorials, Kaggle kernels, and more than a few papers.

I ran the exact same model, changed one line, and the F1 dropped to 0.80 and AUC to 0.943.

The line I changed was the train/test split. That's the whole story, and it's a bigger deal than any model I could have built.

## The one-line experiment

Elliptic is a time series: 203,769 Bitcoin transactions across 49 time steps. In production, a fraud model is trained on the past and scores the future. It never gets to peek at next month's transactions while learning.

A **random** 70/30 split breaks that. It scatters future transactions into the training set, so the model trains on data it could never have in deployment. A **temporal** split (train on steps 1–34, test on 35–49) respects the arrow of time.

Same XGBoost, same hyperparameters, only the split changes:

| Split | F1 (illicit) | AUC | FPR |
|---|---|---|---|
| Random 70/30 (leaky) | 0.9418 | 0.9961 | 0.0007 |
| Temporal (honest) | 0.8016 | 0.9432 | 0.0057 |

Fourteen points of F1, gone. Not because the model got worse — because the leaky version was scoring itself on an exam it had already seen. A large share of the field's optimism on Elliptic is this artifact, not model quality.

That's the headline. The rest of the work is about what you do *after* you stop lying to yourself.

## F1 is the wrong scoreboard for AML

Once you're on an honest split, the next trap is the metric. AML teams don't optimize `1 − F1`. They drown in false positives. Supervisors at the BIS Financial Stability Institute have put production false-positive rates at **90–95%** and U.S. compliance cost around **$25.3 billion a year** (Coelho et al., 2019). Every false positive is an analyst-hour.

So the real objective isn't "maximize F1." It's: **minimize the false-positive rate, subject to a floor on recall** (you still have to catch enough bad actors to satisfy your regulator). A model that lifts F1 by cranking recall can *raise* your false-positive bill. F1 hides that. FPR-at-a-recall-floor exposes it.

Here's the same study targeting the right objective — a cost-sensitive tree (class weight ≈7.63:1) with a cross-validated decision threshold tuned to minimize FPR while holding recall ≥ 0.65, plus a blended isolation-forest anomaly signal:

| Metric | Baseline | FP-optimized | Change |
|---|---|---|---|
| FPR | 0.0057 | 0.0003 | **−94.7% relative** |
| False positives (test) | 89 | 4 | −85 |
| Precision | 0.8981 | 0.9945 | +0.10 |
| Recall (illicit) | 0.7239 | 0.6726 | −0.05 |
| AUC-ROC | 0.9432 | 0.8933 | −0.05 |

Four false positives instead of 89 on the test period. That's the number an operations lead actually cares about.

![Per-time-step FPR and recall](../aml-detection/results/temporal_performance_comparison.png)

Per time step (left), the hybrid flattens the baseline's false-positive spikes almost to zero. But look at recall (right): both models dip below the 0.65 floor at several late steps. The floor holds *on average* across the window, not at every step — and I'd rather show you that than crop it out.

**And I'm going to tell you what it cost**, because the version of this post that doesn't is the dishonest version. Recall dropped 5 points. AUC dropped 5 points. The model also got *less* calibrated — expected calibration error rose from 0.19 to 0.29 — because the same severe feature drift between train and test periods (164 of 165 features shift significantly) that makes thresholds hard to transfer also pushes the scores away from true probabilities. If you feed raw scores into risk tiering, recalibrate first. The flag/no-flag decision is fine; the probability is not.

![Calibration reliability diagram](../aml-detection/results/calibration_curves.png)

The tuned model (blue) sits further from the perfect-calibration diagonal than the baseline (red). That's the cost, drawn out.

This is an honest operating point. It is not state of the art, and I'm not claiming it is. The mechanism — threshold optimization under a recall constraint — is textbook. The contribution is running it on a non-leaky split and reporting the full bill.

## Graph neural nets didn't save the day either

Elliptic is a graph, so the obvious move is a graph neural network. I ran vanilla GCN and GAT under the same temporal protocol:

| Model | FPR | F1 | Precision |
|---|---|---|---|
| FP-optimized tree | 0.0003 | 0.8024 | 0.9945 |
| GCN | 0.0206 | 0.6403 | 0.6735 |
| GAT | 0.1526 | 0.3807 | 0.2549 |

![FPR and F1 by model family](../aml-detection/results/model_family_comparison.png)

GAT's false-positive rate is roughly **500×** the tuned tree's. It flags everything. Graph structure alone does not cut false positives on Elliptic's engineered features — the cost-sensitive threshold does.

Caveat, stated plainly: these are *static, untuned* GNNs. Temporal graph methods like EvolveGCN (Pareja et al., 2020) model time explicitly and do better. I'm not saying graphs lose; I'm saying a vanilla graph net is not a free upgrade over a well-tuned tree, and you should benchmark before you believe the hype.

## Where the gateway comes in

A model that flags 4 transactions instead of 89 is only useful if a regulator can audit *why* each one fired. That's an architecture problem, not a modeling one, and it's where a governance gateway earns its keep.

The reference pattern — no proprietary code, just the shape of it:

1. **Ingress gateway** scrubs PII before any transaction payload reaches the model or a downstream LLM. Pattern-based redaction at the edge, not bolted on later.
2. **Model scores** the transaction; SHAP attributions (Lundberg & Lee, 2017) are computed per flagged transaction.
3. **Audit egress** writes each decision — score, threshold, top SHAP features — to a newline-delimited JSON record structured for **FINRA Rule 4370** and **SEC Rule 17a-4** retention.

The gateway is what turns a model output into a defensible regulatory artifact. The honest evaluation above is what makes the artifact worth defending.

## The actual takeaways

- **Split before you celebrate.** On any time-series fraud problem, a random split inflates your metrics. Use a temporal split or assume your numbers are fiction.
- **Score FPR at a recall floor, not F1.** It's the objective your operations team actually has.
- **Report the costs.** Recall, AUC, and calibration all moved against me. Hiding that would make the 94.7% FPR cut meaningless.
- **Benchmark graph nets, don't assume them.** Vanilla GNNs underperformed a tuned tree here.
- **Make every flag auditable.** PII-scrubbing ingress + SHAP-to-audit-record egress turns model outputs into FINRA/SEC-grade records.

Code, seeds, and the full results set: **github.com/gucifer/financial-llm-governance**. The companion preprint covers the methodology in full.

*One thing I won't claim: this study does not independently prove the industry's 90–95% false-positive figure. That number describes rule-based production banking systems, not an ML benchmark on Bitcoin. The study illustrates the problem and aligns with the figure. It doesn't certify it. Same discipline as everything else here.*

---

**References**
Coelho, R., De Simoni, M., & Prenio, J. (2019). *Suptech applications for anti-money laundering* (FSI Insights No. 18). Bank for International Settlements.
Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*.
Pareja, A., et al. (2020). EvolveGCN: Evolving graph convolutional networks for dynamic graphs. *AAAI 34(04)*.
Weber, M., et al. (2019). Anti-money laundering in Bitcoin: Experimenting with graph convolutional networks for financial forensics. *KDD Workshop on Anomaly Detection in Finance*.
