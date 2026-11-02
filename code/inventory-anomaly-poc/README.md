# Inventory Anomaly Detection POC and Model Comparison

PyTorch · scikit-learn · Azure ML (job + scoring templates) · Python · SQL (SQLite) · Flask · Power BI-style reconciliation

## Abstract
A proof of concept that learns from **historical inventory counts by item and location** and **flags unusual counts for manual
review**. Six detector families are compared on identical data, features and a time-ordered split: a **regression** residual
detector, a **decision tree**, an **SVM**, a **random-forest + gradient-boosting ensemble**, a **PyTorch classification model
(MLP)**, and a **two-stage PyTorch model - the MLP classification model feeding a GRU (RNN) classifier**. Results are averaged over
5 independently generated datasets. A **Flask site** lets a reviewer confirm or dismiss flags, and a **SQL + Python
reconciliation** checks dashboard (Power BI-style) totals against source records.

> **Data note.** Real inventory data is private, so every number here is measured on **synthetic data with injected, labelled
> anomalies** (needed to score detectors objectively). The Azure ML files are **templates that were not run** (no Azure
> subscription in the build environment), and Power BI is represented by a CSV export. Everything else runs locally.

## Requirements
1. Flag unusual counts using **history by item and location** (no peeking at the future).
2. Flags go to a **person for manual review** - so the review queue size (flag rate) matters as much as accuracy.
3. Compare model families **fairly and reproducibly**: same features, same split, thresholds chosen on validation data only.
4. Reported dashboard totals must **match source records**.
5. Provide a usable review interface and data pipeline.

## Goal
Pick a detector that catches most true anomalies with a manageable review load, understand what each model *misses*, and make
the reported numbers trustworthy.

## Work done
| Piece | File |
|---|---|
| Synthetic data: 60 items x 8 locations x 104 weeks, seasonality + trend + per-series noise + **benign promotion surges**; 5 labelled anomaly types | `src/generate_data.py` |
| Leakage-safe features (week *t* uses only counts up to *t*; robust median/MAD stats; seasonal lag) and time-ordered split | `src/features.py` |
| Regression, decision tree, SVM, RF+GB ensemble | `src/classical_models.py` |
| PyTorch **MLP -> GRU** two-stage model (stage 1 classification model; stage 2 RNN over the last 12 weeks of stage-1 embeddings + probabilities) | `src/torch_models.py` |
| Training/evaluation over 5 seeds, thresholding on validation, error analysis, SQLite output | `src/run_comparison.py` |
| SQL + Python reconciliation of dashboard totals vs source | `src/reconcile.py` |
| Flask review site (queue, confirm/dismiss, search, JSON API) | `src/app.py` |
| Azure ML job + scoring script **templates (not run)** | `azureml/` |

**Setup details.** 49,920 rows; 12-week warm-up dropped -> train weeks 12-77 (31,680 rows, ~620 anomalies), validation 78-89, test 90-103
(6,720 rows, 115-154 anomalies per seed). Anomalies (~1.7% of rows): spike (x1.4-4), drop (x0.15-0.7), zero count, x10
unit error, and "stale" (previous week's number repeated). Each model's alert threshold maximises F1 on validation weeks.

## Results (measured; `results/metrics.md`, mean ± std over 5 datasets)
| model | precision | recall | F1 | PR-AUC | flagged / 6,720 test rows |
|---|---:|---:|---:|---:|---:|
| **PyTorch MLP -> GRU (two-stage)** | 0.857 ± 0.079 | 0.752 ± 0.072 | **0.794 ± 0.007** | **0.872 ± 0.022** | 118 (1.8%) |
| decision tree | 0.942 ± 0.030 | 0.669 ± 0.036 | 0.781 ± 0.022 | 0.783 ± 0.036 | 93 (1.4%) |
| regression (residual) | 0.935 ± 0.023 | 0.674 ± 0.061 | 0.781 ± 0.040 | 0.771 ± 0.031 | 94 (1.4%) |
| PyTorch MLP (stage 1 only) | 0.860 ± 0.043 | 0.720 ± 0.060 | 0.781 ± 0.022 | 0.869 ± 0.024 | 111 (1.6%) |
| ensemble (RF + GB) | 0.868 ± 0.092 | 0.706 ± 0.057 | 0.772 ± 0.010 | 0.865 ± 0.020 | 109 (1.6%) |
| SVM (rbf) | 0.737 ± 0.088 | 0.645 ± 0.057 | 0.684 ± 0.051 | 0.711 ± 0.034 | 116 (1.7%) |

**Recall by anomaly type** (seed-42 test set, `results/recall_by_anomaly_type.md`)
| model | spike | drop | zero | x10 | stale |
|---|---:|---:|---:|---:|---:|
| regression | 0.78 | 0.64 | 1.00 | 1.00 | **0.00** |
| decision tree | 0.81 | 0.73 | 0.96 | 1.00 | **0.00** |
| SVM | 0.74 | 0.82 | 0.81 | 0.45 | 0.08 |
| ensemble | 0.85 | 0.73 | 1.00 | 1.00 | 0.17 |
| MLP (stage 1) | 0.85 | 0.64 | 1.00 | 1.00 | 0.17 |
| MLP -> GRU | 0.85 | 0.77 | 1.00 | 1.00 | 0.17 |
(n per type on that test set: 27 spike, 22 drop, 27 zero, 20 x10, 24 stale.)

**Reconciliation** (`results/reconciliation.json`; 126 location-week cells; defects injected into the dashboard import)
| | |
|---|---:|
| cells where dashboard != source | **99 / 126 (78.6%)** |
| dashboard total vs source total | 1,745,085 vs 1,717,031 (**+28,054 net**, 51,670 gross) |
| root cause: duplicate submissions summed twice | 32 cells, +10,547 |
| root cause: corrections summed on top of the original | 19 cells, +5,333 |
| root cause: duplicates + corrections together | 34 cells, +23,982 |
| root cause: NULL-location records dropped | 14 cells, -11,808 |
| residual after fix (independent pandas implementation vs SQL) | **0** |

## Findings
1. **The overall F1 ranking is mostly within noise.** The five best models span F1 0.772-0.794 and the standard deviations
   across datasets (0.007-0.040) overlap. The two-stage MLP->GRU has the highest mean F1 (0.794) and, notably, the most
   *stable* (±0.007), but I would not claim it beats the decision tree (0.781 ± 0.022) on F1 alone. The SVM is clearly worst (0.684).
2. **PR-AUC separates the families more clearly** than F1 at a single threshold: the neural and ensemble models rank true anomalies
   higher (0.865-0.872) than the tree and regression (0.771-0.783). For a *review queue ordered by score* that matters - the
   top of the queue is more likely to be real anomalies - even when a single-threshold F1 ties.
3. **Precision/recall trade-off = review workload.** Tree and regression are conservative (precision 0.94, recall 0.67, ~93 flags);
   the GRU model finds more (recall 0.75) at the cost of precision 0.86 and ~25 more flags per 6,720 rows. Which is "better" depends
   on the cost of a missed anomaly vs a wasted review.
4. **Context helps modestly.** Adding the GRU over 12 weeks of stage-1 outputs raised mean recall 0.720 -> 0.752 and F1
   0.781 -> 0.794 vs stage 1 alone, with similar PR-AUC (0.869 -> 0.872) - a small gain, not a transformation.
5. **Every model misses "stale" counts** (a repeated previous number): recall 0.00-0.17. A stale count looks statistically normal;
   catching it needs a different signal (scan timestamps, update frequency), not a better value-based model. Large
   structural errors (zero counts, x10) are caught at ~1.00 by all except the SVM (x10 recall 0.45).
6. **The dashboard disagreed with the source in 79% of cells**, from three data-handling defects (duplicates, correction rows
   summed with the originals, NULL locations dropped), all fixable in SQL with "latest submission per key" + a bucket for missing
   locations; the fixed aggregate was confirmed by a second, independent implementation (residual 0).

## Honest limitations
- **Synthetic data**: absolute scores say nothing about real inventory. A first version of the generator (large, isolated anomalies
  only) let **every model score ~100%**, so it could not rank them; the final generator adds noisier series, benign promotion
  surges and subtler/stale anomalies *because the comparison was uninformative*, not to favour a model. Anomaly types were fixed
  before looking at results of the final version.
- Differences among the top models are within seed-to-seed variation (5 datasets); PyTorch CPU training adds ±0.005 run-to-run
  jitter (the printed table can shift in the third decimal when re-run).
- The reconciliation defects are **injected** to give the check something to find; it demonstrates the method, not real
  Power BI findings. No Power BI connection was made.
- `azureml/` (job YAML, conda file, scoring script) is a template: not executed on Azure, and exporting the chosen model for the
  scoring script is not implemented. No demo to stakeholders is recorded in this repo.
- Hyper-parameters were set once by hand (no tuning search); SVM trains on a 9,000-row subsample for speed.

## Reproduce
```bash
pip install numpy pandas scipy scikit-learn torch flask
python src/generate_data.py          # optional: writes data/inventory_counts.csv
python src/run_comparison.py         # ~2 min; metrics.{md,json}, recall_by_anomaly_type.md, data/inventory.db
python src/reconcile.py              # reconciliation.json, dashboard_export.csv, mismatches
python src/app.py                    # http://127.0.0.1:5000  (review queue; needs data/inventory.db)
```
