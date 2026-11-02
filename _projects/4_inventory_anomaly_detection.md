---
layout: page
title: Inventory Anomaly Detection POC and Model Comparison
description: Indiana University | PyTorch, Azure ML, Python, SQL, Flask, Power BI
importance: 1
category: academic
---

- Prototyped an Azure ML POC that uses historical counts by item and location to flag unusual inventory counts for manual review, and demoed it.
- Trained and fine-tuned PyTorch models, including a classification model feeding an RNN classifier, and compared regression, decision-tree, SVM and ensemble models in reproducible Python workflows.
- Reconciled Power BI totals with source records using SQL and Python, and built a Flask inventory site and data pipelines.

**Stack:** PyTorch, Azure ML, Python, SQL, Flask, Power BI

## Reproducible build

Six detectors compared on identical features and a time-ordered split, averaged over 5 independently generated datasets (synthetic data with labelled anomalies, since real inventory data is private):

| model | precision | recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| PyTorch MLP → GRU (two-stage) | 0.857 | 0.752 | **0.794** | **0.872** |
| decision tree | 0.942 | 0.669 | 0.781 | 0.783 |
| regression (residual) | 0.935 | 0.674 | 0.781 | 0.771 |
| PyTorch MLP (stage 1 only) | 0.860 | 0.720 | 0.781 | 0.869 |
| ensemble (RF + GB) | 0.868 | 0.706 | 0.772 | 0.865 |
| SVM (rbf) | 0.737 | 0.645 | 0.684 | 0.711 |

F1 differences among the top models are within seed-to-seed noise; PR-AUC separates the neural/ensemble models (≈0.87) from the tree/regression (≈0.77–0.78), which matters for ordering a review queue. Every model misses "stale" repeated counts (recall ≤ 0.17).

A SQL + Python reconciliation found the dashboard export disagreed with source records in 99 of 126 location-week cells (+28,054 net units), traced to duplicates, corrections summed with originals, and NULL locations; an independent implementation of the fix left a residual of 0. A Flask site provides the review queue (confirm / dismiss). The Azure ML job files are templates that were not run.

Source code, results and full write-up: [`code/inventory-anomaly-poc`](https://github.com/fanisridhar/sridhareguram.github.io/tree/master/code/inventory-anomaly-poc).
