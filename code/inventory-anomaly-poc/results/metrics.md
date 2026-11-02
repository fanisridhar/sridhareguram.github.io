| model | precision | recall | F1 | PR-AUC | flagged / test rows |
|---|---:|---:|---:|---:|---:|
| PyTorch MLP -> GRU (two-stage) | 0.857 ± 0.079 | 0.752 ± 0.072 | 0.794 ± 0.007 | 0.872 ± 0.022 | 118 / 6720 (1.8%) |
| decision tree | 0.942 ± 0.030 | 0.669 ± 0.036 | 0.781 ± 0.022 | 0.783 ± 0.036 | 93 / 6720 (1.4%) |
| regression (residual) | 0.935 ± 0.023 | 0.674 ± 0.061 | 0.781 ± 0.040 | 0.771 ± 0.031 | 94 / 6720 (1.4%) |
| PyTorch MLP (stage 1 only) | 0.860 ± 0.043 | 0.720 ± 0.060 | 0.781 ± 0.022 | 0.869 ± 0.024 | 111 / 6720 (1.6%) |
| ensemble (RF+GB) | 0.868 ± 0.092 | 0.706 ± 0.057 | 0.772 ± 0.010 | 0.865 ± 0.020 | 109 / 6720 (1.6%) |
| SVM (rbf) | 0.737 ± 0.088 | 0.645 ± 0.057 | 0.684 ± 0.051 | 0.711 ± 0.034 | 116 / 6720 (1.7%) |

Mean ± std over 5 independently generated datasets (seeds [42, 43, 44, 45, 46]); each test set has ~6720 rows and ~131 true anomalies. train weeks 12-77, val 78-89, test 90-103. Threshold per model = max-F1 on validation weeks.
