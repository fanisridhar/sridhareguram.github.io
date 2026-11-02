Recall by anomaly type, seed-42 test set

| model | spike | drop | zero | x10 | stale |
|---|---:|---:|---:|---:|---:|
| regression (residual) | 0.78 (n=27) | 0.64 (n=22) | 1.00 (n=27) | 1.00 (n=20) | 0.00 (n=24) |
| decision tree | 0.81 (n=27) | 0.73 (n=22) | 0.96 (n=27) | 1.00 (n=20) | 0.00 (n=24) |
| SVM (rbf) | 0.74 (n=27) | 0.82 (n=22) | 0.81 (n=27) | 0.45 (n=20) | 0.08 (n=24) |
| ensemble (RF+GB) | 0.85 (n=27) | 0.73 (n=22) | 1.00 (n=27) | 1.00 (n=20) | 0.17 (n=24) |
| PyTorch MLP (stage 1 only) | 0.85 (n=27) | 0.64 (n=22) | 1.00 (n=27) | 1.00 (n=20) | 0.17 (n=24) |
| PyTorch MLP -> GRU (two-stage) | 0.85 (n=27) | 0.77 (n=22) | 1.00 (n=27) | 1.00 (n=20) | 0.17 (n=24) |
