| config | p50 ms | p95 ms | hit@5 | MRR@5 |
|---|---:|---:|---:|---:|
| 0 baseline | 91.46 | 119.46 | 46/50 (92%) | 0.813 |
| 1 +precomputed f32 norms | 52.91 | 84.63 | 46/50 (92%) | 0.813 |
| 2 +argpartition top-k | 54.81 | 82.51 | 46/50 (92%) | 0.813 |
| 3 +dict record lookup | 1.83 | 2.16 | 46/50 (92%) | 0.813 |
| 4 +hybrid BM25 fusion | 2.32 | 2.84 | 50/50 (100%) | 0.945 |

p95: 119.46 ms -> 2.84 ms (98% lower)
hit@5: 46/50 -> 50/50

Per-stage mean latency (ms):

| config | embed | fetch | fuse | lexical | score | topk |
|---|---:|---:|---:|---:|---:|---:|
| 0 baseline | 2.48 | 58.05 | 0.00 | 0.00 | 31.76 | 0.96 |
| 1 +precomputed f32 norms | 2.49 | 55.54 | 0.00 | 0.00 | 0.78 | 0.87 |
| 2 +argpartition top-k | 2.50 | 56.48 | 0.00 | 0.00 | 0.85 | 0.25 |
| 3 +dict record lookup | 1.25 | 0.04 | 0.00 | 0.00 | 0.31 | 0.16 |
| 4 +hybrid BM25 fusion | 1.18 | 0.03 | 0.23 | 0.46 | 0.49 | 0.17 |
