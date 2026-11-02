---
layout: page
title: "LLM Inference Benchmarking: vLLM vs TensorRT-LLM"
description: Mercor | vLLM, TensorRT-LLM, CUDA, Nsight, Python
importance: 2
category: work
---

Compared two LLM serving stacks on the same workload to understand why latency rose under load.

- Ran identical requests through vLLM and TensorRT-LLM to investigate rising latency under load.
- Profiled GPU behavior with CUDA traces and Nsight, and checked whether quantization changed output quality while reducing resource use.
- Recorded the request set and configuration so results could be re-checked.

**Stack:** vLLM, TensorRT-LLM, CUDA, Nsight, Python

## Reproducible build

A benchmarking harness for any OpenAI-compatible server (works with `vllm serve` and `trtllm-serve`): a fixed, hashed request set; an async streaming load generator reporting TTFT, TPOT and end-to-end p50/p95/p99 across a concurrency sweep; Nsight Systems wrapper and parser (top kernels, GPU idle %); a quantization output-comparison tool; and an environment capture that records engine versions, GPU/CUDA, git commit and the request-set hash.

**Important:** the build environment had no GPU, so the harness was validated against a **simulated** continuous-batching server. The numbers below are simulation output that demonstrate the *mechanism*, not vLLM or TensorRT-LLM results.

| simulated engine A, concurrency | TTFT p95 | E2E p95 | tokens/s |
|---:|---:|---:|---:|
| 1 | 33 ms | 1340 ms | 95 |
| 8 | 66 ms | 1940 ms | 482 |
| 32 | 252 ms | 3153 ms | 839 |
| 64 | 1190 ms | 3518 ms | 842 |

Latency rises well before throughput saturates (bigger batches slow every token), and past the slot limit extra concurrency only adds queueing (TTFT 252 → 1190 ms with flat throughput). The quality tool, run on a simulated 3%-perturbed variant, shows why exact match alone overstates loss (15% identical outputs, 96.8% token similarity).

Source code, results and full write-up: [`code/llm-inference-benchmark`](https://github.com/fanisridhar/sridhareguram.github.io/tree/master/code/llm-inference-benchmark).
