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
