# LLM Inference Benchmarking: vLLM vs TensorRT-LLM

vLLM · TensorRT-LLM · CUDA · Nsight Systems · Python (asyncio, aiohttp)

## Abstract
A benchmarking harness that sends **identical requests** to any OpenAI-compatible LLM server (vLLM `vllm serve`, TensorRT-LLM
`trtllm-serve`), sweeps concurrency to find where **latency starts to rise under load**, profiles the GPU with **CUDA/Nsight
traces** to explain *why*, checks whether **quantization changed output quality**, and records every setting so a run can be
**re-checked**.

> **Read this first.** The build environment had **no GPU** and no model access, so the harness was validated end-to-end
> against a **simulated continuous-batching server** (`sim/mock_server.py`). **All numbers below are from the simulator.
> They say nothing about real vLLM or TensorRT-LLM performance.** The harness, metrics code, profiling scripts and
> quality-comparison tool are the real deliverable: point `--url` at real servers on a GPU machine to get real numbers.

## Requirements
1. Same prompts, same sampling (temperature 0), same token budgets for every engine -> a fixed, hashed request file.
2. Load sweep (closed loop, concurrency 1->96) with TTFT, TPOT, end-to-end latency (p50/p95/p99), throughput, error count.
3. A way to see GPU behaviour under load (kernel time share, GPU idle gaps) -> Nsight Systems wrapper + parser.
4. A quality check for quantization (greedy outputs, reference vs quantized).
5. Everything needed to repeat the run: engine versions, args, GPU/driver/CUDA, git commit, request-file hash.

## Goal
Explain *where* latency begins rising with load, compare two engines fairly, and decide whether a quantized variant is
acceptable on quality - with results that someone else can reproduce.

## Work done
| Piece | File |
|---|---|
| Seeded, hashed request set (32-1024 word prompts, 32-128 output tokens) | `bench/make_requests.py` |
| Async streaming load generator (TTFT/TPOT/E2E, concurrency sweep) | `bench/loadgen.py` |
| Environment/config capture incl. request-set sha256 | `bench/env_capture.py` |
| Sweep comparison table, latency "knee", plot | `bench/report.py` |
| Nsight Systems run wrapper + CSV parser (top kernels, GPU idle %) | `profiling/run_nsys.sh`, `profiling/parse_nsys.py` |
| Quantization output comparison | `quality/compare_outputs.py` |
| Simulated continuous-batching OpenAI-compatible server | `sim/mock_server.py` |
| One-command demo without a GPU | `run_sim.sh` |

### Running against real engines (needs a GPU)
```bash
vllm serve <model> --max-num-seqs 32 --port 8000 &                 # or: trtllm-serve <engine_dir> --port 8000
python bench/make_requests.py
python bench/env_capture.py --engine vllm-<ver> --model <model> --quant none --out results/env_vllm.json
python bench/loadgen.py --model <model> --levels 1,4,8,16,32,64 --out results/sweep_vllm.json --save-text results/text_ref.json
# quantized variant (same requests):  ... --save-text results/text_q.json ; python quality/compare_outputs.py results/text_ref.json results/text_q.json
profiling/run_nsys.sh vllm-c32 "vllm serve <model> --port 8000" 32 # then profiling/parse_nsys.py prints top kernels + GPU idle %
python bench/report.py vllm=results/sweep_vllm.json trtllm=results/sweep_trt.json --md results/report.md --png results/latency_vs_load.png
```

## Results (SIMULATED - `results/report.md`, `results/latency_vs_load.png`)
Simulated engines `sim-a` and `sim-b` have different invented parameters (decode step cost, prefill cost); 32 decode slots each;
40 requests per concurrency level.

**sim-a** (latency knee at concurrency 32)

| conc | TTFT p50 ms | TTFT p95 ms | TPOT p95 ms | E2E p95 ms | tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 18 | 33 | 10.5 | 1340 | 95 |
| 4 | 30 | 47 | 12.8 | 1628 | 296 |
| 8 | 50 | 66 | 15.4 | 1940 | 482 |
| 16 | 89 | 130 | 21.0 | 2492 | 647 |
| 32 | 244 | 252 | 28.6 | 3153 | 839 |
| 64 | 253 | 1190 | 28.4 | 3518 | 842 |

**sim-b**: E2E p95 946 ms at concurrency 1 -> 2319 ms at 32; throughput saturates at ~1145 tok/s (full table in `results/report.md`).

**Quantization-check tool validation** (simulated "quantized" server perturbing 3% of tokens, 40 requests):
exact match 15%, mean token similarity 0.968, first divergence at ~46% of the way through the output, mean length change 0.

## Findings (what the simulation demonstrates - mechanism, not engine facts)
- **Latency rises well before throughput saturates.** E2E p95 grows from 1.34 s (c=1) to 3.15 s (c=32) while throughput
  climbs from 95 to 839 tok/s: batching raises throughput but every token gets slower (TPOT 10.5 -> 28.6 ms, because each
  decode step costs more with a bigger batch).
- **Past the slot limit, requests queue.** TTFT p95 jumps from 252 ms (c=32) to 1190 ms (c=64) while throughput is flat
  (839 -> 842 tok/s): extra concurrency only adds waiting. (The jump appears at c=64 because the 40-request level never
  oversubscribes 32 slots before then.)
- **Small output differences accumulate.** A 3% token perturbation leaves only 15% of outputs identical, though 96.8% similar:
  *exact match alone overstates quality loss* - the tool therefore also reports similarity and first-divergence position.
- **To explain real-engine latency**, `parse_nsys.py` reports top kernels by GPU time and the fraction of the capture window
  the GPU sat idle: high idle % under load points to scheduling/launch/queueing overhead, high kernel share points to compute
  or memory bandwidth.

## Honest limitations
- No real vLLM / TensorRT-LLM / CUDA / Nsight numbers exist in this repo. `profiling/*` and the real-engine commands were
  **not run** (no GPU). Everything marked simulated is simulated.
- The simulator's parameters are invented; do not read `sim-a` vs `sim-b` as a vLLM vs TensorRT-LLM result.
- Concurrency levels above 40 only reach 40 in flight because each level sends 40 requests.
- The quality tool compares generated text only; perplexity/logprob and task-accuracy checks are the next step on real hardware.

## Reproduce the simulated demo
```bash
pip install aiohttp numpy matplotlib
./run_sim.sh      # ~3 min
```
