#!/usr/bin/env bash
# End-to-end demo on a machine WITHOUT a GPU: simulated engines A/B + quantisation check.
set -euo pipefail
PY=${PY:-python}
cd "$(dirname "$0")"
$PY bench/make_requests.py
for P in sim-a sim-b; do
  PORT=$([ $P = sim-a ] && echo 8101 || echo 8102)
  $PY sim/mock_server.py --profile $P --port $PORT & PID=$!
  sleep 1.5
  $PY bench/env_capture.py --engine "SIMULATED:$P" --model none --out results/env_$P.json >/dev/null
  $PY bench/loadgen.py --url http://127.0.0.1:$PORT/v1/completions --n 40 --levels 1,4,8,16,32,64,96 --out results/sweep_$P.json
  kill $PID; wait $PID 2>/dev/null || true
done
$PY bench/report.py "sim-a=results/sweep_sim-a.json" "sim-b=results/sweep_sim-b.json" --md results/report.md --png results/latency_vs_load.png
# quantisation-quality tooling check: reference vs lossy variant (3% tokens perturbed)
$PY sim/mock_server.py --profile sim-a --port 8103 & P1=$!
$PY sim/mock_server.py --profile sim-a --port 8104 --quant-flip 0.03 & P2=$!
sleep 1.5
$PY bench/loadgen.py --url http://127.0.0.1:8103/v1/completions --n 40 --levels 1 --out results/q_ref.json --save-text results/q_ref_text.json
$PY bench/loadgen.py --url http://127.0.0.1:8104/v1/completions --n 40 --levels 1 --out results/q_quant.json --save-text results/q_quant_text.json
kill $P1 $P2; wait 2>/dev/null || true
$PY quality/compare_outputs.py results/q_ref_text.json results/q_quant_text.json --out results/quality.json
