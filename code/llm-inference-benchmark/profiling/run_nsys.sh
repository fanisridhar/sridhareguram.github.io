#!/usr/bin/env bash
# Profile an inference server under load with Nsight Systems (needs a GPU box).
# usage: profiling/run_nsys.sh <tag> "<server launch command>" <concurrency>
set -euo pipefail
TAG=$1; SERVER_CMD=$2; CONC=${3:-32}
mkdir -p results/nsys
nsys profile -o results/nsys/$TAG --trace=cuda,nvtx,osrt --sample=none \
     --capture-range=cudaProfilerApi --capture-range-end=stop -f true \
     bash -c "$SERVER_CMD" &
SERVER_PID=$!
sleep 90                                   # wait for model load + warm-up
python bench/loadgen.py --levels "$CONC" --out results/nsys/$TAG.load.json
kill $SERVER_PID || true; wait $SERVER_PID 2>/dev/null || true
nsys stats --report cuda_gpu_kern_sum --format csv results/nsys/$TAG.nsys-rep > results/nsys/$TAG.kern.csv
nsys stats --report cuda_gpu_trace   --format csv results/nsys/$TAG.nsys-rep > results/nsys/$TAG.trace.csv
python profiling/parse_nsys.py results/nsys/$TAG.kern.csv results/nsys/$TAG.trace.csv
