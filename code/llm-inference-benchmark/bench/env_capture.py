"""Record everything needed to re-check a run: engine+version, model, quant,
GPU/driver/CUDA, git commit, and the sha256 of the request set."""
import argparse
import hashlib
import json
import platform
import subprocess


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default="none")
    ap.add_argument("--server-args", default="")
    ap.add_argument("--requests", default="results/requests.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    info = {
        "engine": a.engine, "model": a.model, "quantization": a.quant, "server_args": a.server_args,
        "requests_sha256": hashlib.sha256(open(a.requests, "rb").read()).hexdigest(),
        "gpu": sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"),
        "cuda": sh("nvcc --version | tail -1"),
        "git_commit": sh("git rev-parse --short HEAD"),
        "python": platform.python_version(), "platform": platform.platform(),
        "pip_freeze_subset": sh("pip freeze | grep -iE 'vllm|tensorrt|torch|transformers'"),
    }
    json.dump(info, open(a.out, "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
