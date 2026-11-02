"""Async load generator for any OpenAI-compatible /v1/completions endpoint.

vLLM (`vllm serve`) and TensorRT-LLM (`trtllm-serve`) both expose this API, so
the SAME client drives both. Closed-loop load: `concurrency` workers each send
the next request as soon as the previous one finishes, sweeping concurrency to
find where latency starts rising.

Metrics per request: TTFT (time to first token), TPOT (mean time per output
token after the first), E2E latency, output token count. Per level: p50/p95/p99
of each, request throughput and output-token throughput, error rate.
"""
import argparse
import asyncio
import json
import time

import aiohttp
import numpy as np


async def one_request(session, url, model, req, timeout_s):
    body = {"model": model, "prompt": req["prompt"], "max_tokens": req["max_tokens"],
            "temperature": req["temperature"], "stream": True}
    t0 = time.perf_counter()
    ttft, n, text, stamps = None, 0, [], []
    try:
        async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            if resp.status != 200:
                return {"id": req["id"], "ok": False, "err": f"http {resp.status}"}
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                piece = chunk["choices"][0].get("text", "")
                if piece:
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    n += 1
                    stamps.append(now)
                    text.append(piece)
    except Exception as e:  # timeouts, resets
        return {"id": req["id"], "ok": False, "err": type(e).__name__}
    e2e = time.perf_counter() - t0
    tpot = (stamps[-1] - stamps[0]) / (n - 1) if n > 1 else 0.0
    return {"id": req["id"], "ok": True, "ttft_s": ttft, "tpot_s": tpot, "e2e_s": e2e,
            "out_tokens": n, "text": "".join(text)}


async def run_level(url, model, reqs, concurrency, timeout_s):
    queue = asyncio.Queue()
    for r in reqs:
        queue.put_nowait(r)
    results = []
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as session:
        async def worker():
            while True:
                try:
                    r = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                results.append(await one_request(session, url, model, r, timeout_s))
        t0 = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(concurrency)])
        wall = time.perf_counter() - t0
    return results, wall


def summarise(results, wall, concurrency):
    ok = [r for r in results if r["ok"]]
    def pct(key, p):
        return float(np.percentile([r[key] for r in ok], p)) if ok else None
    toks = sum(r["out_tokens"] for r in ok)
    return {"concurrency": concurrency, "n": len(results), "errors": len(results) - len(ok),
            "wall_s": wall, "req_per_s": len(ok) / wall, "out_tok_per_s": toks / wall,
            **{f"{k}_p{p}_ms": (pct(k + "_s", p) * 1000 if pct(k + "_s", p) is not None else None)
               for k in ("ttft", "tpot", "e2e") for p in (50, 95, 99)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    ap.add_argument("--model", default="model")
    ap.add_argument("--requests", default="results/requests.jsonl")
    ap.add_argument("--n", type=int, default=80, help="requests per concurrency level")
    ap.add_argument("--levels", default="1,2,4,8,16,32,64")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-text", help="write generated text per request (for quality comparison)")
    a = ap.parse_args()
    reqs = [json.loads(l) for l in open(a.requests)][: a.n]
    sweep, texts = [], {}
    for c in [int(x) for x in a.levels.split(",")]:
        results, wall = asyncio.run(run_level(a.url, a.model, reqs, c, a.timeout))
        s = summarise(results, wall, c)
        sweep.append(s)
        if c == 1:
            texts = {r["id"]: r.get("text", "") for r in results if r["ok"]}
        print(f"c={c:<3} ttft_p95={s['ttft_p95_ms']:.0f}ms e2e_p95={s['e2e_p95_ms']:.0f}ms "
              f"tok/s={s['out_tok_per_s']:.0f} err={s['errors']}")
    json.dump({"url": a.url, "model": a.model, "sweep": sweep}, open(a.out, "w"), indent=1)
    if a.save_text:
        json.dump(texts, open(a.save_text, "w"))


if __name__ == "__main__":
    main()
