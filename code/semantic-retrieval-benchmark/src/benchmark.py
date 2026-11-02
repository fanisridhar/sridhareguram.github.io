"""Run the fixed 50-query benchmark against each configuration.

Every configuration sees the SAME corpus, SAME 50 queries, SAME number of
timed repetitions, SAME warm-up. Reported: p50/p95 latency, hit@5 (a relevant
doc in the top five), MRR@5, and a per-stage latency breakdown from the traces.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from corpus import build
from service import Config, RetrievalService
from tracer import Trace, TraceLog

ROOT = Path(__file__).resolve().parent.parent
REPS = 8
WARMUP = 1

CONFIGS = [
    Config("0 baseline"),
    Config("1 +precomputed f32 norms", precomputed_norm_f32=True),
    Config("2 +argpartition top-k", precomputed_norm_f32=True, partial_topk=True),
    Config("3 +dict record lookup", precomputed_norm_f32=True, partial_topk=True, dict_lookup=True),
    Config("4 +hybrid BM25 fusion", precomputed_norm_f32=True, partial_topk=True, dict_lookup=True, hybrid=True),
]


def evaluate(svc, corpus, cfg, log=None):
    lat, hits, rr = [], 0, 0.0
    stages: dict = {}
    for qi, q in enumerate(corpus.queries):
        for _ in range(WARMUP):
            svc.search(q["text"], cfg)
        per_q = []
        top = None
        for rep in range(REPS):
            tr = Trace(qi, q["text"])
            top, tr = svc.search(q["text"], cfg, tr)
            per_q.append(tr)
            lat.append(tr.total_ms)
            for k, v in tr.spans.items():
                stages.setdefault(k, []).append(v)
        rel = corpus.relevant[qi]
        ranks = [r for r, d in enumerate(top, 1) if d in rel]
        hit = bool(ranks)
        hits += hit
        rr += 1 / ranks[0] if ranks else 0.0
        if log:
            log.add(per_q[-1], config=cfg.name, hit_at_5=hit)
    lat = np.array(lat)
    return {
        "config": cfg.name,
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "mean_ms": float(lat.mean()),
        "hit_at_5": hits,
        "hit_at_5_pct": 100 * hits / len(corpus.queries),
        "mrr_at_5": rr / len(corpus.queries),
        "stage_mean_ms": {k: float(np.mean(v)) for k, v in stages.items()},
    }


def main():
    t0 = time.time()
    corpus = build()
    svc = RetrievalService(corpus.docs)
    print(f"corpus={len(corpus.docs)} docs, queries={len(corpus.queries)}, dim={svc.E32.shape[1]}, "
          f"built in {time.time()-t0:.1f}s")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "queries.json").write_text(json.dumps(
        [{"id": i, **q, "n_relevant": len(r)} for i, (q, r) in enumerate(zip(corpus.queries, corpus.relevant))], indent=1))
    log = TraceLog(ROOT / "results" / "traces.jsonl")
    rows = []
    for cfg in CONFIGS:
        r = evaluate(svc, corpus, cfg, log)
        rows.append(r)
        print(f"{r['config']:<32} p50={r['p50_ms']:7.2f}ms p95={r['p95_ms']:7.2f}ms "
              f"hit@5={r['hit_at_5']}/50 ({r['hit_at_5_pct']:.0f}%) mrr@5={r['mrr_at_5']:.3f}")
    log.flush()
    env = {"python": platform.python_version(), "machine": platform.machine(), "cpu": platform.processor() or "n/a",
           "n_docs": len(corpus.docs), "n_queries": len(corpus.queries), "reps": REPS, "warmup": WARMUP}
    (ROOT / "results" / "results.json").write_text(json.dumps({"env": env, "rows": rows}, indent=1))

    base, best = rows[0], rows[-1]
    md = ["| config | p50 ms | p95 ms | hit@5 | MRR@5 |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['config']} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['hit_at_5']}/50 ({r['hit_at_5_pct']:.0f}%) | {r['mrr_at_5']:.3f} |")
    md += ["", f"p95: {base['p95_ms']:.2f} ms -> {best['p95_ms']:.2f} ms "
               f"({100*(base['p95_ms']-best['p95_ms'])/base['p95_ms']:.0f}% lower)",
           f"hit@5: {base['hit_at_5']}/50 -> {best['hit_at_5']}/50", "",
           "Per-stage mean latency (ms):", "", "| config | " + " | ".join(sorted({k for r in rows for k in r['stage_mean_ms']})) + " |"]
    stage_names = sorted({k for r in rows for k in r['stage_mean_ms']})
    md.append("|---|" + "---:|" * len(stage_names))
    for r in rows:
        md.append(f"| {r['config']} | " + " | ".join(f"{r['stage_mean_ms'].get(s, 0):.2f}" for s in stage_names) + " |")
    (ROOT / "results" / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
