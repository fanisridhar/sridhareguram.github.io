---
layout: page
title: Semantic Retrieval Benchmark and Latency Tuning
description: Mercor | Python, retrieval traces, Langfuse
importance: 1
category: work
---

Evaluated a semantic-retrieval service where relevance and response time both mattered, reproducing slow queries and reviewing retrieval traces.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <table class="table table-sm">
            <thead>
                <tr><th>Metric (fixed 50-query benchmark)</th><th>Before</th><th>After</th></tr>
            </thead>
            <tbody>
                <tr><td>p95 latency</td><td>1.2 s</td><td>900 ms (25% lower)</td></tr>
                <tr><td>Queries with a relevant top-five result</td><td>38 / 50 (76%)</td><td>44 / 50 (88%)</td></tr>
            </tbody>
        </table>
    </div>
</div>

- Measured every change on the same fixed 50-query benchmark, cutting p95 latency from 1.2 s to 900 ms (25%) and raising queries with a relevant top-five result from 38 to 44 of 50 (76% to 88%).
- Documented the queries and measurement method so the comparison could be rerun after later changes.

**Stack:** Python, retrieval traces, Langfuse

## Reproducible build

A self-contained reproduction of the method: a generated 40,000-document corpus, a **fixed 50-query benchmark with labelled relevant documents**, a service with switchable optimisations, and per-stage traces. Measured on that setup (not the Mercor system):

| change (cumulative) | p95 latency | queries with a relevant top-five result |
|---|---:|---:|
| baseline | 119.5 ms | 46 / 50 (92%) |
| + normalised float32 matrix | 84.6 ms | 46 / 50 |
| + `argpartition` top-k | 82.5 ms | 46 / 50 |
| + dict record lookup | 2.2 ms | 46 / 50 |
| + hybrid dense + BM25 fusion | 2.8 ms | **50 / 50 (100%)** |

**What the traces showed:** two stages (a linear-scan record fetch, 58 ms, and re-normalising every vector on each call, 32 ms) caused ~99% of baseline latency; the latency fixes left relevance unchanged, and only the hybrid step improved it. A first benchmark version was saturated at 100%, so I made the queries harder (two concepts, both required) to be able to measure relevance changes.

Source code, results and full write-up: [`code/semantic-retrieval-benchmark`](https://github.com/fanisridhar/sridhareguram.github.io/tree/master/code/semantic-retrieval-benchmark).
