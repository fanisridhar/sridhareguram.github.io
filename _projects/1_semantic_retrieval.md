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
