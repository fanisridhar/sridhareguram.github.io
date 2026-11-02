"""Tiny tracing layer: one Trace per query, one span per pipeline stage.

Traces are appended to a JSONL file so slow queries can be inspected and the
same data can be shipped to an observability backend (e.g. Langfuse) later.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager


class Trace:
    def __init__(self, query_id, text):
        self.query_id, self.text = query_id, text
        self.spans: dict = {}
        self.t0 = time.perf_counter()
        self.total_ms = None

    @contextmanager
    def span(self, name):
        s = time.perf_counter()
        try:
            yield
        finally:
            self.spans[name] = self.spans.get(name, 0.0) + (time.perf_counter() - s) * 1000

    def finish(self):
        self.total_ms = (time.perf_counter() - self.t0) * 1000
        return self

    def to_dict(self):
        return {"query_id": self.query_id, "text": self.text,
                "total_ms": round(self.total_ms, 3),
                "spans_ms": {k: round(v, 3) for k, v in self.spans.items()}}


class TraceLog:
    def __init__(self, path=None):
        self.path, self.rows = path, []

    def add(self, trace: Trace, **extra):
        row = trace.to_dict() | extra
        self.rows.append(row)

    def flush(self):
        if self.path:
            with open(self.path, "w") as f:
                for r in self.rows:
                    f.write(json.dumps(r) + "\n")
