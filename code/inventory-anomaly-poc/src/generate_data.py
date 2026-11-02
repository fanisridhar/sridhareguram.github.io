"""Programmatically generated weekly inventory counts per (item, location) with LABELLED anomalies.

Real inventory data is private, so this generator produces data with the same
structure the POC uses -- historical counts by item and location -- plus ground
truth for the injected anomalies, so detectors can be scored objectively.

Normal behaviour: item/location base level, yearly seasonality, slow trend,
per-series noise (5-20%), and *benign promotion surges* (x1.3-1.8 for 2-4 weeks,
NOT anomalies) that a naive threshold would flag.
Anomalies (~2% of points, label=1): spike (x1.4-4), drop (x0.15-0.7), zero count,
10x unit/decimal-shift error, and "stale" counts (last week's number repeated).
A first version with only large, isolated anomalies let every model score ~100%,
which made the comparison meaningless; this version is deliberately harder.
"""
import argparse
import sqlite3

import numpy as np
import pandas as pd

N_ITEMS, N_LOCS, N_WEEKS = 60, 8, 104


def generate(seed=42, anomaly_rate=0.02) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N_ITEMS):
        for l in range(N_LOCS):
            base = rng.lognormal(np.log(200), 0.7)
            amp = rng.uniform(0.05, 0.25)
            phase = rng.uniform(0, 2 * np.pi)
            trend = rng.normal(0, 0.0015)
            t = np.arange(N_WEEKS)
            clean = base * (1 + amp * np.sin(2 * np.pi * t / 52 + phase)) * (1 + trend * t)
            noise = rng.uniform(0.05, 0.20)
            mult = np.ones(N_WEEKS)
            if rng.random() < 0.5:                      # benign promotion surge (not an anomaly)
                st, ln = rng.integers(14, N_WEEKS - 5), rng.integers(2, 5)
                mult[st:st + ln] = rng.uniform(1.3, 1.8)
            obs = np.maximum(0, np.round(clean * mult * rng.normal(1, noise, N_WEEKS))).astype(int)
            label = np.zeros(N_WEEKS, dtype=int)
            kind = np.array([""] * N_WEEKS, dtype=object)
            for w in range(12, N_WEEKS):
                if rng.random() < anomaly_rate:
                    k = rng.choice(["spike", "drop", "zero", "x10", "stale"])
                    f = {"spike": rng.uniform(1.4, 4.0), "drop": rng.uniform(0.15, 0.7), "zero": 0.0, "x10": 10.0, "stale": 1.0}[k]
                    obs[w] = int(obs[w - 1]) if k == "stale" else int(round(obs[w] * f))
                    label[w], kind[w] = 1, k
            rows.append(pd.DataFrame({"item_id": i, "location_id": l, "week": t, "count": obs,
                                      "is_anomaly": label, "anomaly_kind": kind}))
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/inventory_counts.csv")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df = generate(a.seed)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} rows, {df.is_anomaly.sum()} anomalies ({100*df.is_anomaly.mean():.2f}%) -> {a.out}")
