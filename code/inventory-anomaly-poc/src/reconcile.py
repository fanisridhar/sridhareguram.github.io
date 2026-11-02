"""Reconcile dashboard (Power BI-style) totals against source records with SQL + Python.

Source of truth : `source_records` -- raw transaction-level count submissions
                  (can contain re-submitted duplicates, NULL locations, and
                  late corrections that supersede earlier entries).
Dashboard export: a CSV of totals per (location, week) as a BI report would show
                  after a naive import (the flaws below are INJECTED so the
                  reconciliation has something real to find).

Method: aggregate the source in SQL *correctly* (latest submission per
item/location/week, NULL location bucketed as 'UNASSIGNED'), diff against the
dashboard in pandas, classify every mismatch by root cause, then re-aggregate
with the fix and show zero residual difference.
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))


def build_source(con, seed=5):
    rng = np.random.default_rng(seed)
    base = pd.read_sql("SELECT item_id, location_id, week, count FROM counts WHERE week >= 90", con)
    base["submitted_at"] = base["week"] * 7 * 24 + 10
    base["source_system"] = "scanner"
    # (1) duplicate re-submissions of the same record (same value) ~1.5%
    dups = base.sample(frac=0.015, random_state=1).copy()
    dups["submitted_at"] += 1
    # (2) late corrections: value changed after first submission ~1%
    corr = base.sample(frac=0.01, random_state=2).copy()
    corr["count"] = (corr["count"] * rng.uniform(0.8, 1.2, len(corr))).round().astype(int)
    corr["submitted_at"] += 48
    corr["source_system"] = "manual_correction"
    # (3) records with a missing location ~0.7%
    src = pd.concat([base, dups, corr], ignore_index=True)
    nulls = src.sample(frac=0.007, random_state=3).index
    src["location_id"] = src["location_id"].astype("Float64")
    src.loc[nulls, "location_id"] = pd.NA
    src.to_sql("source_records", con, index=False, if_exists="replace")
    return src


CORRECT_SQL = """
WITH latest AS (
  SELECT item_id, COALESCE(location_id, -1) AS location_id, week, count,
         ROW_NUMBER() OVER (PARTITION BY item_id, COALESCE(location_id, -1), week
                            ORDER BY submitted_at DESC) AS rn
  FROM source_records)
SELECT location_id, week, SUM(count) AS total FROM latest WHERE rn = 1 GROUP BY location_id, week
"""

NAIVE_DASHBOARD_SQL = """   -- what the flawed report effectively did
SELECT location_id, week, SUM(count) AS total FROM source_records
WHERE location_id IS NOT NULL GROUP BY location_id, week
"""


def main():
    db = ROOT / "data" / "inventory.db"
    con = sqlite3.connect(db)
    src = build_source(con)
    dash = pd.read_sql(NAIVE_DASHBOARD_SQL, con)
    dash.to_csv(ROOT / "results" / "dashboard_export.csv", index=False)       # the "Power BI" numbers
    truth = pd.read_sql(CORRECT_SQL, con)

    cmp = truth.merge(dash, on=["location_id", "week"], how="outer", suffixes=("_source", "_dashboard")).fillna(0)
    cmp["diff"] = cmp["total_dashboard"] - cmp["total_source"]
    bad = cmp[cmp["diff"] != 0].copy()

    # root-cause classification: which defect types exist in the source rows behind each mismatched cell
    # (cell-level labelling; the magnitude of each cause inside a mixed cell is not separated)
    def cause(row):
        loc, wk = row["location_id"], row["week"]
        if loc == -1:
            return "NULL location dropped from dashboard"
        sub = src[(src.location_id == loc) & (src.week == wk)]
        has_dup = sub.duplicated(["item_id", "week", "count"], keep=False).any()
        has_corr = (sub.source_system == "manual_correction").any()
        if has_corr and has_dup:
            return "duplicates + corrections both summed"
        if has_corr:
            return "correction double-counted (original + corrected summed)"
        if has_dup:
            return "duplicate submission summed twice"
        return "other"
    bad["root_cause"] = bad.apply(cause, axis=1)

    report = {
        "cells_compared": int(len(cmp)), "cells_mismatched": int(len(bad)),
        "mismatch_rate_pct": round(100 * len(bad) / len(cmp), 2),
        "total_source": int(cmp.total_source.sum()), "total_dashboard": int(cmp.total_dashboard.sum()),
        "net_abs_diff": int(cmp["diff"].sum()),
        "gross_abs_diff": int(cmp["diff"].abs().sum()),
        "by_root_cause": bad.groupby("root_cause")["diff"].agg(cells="count", net_diff="sum").reset_index().to_dict("records"),
    }
    # Verify the fix independently: re-implement the corrected rule in pandas (different code path
    # from the SQL window function) and require it to match the SQL aggregate exactly.
    fixed = (src.sort_values("submitted_at")
                .assign(location_id=lambda d: d.location_id.astype("float").fillna(-1))
                .drop_duplicates(["item_id", "location_id", "week"], keep="last")
                .groupby(["location_id", "week"], as_index=False)["count"].sum()
                .rename(columns={"count": "total"}))
    chk = fixed.merge(truth, on=["location_id", "week"], how="outer", suffixes=("_pandas", "_sql")).fillna(0)
    report["residual_after_fix"] = int((chk.total_pandas - chk.total_sql).abs().sum())
    import json
    (ROOT / "results" / "reconciliation.json").write_text(json.dumps(report, indent=1))
    bad.to_csv(ROOT / "results" / "reconciliation_mismatches.csv", index=False)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
