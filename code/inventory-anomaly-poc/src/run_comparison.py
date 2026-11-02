"""Train every model on the TRAIN weeks, pick each model's alert threshold on the
VALIDATION weeks (max F1), and report precision/recall/F1/PR-AUC on the held-out
TEST weeks. Same data, same features, same time-ordered split for everyone.
Also writes scores for the best model to SQLite for the Flask review site."""
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

sys.path.insert(0, str(Path(__file__).parent))
from classical_models import build
from features import add_features
from generate_data import generate
from torch_models import fit_two_stage

ROOT = Path(__file__).resolve().parent.parent


def best_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    return float(t[np.argmax(f1[:-1])])


def metrics(y, s, thr):
    pred = s >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-12),
            "pr_auc": float(average_precision_score(y, s)), "flagged": int(pred.sum()),
            "flag_rate_pct": 100 * float(pred.mean()), "tp": tp, "fp": fp, "fn": fn}


SEEDS = [42, 43, 44, 45, 46]


def run_seed(seed):
    raw = generate(seed=seed)
    df = add_features(raw)
    tr, va, te = (df[df.split == k] for k in ("train", "val", "test"))
    scorers = build(tr, seed=seed)
    s1, s2 = fit_two_stage(tr, seed=seed)
    scorers["PyTorch MLP (stage 1 only)"] = s1
    scorers["PyTorch MLP -> GRU (two-stage)"] = s2
    results, all_scores = [], {}
    for name, fn in scorers.items():
        full = np.asarray(fn(df), dtype=float)          # score the full timeline (causal windows)
        all_scores[name] = full
        sv, st = full[(df.split == "val").to_numpy()], full[(df.split == "test").to_numpy()]
        thr = best_threshold(va.is_anomaly.to_numpy(), sv)
        results.append(metrics(te.is_anomaly.to_numpy(), st, thr) | {"model": name, "threshold": thr, "seed": seed})
    info = {"seed": seed, "train": len(tr), "val": len(va), "test": len(te), "test_anomalies": int(te.is_anomaly.sum()),
            "train_anomalies": int(tr.is_anomaly.sum())}
    return raw, df, results, all_scores, info


def main():
    runs, art = [], {}
    for sd in SEEDS:
        raw, df, results, scores, info = run_seed(sd)
        runs.append((results, info))
        print(f"seed {sd}: test rows={info['test']} anomalies={info['test_anomalies']} | " +
              " ".join(f"{m['model'].split('(')[0].strip()[:10]}={m['f1']:.2f}" for m in results))
        if sd == SEEDS[0]:
            art = {"raw": raw, "df": df, "results": results, "scores": scores}
    models = [m["model"] for m in runs[0][0]]
    agg = []
    for name in models:
        rows = [m for res, _ in runs for m in res if m["model"] == name]
        a = {"model": name}
        for k in ("precision", "recall", "f1", "pr_auc", "flagged"):
            v = np.array([r[k] for r in rows], dtype=float)
            a[k], a[k + "_std"] = float(v.mean()), float(v.std())
        agg.append(a)
    (ROOT / "results").mkdir(exist_ok=True)
    meta = {"seeds": SEEDS, "split": "train weeks 12-77, val 78-89, test 90-103",
            "per_seed": [i for _, i in runs]}
    json.dump({"meta": meta, "aggregate": agg, "per_seed_results": [r for res, _ in runs for r in res]},
              open(ROOT / "results" / "metrics.json", "w"), indent=1)
    n_test = int(np.mean([i["test"] for _, i in runs])); n_pos = np.mean([i["test_anomalies"] for _, i in runs])
    md = ["| model | precision | recall | F1 | PR-AUC | flagged / test rows |", "|---|---:|---:|---:|---:|---:|"]
    for a in sorted(agg, key=lambda a: -a["f1"]):
        md.append(f"| {a['model']} | {a['precision']:.3f} ± {a['precision_std']:.3f} | {a['recall']:.3f} ± {a['recall_std']:.3f} | "
                  f"{a['f1']:.3f} ± {a['f1_std']:.3f} | {a['pr_auc']:.3f} ± {a['pr_auc_std']:.3f} | "
                  f"{a['flagged']:.0f} / {n_test} ({100*a['flagged']/n_test:.1f}%) |")
    md.append(f"\nMean ± std over {len(SEEDS)} independently generated datasets (seeds {SEEDS}); each test set has ~{n_test} rows and "
              f"~{n_pos:.0f} true anomalies. {meta['split']}. Threshold per model = max-F1 on validation weeks.")
    (ROOT / "results" / "metrics.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # ---- error analysis (seed 42): which anomaly types does each model catch? ----
    d0 = art["df"]
    te_mask = (d0.split == "test").to_numpy()
    kinds = ["spike", "drop", "zero", "x10", "stale"]
    thr = {m["model"]: m["threshold"] for m in art["results"]}
    lines = ["| model | " + " | ".join(kinds) + " |", "|---|" + "---:|" * len(kinds)]
    for name, sc_ in art["scores"].items():
        row = []
        for k in kinds:
            sel = te_mask & (d0.anomaly_kind == k).to_numpy()
            row.append(f"{(sc_[sel] >= thr[name]).mean():.2f} (n={int(sel.sum())})" if sel.sum() else "-")
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    (ROOT / "results" / "recall_by_anomaly_type.md").write_text("Recall by anomaly type, seed-42 test set\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))

    best = max(agg, key=lambda a: a["f1"])["model"]
    bm = next(m for m in art["results"] if m["model"] == best)
    print("best by mean F1:", best)
    # ---- persist seed-42 scores of the best model for the Flask review site ----
    df, raw = art["df"], art["raw"]
    db = ROOT / "data" / "inventory.db"
    db.unlink(missing_ok=True)
    con = sqlite3.connect(db)
    raw.drop(columns=["anomaly_kind"]).to_sql("counts", con, index=False)
    sc = df[["item_id", "location_id", "week"]].copy()
    sc["score"] = art["scores"][best]
    sc["flagged"] = (sc["score"] >= bm["threshold"]).astype(int)
    sc["model"] = best
    sc["review_status"] = "pending"
    sc[sc.flagged == 1].query("week >= 90").to_sql("flags", con, index=False)
    con.close()


if __name__ == "__main__":
    main()
