"""Leakage-safe features: for week t only counts up to week t are used.
Robust statistics (median/MAD) are used because past anomalies contaminate means."""
import numpy as np
import pandas as pd

FEATS = ["log_count", "log_lag1", "log_lag2", "log_lag3", "log_med8", "log_ratio_med", "robust_z",
         "pct_chg_lag1", "log_seasonal", "log_ratio_seasonal", "is_zero", "same_as_lag1"]
TRAIN_END, VAL_END = 78, 90   # train weeks <78, val 78..89, test 90..103 (time-ordered split)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["item_id", "location_id", "week"]).copy()
    g = df.groupby(["item_id", "location_id"])["count"]
    L = lambda s: np.log1p(s)
    df["log_count"] = L(df["count"])
    for k in (1, 2, 3):
        df[f"log_lag{k}"] = L(g.shift(k))
    prior = g.shift(1)
    roll = prior.groupby([df["item_id"], df["location_id"]]).rolling(8, min_periods=3)
    med = roll.median().reset_index(level=[0, 1], drop=True)
    mad = roll.apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True).reset_index(level=[0, 1], drop=True)
    df["log_med8"] = L(med)
    df["log_ratio_med"] = df["log_count"] - df["log_med8"]
    df["robust_z"] = ((df["count"] - med) / (1.4826 * mad + 1.0)).clip(-50, 50)
    df["pct_chg_lag1"] = ((df["count"] - prior) / (prior + 1.0)).clip(-20, 20)
    seas = g.shift(52)
    df["log_seasonal"] = L(seas).fillna(df["log_med8"])
    df["log_ratio_seasonal"] = df["log_count"] - df["log_seasonal"]
    df["is_zero"] = (df["count"] == 0).astype(int)
    df["same_as_lag1"] = (df["count"] == prior).astype(int)
    df = df[df["week"] >= 12].copy()
    df[FEATS] = df[FEATS].fillna(0.0)
    df["split"] = np.where(df.week < TRAIN_END, "train", np.where(df.week < VAL_END, "val", "test"))
    return df.reset_index(drop=True)
