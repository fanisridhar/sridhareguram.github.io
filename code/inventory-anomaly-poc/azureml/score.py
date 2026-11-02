"""Azure ML online-endpoint scoring script (init/run contract).
Input JSON: {"history": [{"item_id":..,"location_id":..,"week":..,"count":..}, ...]}  (>= 13 weeks per series)
Output: per-row anomaly score + flag for manual review.  Expects model.pkl (a fitted scorer + threshold)
registered with the endpoint.
TEMPLATE ONLY: exporting the chosen model to model.pkl is NOT implemented in this repo (the comparison scorers are closures),
and this script was not deployed or tested against a live Azure ML endpoint."""
import json
import os
import pickle
import sys

import pandas as pd


def init():
    global MODEL, THR
    path = os.path.join(os.getenv("AZUREML_MODEL_DIR", "."), "model.pkl")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    with open(path, "rb") as f:
        MODEL, THR = pickle.load(f)


def run(raw):
    from features import add_features
    df = pd.DataFrame(json.loads(raw)["history"])
    df["is_anomaly"] = 0
    feats = add_features(df)
    feats["score"] = MODEL(feats)
    feats["review"] = feats["score"] >= THR
    return feats[["item_id", "location_id", "week", "count", "score", "review"]].to_dict("records")
