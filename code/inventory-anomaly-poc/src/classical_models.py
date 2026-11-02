"""Classical model families compared on the same features/split."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from features import FEATS

REG_X = ["log_lag1", "log_lag2", "log_lag3", "log_med8", "log_seasonal"]


def fit_regression(tr):
    """Regression approach: predict log(count) from history using NORMAL rows only;
    anomaly score = standardised absolute residual."""
    normal = tr[tr.is_anomaly == 0]
    m = LinearRegression().fit(normal[REG_X], normal["log_count"])
    sd = np.std(normal["log_count"] - m.predict(normal[REG_X]))
    return lambda d: np.abs(d["log_count"] - m.predict(d[REG_X])).to_numpy() / sd


def _clf(model, tr, subsample=None, seed=0):
    d = tr
    if subsample and len(d) > subsample:
        pos, neg = d[d.is_anomaly == 1], d[d.is_anomaly == 0].sample(subsample - int((d.is_anomaly == 1).sum()), random_state=seed)
        d = pd.concat([pos, neg])
    model.fit(d[FEATS], d["is_anomaly"])
    return model


def build(tr, seed=0):
    models = {}
    models["regression (residual)"] = fit_regression(tr)
    dt = _clf(DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=seed), tr)
    models["decision tree"] = lambda d, m=dt: m.predict_proba(d[FEATS])[:, 1]
    svm = _clf(make_pipeline(StandardScaler(), SVC(kernel="rbf", C=3.0, class_weight="balanced")), tr, subsample=9000, seed=seed)
    models["SVM (rbf)"] = lambda d, m=svm: m.decision_function(d[FEATS])
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=seed)
    gb = GradientBoostingClassifier(random_state=seed)
    ens = VotingClassifier([("rf", rf), ("gb", gb)], voting="soft", n_jobs=-1)
    ens.fit(tr[FEATS], tr["is_anomaly"])
    models["ensemble (RF+GB)"] = lambda d, m=ens: m.predict_proba(d[FEATS])[:, 1]
    return models
