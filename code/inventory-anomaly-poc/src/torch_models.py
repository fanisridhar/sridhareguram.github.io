"""Two-stage PyTorch detector:  classification model -> RNN classifier.

Stage 1 (MLP): per-week features -> anomaly probability + 32-d hidden embedding.
Stage 2 (GRU): the last 12 weeks of [stage-1 embedding, stage-1 probability]
               -> final anomaly decision for the latest week.
Stage 2 therefore sees how the series has been behaving (context) in addition
to the single-week verdict.
"""
import numpy as np
import torch
import torch.nn as nn

from features import FEATS

WIN = 12


class Stage1(nn.Module):
    def __init__(self, d_in, h=32):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, h), nn.ReLU())
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        z = self.body(x)
        return self.head(z).squeeze(-1), z


class Stage2(nn.Module):
    def __init__(self, d_in, h=32):
        super().__init__()
        self.gru = nn.GRU(d_in, h, batch_first=True)
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        _, hn = self.gru(x)
        return self.head(hn[-1]).squeeze(-1)


def _train(model, forward, X, y, epochs, lr=2e-3, bs=512, seed=0):
    torch.manual_seed(seed)
    pw = ((len(y) - y.sum()) / y.sum().clamp(min=1)).float().clamp(max=50)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(y)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss_fn(forward(X[idx]), y[idx]).backward()
            opt.step()


def windows(df, emb, prob):
    """Build (N, WIN, d) sequences per row using only that series' prior weeks."""
    d = emb.shape[1] + 1
    feat = np.concatenate([emb, prob[:, None]], axis=1).astype(np.float32)
    out = np.zeros((len(df), WIN, d), dtype=np.float32)
    keys = df[["item_id", "location_id"]].to_numpy()
    pos: dict = {}
    for i, k in enumerate(map(tuple, keys)):
        pos.setdefault(k, []).append(i)
    for idxs in pos.values():          # df is sorted by week within a series
        a = feat[idxs]
        for j, row in enumerate(idxs):
            lo = max(0, j - WIN + 1)
            seq = a[lo:j + 1]
            out[row, WIN - len(seq):] = seq
    return out


def fit_two_stage(tr, epochs1=12, epochs2=12, seed=0):
    mu, sd = tr[FEATS].mean(), tr[FEATS].std() + 1e-6
    norm = lambda d: torch.tensor(((d[FEATS] - mu) / sd).to_numpy(np.float32))
    s1 = Stage1(len(FEATS))
    Xtr, ytr = norm(tr), torch.tensor(tr["is_anomaly"].to_numpy(np.float32))
    _train(s1, lambda x: s1(x)[0], Xtr, ytr, epochs1, seed=seed)
    s1.eval()

    def stage1_out(d):
        with torch.no_grad():
            lg, z = s1(norm(d))
        return z.numpy(), torch.sigmoid(lg).numpy()

    z, p = stage1_out(tr)
    Wtr = torch.tensor(windows(tr, z, p))
    s2 = Stage2(Wtr.shape[2])
    _train(s2, s2, Wtr, ytr, epochs2, seed=seed)
    s2.eval()

    def score_stage1(d):
        return stage1_out(d)[1]

    def score_two_stage(d):
        z, p = stage1_out(d)
        with torch.no_grad():
            return torch.sigmoid(s2(torch.tensor(windows(d, z, p)))).numpy()
    return score_stage1, score_two_stage
