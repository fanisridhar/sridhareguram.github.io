"""Interview-sized snippets you can write from memory and talk through.

Each section is short on purpose. test_snippets.py checks every one of them against
PyTorch's own implementation or a hand-computed answer.
"""
import asyncio
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------
# 1. Softmax and cross-entropy (stable)
# --------------------------------------------------------------------------------
def softmax(z, dim=-1):
    z = z - z.max(dim=dim, keepdim=True).values          # shift: same result, no overflow
    e = z.exp()
    return e / e.sum(dim=dim, keepdim=True)


def cross_entropy(logits, targets, ignore_index=-100):
    """Mean negative log-likelihood of the target class. logits (N, C), targets (N,)."""
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)   # log-softmax
    keep = targets != ignore_index
    t = targets.clamp(min=0)                                         # safe index for ignored rows
    nll = -logp.gather(1, t[:, None]).squeeze(1)
    return (nll * keep).sum() / keep.sum().clamp(min=1)


# --------------------------------------------------------------------------------
# 2. RMSNorm
# --------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# --------------------------------------------------------------------------------
# 3. Scaled dot-product attention (with causal and padding masks)
# --------------------------------------------------------------------------------
def sdpa(q, k, v, causal=False, mask=None):
    """q (..., Tq, d), k/v (..., Tk, d). mask: bool, True = may attend.
    Causal alignment is bottom-right, so a single new query (Tq=1) sees all Tk keys,
    which is exactly what a KV-cache decode step needs."""
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    Tq, Tk = scores.shape[-2:]
    if causal:
        allowed = torch.ones(Tq, Tk, dtype=torch.bool, device=q.device).tril(diagonal=Tk - Tq)
        scores = scores.masked_fill(~allowed, float("-inf"))
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    w = scores.softmax(dim=-1)
    return w @ v, w


# --------------------------------------------------------------------------------
# 4. LoRA
# --------------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B (A x). W is frozen; only A (r x in) and B (out x r) train.
    B starts at zero, so the adapted layer equals the base layer at step 0."""

    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * (self.drop(x) @ self.A.T) @ self.B.T

    @torch.no_grad()
    def merge(self):
        """Fold the adapter into W for zero-overhead inference."""
        self.base.weight += self.scale * (self.B @ self.A)
        self.B.zero_()


# --------------------------------------------------------------------------------
# 5. Tiny decoder-only model with a KV cache, and the decode loop
# --------------------------------------------------------------------------------
class CachedAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dk = h, d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, cache=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)  # (B,h,T,dk)
        if cache is not None:                                    # append new keys/values
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        o, _ = sdpa(q, k, v, causal=True)
        return self.out(o.transpose(1, 2).reshape(B, T, D)), (k, v)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = CachedAttention(d, h)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, cache=None):
        a, cache = self.attn(self.n1(x), cache)
        x = x + a
        return x + self.mlp(self.n2(x)), cache


class TinyLM(nn.Module):
    def __init__(self, vocab=50, d=32, h=4, layers=2, max_len=256):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(layers))
        self.norm, self.head = RMSNorm(d), nn.Linear(d, vocab, bias=False)

    def forward(self, ids, cache=None):
        past = 0 if cache is None else cache[0][0].size(2)       # cached length = position offset
        x = self.tok(ids) + self.pos(torch.arange(past, past + ids.size(1), device=ids.device))
        new = []
        for i, blk in enumerate(self.blocks):
            x, c = blk(x, None if cache is None else cache[i])
            new.append(c)
        return self.head(self.norm(x)), new


@torch.no_grad()
def greedy_decode(model, prompt, max_new, use_cache=True):
    ids = prompt
    if use_cache:
        logits, cache = model(ids)                               # prefill: whole prompt once
        for _ in range(max_new):
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], 1)
            logits, cache = model(nxt, cache)                    # decode: ONE token per step
    else:
        for _ in range(max_new):                                 # recompute everything each step
            logits, _ = model(ids)
            ids = torch.cat([ids, logits[:, -1].argmax(-1, keepdim=True)], 1)
    return ids


# --------------------------------------------------------------------------------
# 6. Temperature, top-k and top-p (nucleus) sampling
# --------------------------------------------------------------------------------
def filter_logits(logits, top_k=0, top_p=1.0):
    logits = logits.clone()
    if top_k > 0:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits[logits < kth] = float("-inf")
    if top_p < 1.0:
        sl, si = torch.sort(logits, descending=True, dim=-1)
        p = sl.softmax(-1)
        before = p.cumsum(-1) - p                                # mass strictly before each token
        sl = sl.masked_fill(before >= top_p, float("-inf"))      # first token always survives
        logits = torch.full_like(logits, float("-inf")).scatter(-1, si, sl)
    return logits


def sample(logits, temperature=1.0, top_k=0, top_p=1.0):
    if temperature == 0:
        return logits.argmax(-1)
    probs = filter_logits(logits / temperature, top_k, top_p).softmax(-1)
    return torch.multinomial(probs, 1).squeeze(-1)


# --------------------------------------------------------------------------------
# 7. Retrieval metrics: hit@k, MRR, recall@k, nDCG@k, p95
# --------------------------------------------------------------------------------
def hit_at_k(ranked, relevant, k):
    return float(any(d in relevant for d in ranked[:k]))


def reciprocal_rank(ranked, relevant, k=None):
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def ndcg_at_k(ranked, relevant, k):
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def mean_over_queries(metric, rankings, relevants, **kw):
    return sum(metric(r, rel, **kw) for r, rel in zip(rankings, relevants)) / len(rankings)


def percentile(values, p):
    """Nearest-rank percentile: smallest value with at least p% of the data at or below it."""
    s = sorted(values)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


# --------------------------------------------------------------------------------
# 8. asyncio: bounded concurrency with per-call timeout and retry
# --------------------------------------------------------------------------------
async def run_bounded(items, fn, limit=5, timeout=2.0, retries=2, backoff=0.05):
    """Run fn(item) for every item with at most `limit` in flight.
    Results keep input order; a failure becomes an exception object in its slot
    instead of cancelling everything else."""
    sem = asyncio.Semaphore(limit)

    async def one(x):
        for attempt in range(retries + 1):
            try:
                async with sem:                                  # hold a slot only while working
                    return await asyncio.wait_for(fn(x), timeout)
            except asyncio.TimeoutError:
                if attempt == retries:
                    raise
            await asyncio.sleep(backoff * 2 ** attempt)          # backoff OUTSIDE the semaphore

    return await asyncio.gather(*(one(x) for x in items), return_exceptions=True)


# --------------------------------------------------------------------------------
# 9. DPO loss
# --------------------------------------------------------------------------------
def sequence_logprob(logits, labels, mask):
    """Sum of token log-probs. logits (B,T,V) predict labels (B,T); mask 1 on response tokens."""
    logp = logits.log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
    return (logp * mask).sum(-1)


def dpo_loss(pol_w, pol_l, ref_w, ref_l, beta=0.1):
    """Inputs are sequence log-probs for the chosen (w) and rejected (l) responses."""
    margin = beta * ((pol_w - ref_w) - (pol_l - ref_l))
    return -F.logsigmoid(margin).mean(), (margin > 0).float().mean()
