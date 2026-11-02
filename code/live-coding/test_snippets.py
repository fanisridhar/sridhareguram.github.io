"""Run with:  python test_snippets.py   (or pytest). Every snippet is checked against PyTorch
or a hand-computed answer, and the derivation formulas are verified numerically."""
import asyncio
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

import snippets as S

torch.manual_seed(0)


# ---------------- softmax / cross-entropy ----------------
def test_softmax_matches_torch_and_is_stable():
    z = torch.randn(4, 7)
    assert torch.allclose(S.softmax(z), z.softmax(-1), atol=1e-6)
    big = torch.tensor([[1000.0, 1001.0, 999.0]])                 # naive exp() would overflow
    assert torch.isfinite(S.softmax(big)).all()
    assert torch.allclose(S.softmax(big), big.softmax(-1))


def test_softmax_jacobian_formula():
    z = torch.randn(5, dtype=torch.double)
    J = torch.autograd.functional.jacobian(lambda t: t.softmax(-1), z)
    p = z.softmax(-1)
    assert torch.allclose(J, torch.diag(p) - torch.outer(p, p))   # dp_i/dz_j = p_i (delta_ij - p_j)


def test_cross_entropy_matches_torch():
    logits, y = torch.randn(16, 10), torch.randint(0, 10, (16,))
    assert torch.allclose(S.cross_entropy(logits, y), F.cross_entropy(logits, y), atol=1e-6)
    y2 = y.clone(); y2[:5] = -100
    assert torch.allclose(S.cross_entropy(logits, y2), F.cross_entropy(logits, y2), atol=1e-6)
    huge = torch.tensor([[1e4, 0.0, -1e4]])
    assert torch.isfinite(S.cross_entropy(huge, torch.tensor([0])))


def test_cross_entropy_gradient_is_p_minus_onehot():
    z = torch.randn(3, 6, dtype=torch.double, requires_grad=True)
    y = torch.tensor([2, 0, 5])
    F.cross_entropy(z, y, reduction="sum").backward()
    expected = z.softmax(-1).detach() - F.one_hot(y, 6).double()
    assert torch.allclose(z.grad, expected)


# ---------------- RMSNorm ----------------
def test_rmsnorm_matches_torch_and_formula():
    x = torch.randn(3, 5, 16)
    m = S.RMSNorm(16, eps=1e-6)
    ref = torch.nn.RMSNorm(16, eps=1e-6)
    assert torch.allclose(m(x), ref(x), atol=1e-6)
    out = S.RMSNorm(16)(x)                                       # with weight = 1, rms of output is 1
    assert torch.allclose(out.pow(2).mean(-1).sqrt(), torch.ones(3, 5), atol=1e-3)
    assert torch.allclose(S.RMSNorm(16)(3.0 * x), out, atol=1e-4)  # scale invariance


def test_rmsnorm_jacobian_formula():
    n, eps = 6, 1e-6
    x = torch.randn(n, dtype=torch.double)
    g = torch.randn(n, dtype=torch.double)
    f = lambda t: t * torch.rsqrt(t.pow(2).mean() + eps) * g
    J = torch.autograd.functional.jacobian(f, x)
    r = torch.sqrt(x.pow(2).mean() + eps)
    analytic = g[:, None] * (torch.eye(n, dtype=torch.double) / r - torch.outer(x, x) / (n * r ** 3))
    assert torch.allclose(J, analytic, atol=1e-8)                # dy_i/dx_j = g_i (delta_ij / r - x_i x_j / (n r^3))


# ---------------- attention ----------------
def test_sdpa_matches_torch():
    q, k, v = (torch.randn(2, 4, 6, 8) for _ in range(3))
    for causal in (False, True):
        out, w = S.sdpa(q, k, v, causal=causal)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        assert torch.allclose(out, ref, atol=1e-5)
        assert torch.allclose(w.sum(-1), torch.ones(2, 4, 6), atol=1e-5)
    # causal: weight on future positions is exactly zero
    _, w = S.sdpa(q, k, v, causal=True)
    assert torch.all(w.triu(1) == 0)


def test_sdpa_causal_is_bottom_right_aligned_for_cached_decode():
    q, k, v = torch.randn(1, 2, 1, 8), torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8)
    out, w = S.sdpa(q, k, v, causal=True)                        # one new query sees all 5 keys
    ref = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out, ref, atol=1e-5) and (w > 0).all()
    q2 = torch.randn(1, 2, 2, 8)                                  # 2 new queries over 5 keys
    out2, _ = S.sdpa(q2, k, v, causal=True)
    allowed = torch.ones(2, 5, dtype=torch.bool).tril(diagonal=3)
    assert torch.allclose(out2, F.scaled_dot_product_attention(q2, k, v, attn_mask=allowed), atol=1e-5)


def test_sdpa_padding_mask():
    q, k, v = torch.randn(1, 1, 3, 4), torch.randn(1, 1, 5, 4), torch.randn(1, 1, 5, 4)
    keep = torch.tensor([True, True, True, False, False])
    _, w = S.sdpa(q, k, v, mask=keep[None, None, None, :])
    assert torch.all(w[..., 3:] == 0) and torch.allclose(w.sum(-1), torch.ones(1, 1, 3), atol=1e-6)


def test_why_divide_by_sqrt_d():
    d = 256
    q, k = torch.randn(20000, d), torch.randn(20000, d)
    raw = (q * k).sum(-1)
    assert abs(raw.var().item() - d) / d < 0.1                    # variance of q.k grows like d
    assert abs((raw / math.sqrt(d)).var().item() - 1) < 0.1       # scaling restores unit variance


# ---------------- LoRA ----------------
def test_lora_layer():
    base = torch.nn.Linear(12, 20)
    x = torch.randn(4, 12)
    lora = S.LoRALinear(base, r=4, alpha=8)
    assert torch.allclose(lora(x), base(x), atol=1e-6)            # B = 0, so identical at step 0
    trainable = [n for n, p in lora.named_parameters() if p.requires_grad]
    assert sorted(trainable) == ["A", "B"]
    assert sum(p.numel() for p in lora.parameters() if p.requires_grad) == 4 * (12 + 20)
    with torch.no_grad():
        lora.B.normal_()
    out = lora(x)
    W = base.weight + lora.scale * (lora.B @ lora.A)
    assert torch.allclose(out, F.linear(x, W, base.bias), atol=1e-5)
    out.sum().backward()
    assert lora.A.grad is not None and lora.B.grad is not None and base.weight.grad is None
    lora.merge()
    assert torch.allclose(lora(x), out, atol=1e-5)               # merged weights give the same output


# ---------------- KV cache ----------------
def test_kv_cache_decode_equals_full_recompute():
    model = S.TinyLM().eval()
    prompt = torch.randint(0, 50, (1, 7))
    a = S.greedy_decode(model, prompt, 12, use_cache=True)
    b = S.greedy_decode(model, prompt, 12, use_cache=False)
    assert torch.equal(a, b)
    full, _ = model(a[:, :-1])
    logits, cache = model(a[:, :6])
    for t in range(6, a.size(1) - 1):
        step, cache = model(a[:, t:t + 1], cache)
        assert torch.allclose(step[:, -1], full[:, t], atol=1e-4)  # same logits at every step


def test_kv_cache_does_linear_work_and_model_is_causal():
    model = S.TinyLM().eval()
    seen = []
    orig = model.forward
    model.forward = lambda ids, cache=None: (seen.append(ids.numel()), orig(ids, cache))[1]
    prompt, n = torch.randint(0, 50, (1, 7)), 12
    S.greedy_decode(model, prompt, n, use_cache=True)
    with_cache = sum(seen)
    seen.clear()
    S.greedy_decode(model, prompt, n, use_cache=False)
    without = sum(seen)
    assert with_cache == 7 + n and without == sum(7 + i for i in range(n))   # O(n) vs O(n^2) tokens processed
    model.forward = orig
    x = torch.randint(0, 50, (1, 10))
    y = x.clone(); y[0, 6:] = (y[0, 6:] + 1) % 50                # change only the future
    assert torch.allclose(model(x)[0][:, :6], model(y)[0][:, :6], atol=1e-5)


# ---------------- sampling ----------------
P = torch.tensor([0.5, 0.3, 0.15, 0.05])
LOGITS = P.log()


def test_top_k_and_top_p_filters():
    assert torch.isfinite(S.filter_logits(LOGITS, top_k=2)).sum() == 2
    assert torch.isfinite(S.filter_logits(LOGITS, top_p=0.85)).tolist() == [True, True, True, False]
    assert torch.isfinite(S.filter_logits(LOGITS, top_p=0.4)).tolist() == [True, False, False, False]  # never empty
    both = S.filter_logits(LOGITS, top_k=3, top_p=0.4)
    assert torch.isfinite(both).sum() == 1
    batch = S.filter_logits(torch.stack([LOGITS, LOGITS.flip(0)]), top_k=2)
    assert torch.isfinite(batch).sum(-1).tolist() == [2, 2]


def test_sampling_frequencies_and_temperature():
    torch.manual_seed(0)
    draws = torch.stack([S.sample(LOGITS, top_k=2) for _ in range(6000)])
    assert set(draws.tolist()) == {0, 1}                          # tokens outside the top-2 never appear
    assert abs((draws == 0).float().mean().item() - 0.5 / 0.8) < 0.03
    assert S.sample(LOGITS, temperature=0).item() == 0            # temperature 0 = greedy
    cold = torch.stack([S.sample(LOGITS, temperature=0.05) for _ in range(300)])
    assert (cold == 0).float().mean() > 0.99                      # low temperature sharpens
    assert S.sample(torch.randn(3, 10), top_p=0.9).shape == (3,)


# ---------------- retrieval metrics ----------------
def test_hit_mrr_recall_ndcg_percentile():
    ranked, rel = ["a", "b", "c", "d", "e"], {"c", "e"}
    assert S.hit_at_k(ranked, rel, 2) == 0 and S.hit_at_k(ranked, rel, 3) == 1
    assert S.reciprocal_rank(ranked, rel) == 1 / 3 and S.reciprocal_rank(ranked, rel, k=2) == 0
    assert S.recall_at_k(ranked, rel, 3) == 0.5 and S.recall_at_k(ranked, rel, 5) == 1.0
    expected = (1 / math.log2(4) + 1 / math.log2(6)) / (1 + 1 / math.log2(3))
    assert abs(S.ndcg_at_k(ranked, rel, 5) - expected) < 1e-12
    assert S.ndcg_at_k(["c", "e", "a"], rel, 3) == 1.0            # perfect ranking
    rankings, rels = [["x", "y"], ["a", "b"]], [{"x"}, {"b"}]
    assert S.mean_over_queries(S.reciprocal_rank, rankings, rels) == 0.75   # (1 + 1/2) / 2
    assert S.mean_over_queries(S.hit_at_k, rankings, rels, k=1) == 0.5
    data = list(range(1, 101))
    assert S.percentile(data, 95) == 95 and abs(np.percentile(data, 95) - 95.05) < 1e-9   # methods differ slightly


# ---------------- asyncio ----------------
def test_run_bounded_limit_order_timeout_and_retry():
    state = {"now": 0, "peak": 0, "calls": {}}

    async def fn(x):
        state["now"] += 1
        state["peak"] = max(state["peak"], state["now"])
        state["calls"][x] = state["calls"].get(x, 0) + 1
        try:
            if x == "slow":
                await asyncio.sleep(1.0)                          # always exceeds the timeout
            elif x == "flaky" and state["calls"][x] == 1:
                await asyncio.sleep(1.0)                          # first attempt times out, retry succeeds
            else:
                await asyncio.sleep(0.02)
            return x.upper()
        finally:
            state["now"] -= 1

    items = [f"i{n}" for n in range(12)] + ["slow", "flaky"]
    t0 = time.perf_counter()
    res = asyncio.run(S.run_bounded(items, fn, limit=3, timeout=0.1, retries=1, backoff=0.01))
    assert state["peak"] <= 3                                      # never more than `limit` in flight
    assert res[:12] == [s.upper() for s in items[:12]]             # order preserved
    assert isinstance(res[12], asyncio.TimeoutError) and state["calls"]["slow"] == 2   # retried once, then gave up
    assert res[13] == "FLAKY" and state["calls"]["flaky"] == 2     # retry rescued it
    assert time.perf_counter() - t0 < 2.0                          # one slow item did not block the rest


# ---------------- DPO ----------------
def test_dpo_loss_and_gradients():
    zeros = torch.zeros(4)
    loss, acc = S.dpo_loss(zeros, zeros, zeros, zeros, beta=0.1)
    assert abs(loss.item() - math.log(2)) < 1e-6                   # policy == reference -> log 2
    pw = torch.tensor([-10.0, -12.0], requires_grad=True)
    pl = torch.tensor([-11.0, -9.0], requires_grad=True)
    rw, rl = torch.tensor([-10.5, -11.0]), torch.tensor([-10.0, -10.0])
    loss, acc = S.dpo_loss(pw, pl, rw, rl, beta=0.5)
    m = 0.5 * ((pw - rw) - (pl - rl))
    assert torch.allclose(loss, -F.logsigmoid(m).mean())
    loss.backward()
    assert torch.allclose(pw.grad, -0.5 * torch.sigmoid(-m).detach() / 2)   # dL/d pi_w = -beta * sigma(-m) / N
    assert torch.allclose(pl.grad, 0.5 * torch.sigmoid(-m).detach() / 2)    # pushes rejected log-prob down
    logits = torch.randn(2, 5, 9)
    labels, mask = torch.randint(0, 9, (2, 5)), torch.tensor([[0, 0, 1, 1, 1]] * 2)
    lp = S.sequence_logprob(logits, labels, mask)
    manual = sum(logits[0, t].log_softmax(-1)[labels[0, t]] for t in (2, 3, 4))
    assert torch.allclose(lp[0], manual, atol=1e-5)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print("OK ", name)
    print(f"\n{len(tests)} tests passed")
