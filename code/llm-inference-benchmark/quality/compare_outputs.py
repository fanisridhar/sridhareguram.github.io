"""Did quantisation change the answers?  Compare greedy (temperature 0) outputs
of a reference run vs a quantised run on the same fixed request set.

Metrics: exact-match rate, mean token-level similarity (difflib ratio), mean
position of first divergence, and mean length change. If logprobs are
available from the engine you can additionally compare perplexity on a held-out
text (see README) -- this script covers the output-level check.
"""
import argparse
import difflib
import json


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("quant")
    ap.add_argument("--out")
    a = ap.parse_args()
    ref, q = json.load(open(a.ref)), json.load(open(a.quant))
    ids = sorted(set(ref) & set(q), key=int)
    exact = sim = div = dlen = 0
    for i in ids:
        r, c = ref[i].split(), q[i].split()
        exact += r == c
        sim += difflib.SequenceMatcher(None, r, c).ratio()
        div += first_div(r, c) / max(len(r), 1)
        dlen += len(c) - len(r)
    n = len(ids)
    out = {"n": n, "exact_match_rate": exact / n, "mean_token_similarity": sim / n,
           "mean_first_divergence_frac": div / n, "mean_len_change": dlen / n}
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
