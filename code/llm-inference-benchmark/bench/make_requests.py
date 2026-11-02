"""Build the FIXED request set used for every engine/config (seeded + hashed).

The same file is replayed against vLLM, TensorRT-LLM (or the simulator) so the
comparison is like-for-like. The sha256 of the file is stored with every run.
"""
import argparse
import hashlib
import json
import random

WORDS = ("data model latency batch token cache kernel memory stream queue prefill decode "
         "throughput request engine tensor shard quantize attention context window").split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="results/requests.jsonl")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = []
    for i in range(a.n):
        # mix of short chat-like and long-context prompts; output lengths vary
        plen = rng.choice([32, 64, 128, 256, 512, 1024], )
        prompt = " ".join(rng.choice(WORDS) for _ in range(plen))
        rows.append({"id": i, "prompt": prompt, "prompt_words": plen,
                     "max_tokens": rng.choice([32, 64, 96, 128]), "temperature": 0.0})
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(a.out, "sha256=", hashlib.sha256(open(a.out, "rb").read()).hexdigest()[:16])


if __name__ == "__main__":
    main()
