"""Compare two sweep JSONs: table + the concurrency 'knee' (first level where
p95 E2E is >= 2x its concurrency-1 value) + an optional PNG."""
import argparse
import json


def knee(sweep):
    base = sweep[0]["e2e_p95_ms"]
    for s in sweep:
        if s["e2e_p95_ms"] >= 2 * base:
            return s["concurrency"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=path.json")
    ap.add_argument("--md")
    ap.add_argument("--png")
    a = ap.parse_args()
    runs = {}
    for r in a.runs:
        label, path = r.split("=")
        runs[label] = json.load(open(path))["sweep"]
    lines = []
    for label, sw in runs.items():
        lines += [f"### {label}  (latency knee at concurrency = {knee(sw)})", "",
                  "| conc | TTFT p50 | TTFT p95 | TPOT p95 | E2E p95 | tok/s | req/s | err |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for s in sw:
            lines.append(f"| {s['concurrency']} | {s['ttft_p50_ms']:.0f} | {s['ttft_p95_ms']:.0f} | {s['tpot_p95_ms']:.1f} | "
                         f"{s['e2e_p95_ms']:.0f} | {s['out_tok_per_s']:.0f} | {s['req_per_s']:.1f} | {s['errors']} |")
        lines.append("")
    out = "\n".join(lines)
    print(out)
    if a.md:
        open(a.md, "w").write(out)
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        for label, sw in runs.items():
            c = [s["concurrency"] for s in sw]
            ax[0].plot(c, [s["e2e_p95_ms"] for s in sw], marker="o", label=label)
            ax[1].plot(c, [s["out_tok_per_s"] for s in sw], marker="o", label=label)
        ax[0].set(xscale="log", xlabel="concurrency", ylabel="E2E p95 (ms)", title="Latency vs load")
        ax[1].set(xscale="log", xlabel="concurrency", ylabel="output tokens/s", title="Throughput vs load")
        ax[0].legend()
        fig.tight_layout()
        fig.savefig(a.png, dpi=130)


if __name__ == "__main__":
    main()
