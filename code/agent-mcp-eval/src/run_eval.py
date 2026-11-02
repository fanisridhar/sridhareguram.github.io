"""Run every runtime configuration over the same 40 cases and report failures by cause."""
import asyncio
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, str(Path(__file__).parent))
from cases import build
from policy import Policy
from runtime import McpTools, RuntimeConfig, attribute, grade, run_case

ROOT = Path(__file__).resolve().parent.parent
RISKY = Policy("eager", "naive")        # skips approval, fabricates around tool errors
CAREFUL = Policy("careful", "retry")

MATRIX = [
    ("loop", "baseline", RuntimeConfig("loop/baseline", "loop"), RISKY),
    ("groupchat", "baseline", RuntimeConfig("groupchat/baseline", "groupchat"), RISKY),
    ("loop", "+entity_memory", RuntimeConfig("loop/+entity_memory", "loop", entity_memory=True), RISKY),
    ("loop", "+safe_state", RuntimeConfig("loop/+safe_state", "loop", safe_state=True), RISKY),
    ("loop", "+approval_gate", RuntimeConfig("loop/+approval_gate", "loop", approval_gate=True), RISKY),
    ("loop", "all mitigations", RuntimeConfig("loop/all", "loop", entity_memory=True, safe_state=True, approval_gate=True), RISKY),
    ("groupchat", "all mitigations", RuntimeConfig("groupchat/all", "groupchat", entity_memory=True, safe_state=True, approval_gate=True), RISKY),
    ("loop", "baseline, careful policy", RuntimeConfig("loop/baseline+careful", "loop"), CAREFUL),
    ("groupchat", "baseline, careful policy", RuntimeConfig("groupchat/baseline+careful", "groupchat"), CAREFUL),
]


async def main():
    cases = build()
    params = StdioServerParameters(command=sys.executable, args=[str(Path(__file__).parent / "mcp_server.py")])
    out_rows, traces = [], open(ROOT / "results" / "traces.jsonl", "w")
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = McpTools(session)
            listed = [t.name for t in (await session.list_tools()).tools]
            print("MCP tools:", listed)
            for style, label, cfg, policy in MATRIX:
                graded, causes, by_kind = [], Counter(), defaultdict(lambda: [0, 0])
                for case in cases:
                    run = await run_case(case, tools, cfg, policy)
                    g = grade(run)
                    cause = attribute(run, g)
                    graded.append(g)
                    by_kind[g["kind"]][0] += g["pass"]
                    by_kind[g["kind"]][1] += 1
                    if cause:
                        causes[cause] += 1
                    traces.write(json.dumps({"config": cfg.name, "case": case["id"], "pass": g["pass"], "why": g["why"], "cause": cause,
                                             "answers": run["answers"], "trace": run["trace"], "events": run["events"]}) + "\n")
                n_pass = sum(g["pass"] for g in graded)
                row = {"runtime": cfg.name, "policy": f"{policy.approval}/{policy.resilience}", "passed": n_pass, "total": len(graded),
                       "pass_rate_pct": round(100 * n_pass / len(graded), 1),
                       "by_kind": {k: f"{v[0]}/{v[1]}" for k, v in by_kind.items()},
                       "failures_by_cause": dict(causes),
                       "unapproved_proposed": sum(g["unapproved_proposed"] for g in graded),
                       "unapproved_executed": sum(g["unapproved_executed"] for g in graded)}
                out_rows.append(row)
                print(f"{cfg.name:<28} {n_pass:>2}/{len(graded)}  causes={dict(causes)}  proposed_unapproved={row['unapproved_proposed']} executed={row['unapproved_executed']}")
    traces.close()
    (ROOT / "results" / "summary.json").write_text(json.dumps(out_rows, indent=1))
    kinds = ["recall", "approval", "kb", "flaky", "multi"]
    md = ["| runtime | policy | pass | " + " | ".join(kinds) + " | failures by cause | unapproved proposed / executed |",
          "|---|---|---:|" + "---:|" * len(kinds) + "---|---:|"]
    for r in out_rows:
        md.append(f"| {r['runtime']} | {r['policy']} | {r['passed']}/{r['total']} | " + " | ".join(r["by_kind"].get(k, "-") for k in kinds) +
                  f" | {r['failures_by_cause'] or '-'} | {r['unapproved_proposed']} / {r['unapproved_executed']} |")
    (ROOT / "results" / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    asyncio.run(main())
