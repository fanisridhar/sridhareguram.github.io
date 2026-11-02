"""Compare REAL AutoGen vs REAL LlamaIndex agents on the same 40 cases (same MCP server, tools, state, policy)."""
import asyncio
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "frameworks"))
from cases import build
from policy import Policy
from runtime import McpTools, attribute, grade

ROOT = Path(__file__).resolve().parent.parent
RISKY, CAREFUL = Policy("eager", "naive"), Policy("careful", "retry")

CONFIGS = [  # (framework, label, policy, kwargs)
    ("autogen", "baseline", RISKY, {}),
    ("autogen", "+safe_state", RISKY, {"safe_state": True}),
    ("autogen", "all mitigations", RISKY, {"entity_memory": True, "safe_state": True, "approval_gate": True}),
    ("autogen", "baseline, careful policy", CAREFUL, {}),
    ("llamaindex", "baseline", RISKY, {}),
    ("llamaindex", "+safe_state", RISKY, {"safe_state": True}),
    ("llamaindex", "all mitigations", RISKY, {"entity_memory": True, "safe_state": True, "approval_gate": True}),
    ("llamaindex", "baseline, careful policy", CAREFUL, {}),
]


async def main():
    import autogen_runner
    try:
        import llamaindex_runner
    except Exception as e:  # pragma: no cover
        print("llamaindex runner unavailable:", repr(e))
        llamaindex_runner = None
    runners = {"autogen": autogen_runner, "llamaindex": llamaindex_runner}
    cases = build()
    params = StdioServerParameters(command=sys.executable, args=[str(Path(__file__).parent / "mcp_server.py")])
    rows = []
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = McpTools(session)
            for fw, label, policy, kw in CONFIGS:
                mod = runners[fw]
                if mod is None:
                    continue
                causes, by_kind, graded = Counter(), defaultdict(lambda: [0, 0]), []
                for case in cases:
                    run = await mod.run_case(case, tools, policy, **kw)
                    g = grade(run)
                    c = attribute(run, g)
                    graded.append(g)
                    by_kind[g["kind"]][0] += g["pass"]
                    by_kind[g["kind"]][1] += 1
                    if c:
                        causes[c] += 1
                n = sum(g["pass"] for g in graded)
                row = {"framework": fw, "config": label, "policy": f"{policy.approval}/{policy.resilience}", "passed": n, "total": len(graded),
                       "by_kind": {k: f"{v[0]}/{v[1]}" for k, v in by_kind.items()}, "failures_by_cause": dict(causes),
                       "unapproved_proposed": sum(g["unapproved_proposed"] for g in graded),
                       "unapproved_executed": sum(g["unapproved_executed"] for g in graded)}
                rows.append(row)
                print(f"{fw:<11}{label:<26} {n:>2}/{len(graded)} causes={dict(causes)} proposed={row['unapproved_proposed']} executed={row['unapproved_executed']}")
    (ROOT / "results" / "frameworks_summary.json").write_text(json.dumps(rows, indent=1))
    kinds = ["recall", "approval", "kb", "flaky", "multi"]
    md = ["| framework | config | policy | pass | " + " | ".join(kinds) + " | failures by cause | unapproved proposed / executed |",
          "|---|---|---|---:|" + "---:|" * len(kinds) + "---|---:|"]
    for r in rows:
        md.append(f"| {r['framework']} | {r['config']} | {r['policy']} | {r['passed']}/{r['total']} | " + " | ".join(r["by_kind"].get(k, "-") for k in kinds) +
                  f" | {r['failures_by_cause'] or '-'} | {r['unapproved_proposed']} / {r['unapproved_executed']} |")
    (ROOT / "results" / "frameworks_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    asyncio.run(main())
