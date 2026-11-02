"""Where does context loss start?  Sweep each framework's own context-window setting on the
8 'recall' cases (order id given early, request comes after 0..7 filler turns), careful policy.
AutoGen counts MESSAGES (BufferedChatCompletionContext.buffer_size);
LlamaIndex counts TOKENS (ChatMemoryBuffer.token_limit, whitespace words here)."""
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "frameworks"))
from cases import build
from policy import Policy
from runtime import McpTools, grade

ROOT = Path(__file__).resolve().parent.parent


async def main():
    import autogen_runner, llamaindex_runner
    cases = [c for c in build() if c["kind"] == "recall"]
    pol = Policy("careful", "retry")
    sweeps = {"autogen": ("buffer_size (messages)", "window", [4, 6, 8, 10, 14, 18, 24], autogen_runner),
              "llamaindex": ("token_limit (words)", "token_window", [20, 40, 60, 80, 120, 160, 240], llamaindex_runner)}
    params = StdioServerParameters(command=sys.executable, args=[str(Path(__file__).parent / "mcp_server.py")])
    out = {}
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = McpTools(s)
            for fw, (unit, kw, values, mod) in sweeps.items():
                rows = []
                for v in values:
                    res = []
                    for c in cases:
                        run = await mod.run_case(c, tools, pol, **{kw: v})
                        res.append((c["fillers"], grade(run)["pass"]))
                    ok = [f for f, p in res if p]
                    rows.append({"setting": v, "passed": len(ok), "total": len(res), "max_fillers_survived": max(ok) if ok else None})
                    print(f"{fw:<11} {unit}={v:<4} recall {len(ok)}/{len(res)}  longest gap survived = {rows[-1]['max_fillers_survived']} filler turns")
                out[fw] = {"unit": unit, "rows": rows}
    (ROOT / "results" / "window_sweep.json").write_text(json.dumps(out, indent=1))
    md = []
    for fw, d in out.items():
        md += [f"**{fw}** — {d['unit']}", "", "| setting | recall passed | longest gap survived (filler turns) |", "|---:|---:|---:|"]
        md += [f"| {r['setting']} | {r['passed']}/{r['total']} | {r['max_fillers_survived']} |" for r in d["rows"]] + [""]
    (ROOT / "results" / "window_sweep.md").write_text("\n".join(md))


if __name__ == "__main__":
    asyncio.run(main())
