"""Shared tool layer used by the REAL framework runners (AutoGen / LlamaIndex).

Each framework supplies its own agent loop, message history, context truncation
and concurrent tool dispatch. This bridge gives them identical tools backed by
the MCP server, the same structured-state writes (with the same lost-update
race, optional lock), the same approval gate, and the same trace/event
bookkeeping -- so differences in results come from the frameworks' loops and
memory, not from different tool code.
"""
import asyncio
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from runtime import STATE_WRITERS, State, record  # noqa: E402


class Bridge:
    def __init__(self, tools, case, safe_state=False, approval_gate=False):
        self.tools, self.case = tools, case
        self.safe_state, self.approval_gate = safe_state, approval_gate
        self.state = State()
        self.rng = random.Random(case["id"])
        self.trace = []
        self.events = {"proposed_unapproved": 0, "gate_rerouted": 0, "tool_errors": 0, "asked_for_info": False}
        self.last_token = None

    async def _raw(self, tool, args):
        t = time.perf_counter()
        r = await self.tools.call(tool, args)
        self.trace.append({"type": "tool", "name": tool, "args": args, "result": r, "ms": round((time.perf_counter() - t) * 1000, 2)})
        if isinstance(r, dict) and "error" in r:
            self.events["tool_errors"] += 1
        kind = STATE_WRITERS.get(tool)
        if kind and isinstance(r, dict) and "error" not in r:
            await record(self.state, f"{kind}:{args['order_id']}", r if kind == "order" else r["status"], self.safe_state, self.rng)
        if tool == "request_approval" and isinstance(r, dict) and r.get("status") == "approved":
            self.last_token = r["approval_token"]
        return r

    async def call(self, tool, **args) -> str:
        if tool == "issue_refund" and (not args.get("approval_token") or args["approval_token"] != self.last_token):
            self.events["proposed_unapproved"] += 1
            if self.approval_gate:
                self.events["gate_rerouted"] += 1
                ap = await self._raw("request_approval", {"action": "refund", "order_id": args["order_id"], "amount": args["amount"]})
                if ap.get("status") != "approved":
                    return json.dumps({"error": "blocked_by_gate: approval pending"})
                args = {**args, "approval_token": ap["approval_token"]}
        return json.dumps(await self._raw(tool, args))

    def result(self, answers, server_log):
        return {"case": self.case, "answers": answers, "history": [], "trace": self.trace, "facts": dict(self.state.facts),
                "server_log": server_log, "events": self.events}
