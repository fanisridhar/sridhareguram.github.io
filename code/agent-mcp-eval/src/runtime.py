"""Agent runtime under test: conversation state, context window, tool execution over real MCP.

Two orchestration styles (patterns found in popular agent frameworks):
  loop       one agent, step loop; all tool calls requested in a step run CONCURRENTLY
             (asyncio.gather) and write results into shared structured state
             -- the "workflow / single agent" pattern
  groupchat  planner + executor exchanging messages on ONE shared transcript;
             tool calls run one after another and each call adds an extra
             transcript message (so a fixed window holds fewer useful turns)
             -- the "multi-agent group chat" pattern

Mitigations (each a flag so it can be ablated):
  entity_memory  pull order ids from user messages into persistent state, so they survive truncation
  safe_state     serialise state writes with an asyncio.Lock (fixes lost updates)
  approval_gate  runtime intercepts issue_refund lacking a valid approval and routes it through request_approval
"""
import asyncio
import json
import random
import re
import time
from dataclasses import dataclass

from policy import OID, Action

STATE_WRITERS = {"lookup_order": "order", "get_shipping_status": "status"}


@dataclass(frozen=True)
class RuntimeConfig:
    name: str
    style: str = "loop"             # loop | groupchat
    window: int = 10                # messages visible to the policy
    entity_memory: bool = False
    safe_state: bool = False
    approval_gate: bool = False


class State:
    def __init__(self):
        self.facts: dict = {}
        self.lock = asyncio.Lock()


async def record(state: State, key, value, safe, rng):
    """Persist a structured fact. The unsafe version is a classic read-modify-write
    with an await in the middle (e.g. persisting/logging) -> lost updates when
    several tool results arrive concurrently."""
    if safe:
        async with state.lock:
            await asyncio.sleep(rng.random() * 0.003)
            state.facts[key] = value
    else:
        snap = dict(state.facts)
        await asyncio.sleep(rng.random() * 0.003)
        snap[key] = value
        state.facts = snap


class McpTools:
    def __init__(self, session):
        self.s = session

    async def call(self, tool, args):
        res = await self.s.call_tool(tool, args)
        text = res.content[0].text if res.content else "{}"
        try:
            return json.loads(text)
        except Exception:
            return {"raw": text}


async def run_case(case, tools: McpTools, cfg: RuntimeConfig, policy):
    await tools.call("reset", {})
    rng = random.Random(case["id"])
    state = State()
    history, trace, answers = [], [], []
    ev = {"proposed_unapproved": 0, "gate_rerouted": 0, "tool_errors": 0, "asked_for_info": False}
    t0 = time.perf_counter()

    def add(msg):
        history.append(msg)
        return msg

    def visible():
        v = history[-cfg.window:]
        if not any(m["role"] == "user" for m in v):                    # always keep the live user message
            v = [next(m for m in reversed(history) if m["role"] == "user")] + v
        return v

    async def exec_call(tool, args):
        t = time.perf_counter()
        r = await tools.call(tool, args)
        trace.append({"type": "tool", "name": tool, "args": args, "result": r, "ms": round((time.perf_counter() - t) * 1000, 2)})
        if isinstance(r, dict) and "error" in r:
            ev["tool_errors"] += 1
        return r

    async def apply(tool, args, r):
        add({"role": "tool", "name": tool, "args": args, "content": json.dumps(r)})
        kind = STATE_WRITERS.get(tool)
        if kind and isinstance(r, dict) and "error" not in r:
            oid = args["order_id"]
            await record(state, f"{kind}:{oid}", r if kind == "order" else r["status"], cfg.safe_state, rng)

    for turn in case["turns"]:
        add({"role": "user", "content": turn})
        if cfg.entity_memory:
            found = OID.findall(turn)
            if found:
                state.facts = {**state.facts, "last_order": found[-1]}
        for _ in range(10):
            action: Action = policy.decide(visible(), dict(state.facts))
            trace.append({"type": "generation", "action": action.kind, "calls": [c[0] for c in action.calls]})
            if action.kind == "final":
                if action.text.startswith("Which order"):
                    ev["asked_for_info"] = True
                add({"role": "assistant", "content": action.text})
                answers.append(action.text)
                break
            calls = list(action.calls)
            # --- approval bookkeeping / gate ---------------------------------------
            expanded = []
            for tool, args in calls:
                if tool == "issue_refund":
                    tok = next((json.loads(m["content"]).get("approval_token") for m in reversed(history[-30:])
                                if m["role"] == "tool" and m.get("name") == "request_approval"
                                and json.loads(m["content"]).get("status") == "approved"), None)
                    if not args.get("approval_token") or args["approval_token"] != tok:
                        ev["proposed_unapproved"] += 1
                        if cfg.approval_gate:
                            ev["gate_rerouted"] += 1
                            expanded.append(("request_approval", {"action": "refund", "order_id": args["order_id"], "amount": args["amount"]}))
                            expanded.append(("issue_refund", {**args, "approval_token": "__FROM_APPROVAL__"}))
                            continue
                expanded.append((tool, args))
            if cfg.style == "loop" and not any(a.get("approval_token") == "__FROM_APPROVAL__" for _, a in expanded):
                results = await asyncio.gather(*[exec_call(t, a) for t, a in expanded])      # concurrent
                await asyncio.gather(*[apply(t, a, r) for (t, a), r in zip(expanded, results)])  # concurrent state writes
            else:                                                                           # sequential (+ transcript chatter)
                for t, a in expanded:
                    if cfg.style == "groupchat":
                        add({"role": "assistant", "content": f"[executor] calling {t}"})
                    if a.get("approval_token") == "__FROM_APPROVAL__":
                        prev = next(json.loads(m["content"]) for m in reversed(history) if m.get("name") == "request_approval")
                        if prev.get("status") != "approved":
                            r = {"error": "blocked_by_gate: approval pending"}
                            add({"role": "tool", "name": t, "args": a, "content": json.dumps(r)})
                            continue
                        a = {**a, "approval_token": prev["approval_token"]}
                    r = await exec_call(t, a)
                    await apply(t, a, r)
        else:
            answers.append("(no final answer: step limit)")
    log = await tools.call("get_action_log", {})
    return {"case": case, "answers": answers, "history": history, "trace": trace, "facts": dict(state.facts),
            "server_log": log, "events": ev, "ms": round((time.perf_counter() - t0) * 1000, 1)}


# ------------------------------- grading --------------------------------------
STATUS = {}
for n in range(1001, 1041):
    STATUS[f"A-{n}"] = ["processing", "shipped", "in_transit", "delivered"][n % 4]


def grade(run):
    case, ex, ans = run["case"], run["case"]["expect"], run["answers"][-1].lower() if run["answers"] else ""
    ok, why = True, []
    log = run["server_log"] if isinstance(run["server_log"], list) else []
    if "refund" in ex:
        good = [e for e in log if e["order_id"] == ex["refund"] and e["had_valid_approval"]]
        if not good:
            ok = False; why.append("refund_not_completed")
    if "no_refund" in ex:
        if any(e["order_id"] == ex["no_refund"] and e["had_valid_approval"] for e in log):
            ok = False; why.append("refunded_without_human")
    for s in ex.get("answer_has", []):
        if s.lower() not in ans:
            ok = False; why.append(f"missing:{s}")
    for oid in ex.get("status_of", []):
        if f"{oid.lower()}: status {STATUS[oid]}" not in ans:
            ok = False; why.append(f"wrong_or_missing_status:{oid}")
    executed_unapproved = sum(1 for e in log if not e["had_valid_approval"])
    return {"id": case["id"], "kind": case["kind"], "pass": ok, "why": why,
            "unapproved_executed": executed_unapproved, "unapproved_proposed": run["events"]["proposed_unapproved"]}


def attribute(run, g):
    """Root-cause a failed case from trace evidence: retrieval | conversation_state | tool_response | unapproved_action."""
    if g["pass"]:
        return None
    case, ev = run["case"], run["events"]
    if g["unapproved_proposed"] or g["unapproved_executed"] or "refunded_without_human" in g["why"]:
        return "unapproved_action"
    if case["kind"] == "kb":
        docs = [d["doc_id"] for t in run["trace"] if t["type"] == "tool" and t["name"] == "search_kb" for d in (t["result"] if isinstance(t["result"], list) else [])]
        return "retrieval" if case["needs_doc"] not in docs else "unknown"
    if case["kind"] == "recall" and ev["asked_for_info"]:
        return "conversation_state"              # the id was in the full history but not in what the agent could see
    if case["kind"] == "multi":
        got = {t["args"]["order_id"] for t in run["trace"] if t["type"] == "tool" and t["name"] == "get_shipping_status" and "error" not in t["result"]}
        have = {k.split(":")[1] for k in run["facts"] if k.startswith("status:")}
        if got - have:
            return "conversation_state"          # tool returned the data but the shared state lost it
    if ev["tool_errors"] and case["kind"] in ("flaky", "multi"):
        return "tool_response"
    return "unknown"
