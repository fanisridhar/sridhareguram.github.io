"""Deterministic stand-in for the LLM ("policy").

It sees ONLY what the runtime shows it: the visible (possibly truncated) message
window and the runtime's structured state facts. That is deliberate: this
project evaluates the AGENT RUNTIME (context management, approval gating,
concurrent tool execution, state handling), and the failures it finds are
properties of the runtime, not of a particular LLM. The interface
(`decide(visible, facts) -> Action`) is the seam where a real LLM goes.

Knobs model two common LLM behaviours:
  approval   'careful' asks request_approval before issue_refund; 'eager' skips straight to issue_refund
  resilience 'retry'  retries a tool once after an error;        'naive' reports/fabricates around the error
"""
import re
from dataclasses import dataclass, field

OID = re.compile(r"A-\d{4}")


@dataclass
class Action:
    kind: str                       # 'tools' | 'final'
    calls: list = field(default_factory=list)   # [(tool, args)]
    text: str = ""


class Policy:
    def __init__(self, approval="careful", resilience="retry"):
        self.approval, self.resilience = approval, resilience

    @staticmethod
    def _find(visible, pattern, roles=("user", "tool")):
        for m in reversed(visible):
            if m["role"] in roles:
                f = pattern.findall(m["content"])
                if f:
                    return f[-1]
        return None

    @staticmethod
    def _tool_results(visible, tool):
        import json
        out = []
        for m in visible:
            if m["role"] == "tool" and m.get("name") == tool:
                try:
                    out.append((m.get("args", {}), json.loads(m["content"])))
                except Exception:
                    pass
        return out

    def decide(self, visible, facts) -> Action:
        last_user = next((m["content"] for m in reversed(visible) if m["role"] == "user"), None)
        if last_user is None:   # context window cut off the user's request entirely
            return Action("final", text="Which order do you mean? I can't see your request any more.")
        low = last_user.lower()
        # a tool result is waiting to be consumed -> continue the flow of the current user turn
        turn_msgs = visible[max(i for i, m in enumerate(visible) if m["role"] == "user"):]
        if "refund" in low:
            return self._refund(visible, facts, turn_msgs)
        if "ship" in low or "where is" in low or "status" in low:
            return self._status(visible, facts, turn_msgs, last_user)
        if any(w in low for w in ("return", "send back", "money back", "bring back", "window", "policy")):
            return self._kb(turn_msgs, last_user)
        return Action("final", text="You're welcome! Anything else I can help with?")

    # --- flows ---------------------------------------------------------
    def _refund(self, visible, facts, turn):
        oid = self._find(visible, OID, ("user",)) or facts.get("last_order") or self._find(visible, OID)
        if not oid:
            return Action("final", text="Which order do you mean? I don't see an order number in our conversation.")
        order = facts.get(f"order:{oid}") or next((r for a, r in self._tool_results(turn, "lookup_order")), None)
        if not order or "total" not in order:
            return Action("tools", [("lookup_order", {"order_id": oid})])
        amount = order["total"]
        appr = [r for a, r in self._tool_results(turn, "request_approval")]
        ref = [r for a, r in self._tool_results(turn, "issue_refund")]
        ship = [r for a, r in self._tool_results(turn, "get_shipping_status") if "ship_date" in r]
        if ref:
            if "refunded" in ref[-1].get("status", ""):
                date = order.get("ship_date", "unknown")
                return Action("final", text=f"Refund of ${amount:.2f} issued for {oid}. It shipped on {date}.")
            return Action("final", text=f"I could not refund {oid}: {ref[-1].get('error')}.")
        if self.approval == "eager":
            return Action("tools", [("issue_refund", {"order_id": oid, "amount": amount, "approval_token": ""})])
        if not appr:
            return Action("tools", [("request_approval", {"action": "refund", "order_id": oid, "amount": amount})])
        if appr[-1].get("status") == "approved":
            return Action("tools", [("issue_refund", {"order_id": oid, "amount": amount, "approval_token": appr[-1]["approval_token"]})])
        return Action("final", text=f"Refund of ${amount:.2f} for {oid} needs human approval; I have escalated it.")

    def _status(self, visible, facts, turn, last_user):
        oids = OID.findall(last_user) or ([facts["last_order"]] if facts.get("last_order") else [])
        if not oids:
            return Action("final", text="Which order would you like the status of?")
        done = {a.get("order_id"): r for a, r in self._tool_results(turn, "get_shipping_status")}
        todo = [o for o in oids if o not in done]
        if todo:
            return Action("tools", [("get_shipping_status", {"order_id": o}) for o in todo])
        errs = [o for o in oids if "error" in done[o]]
        if errs:
            n_calls = sum(1 for m in turn if m["role"] == "tool" and m.get("name") == "get_shipping_status")
            if self.resilience == "retry" and n_calls <= len(oids):
                return Action("tools", [("get_shipping_status", {"order_id": o}) for o in errs])
            if self.resilience == "naive":      # fabricates instead of surfacing the error
                return Action("final", text=f"Order {errs[0]} status: shipped, arriving soon.")
            return Action("final", text=f"I could not get the status of {errs[0]} (upstream_timeout).")
        # compose from STRUCTURED STATE written by the runtime (not from raw messages)
        parts = []
        for o in oids:
            st = facts.get(f"status:{o}")
            parts.append(f"{o}: status {st}" if st else f"{o}: status unknown (no data)")
        return Action("final", text="; ".join(parts))

    def _kb(self, turn, last_user):
        res = self._tool_results(turn, "search_kb")
        if not res:
            q = " ".join(w for w in last_user.lower().replace("?", "").split())
            return Action("tools", [("search_kb", {"query": q})])
        docs = res[-1][1] if isinstance(res[-1][1], list) else []
        if not docs:
            return Action("final", text="I couldn't find that in our policies.")
        return Action("final", text=docs[0]["text"])
