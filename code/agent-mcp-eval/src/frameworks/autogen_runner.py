"""Run the evaluation cases through REAL AutoGen (autogen-agentchat AssistantAgent).

Real parts: AssistantAgent's tool-call loop (max_tool_iterations), its concurrent
execution of multiple tool calls in one step, its FunctionExecutionResult message
format, and BufferedChatCompletionContext (its message-count context window).
Stand-in part: the model client -- a deterministic policy instead of an LLM.
"""
import json
import sys
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_core import FunctionCall
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import (AssistantMessage, ChatCompletionClient, CreateResult, FunctionExecutionResultMessage,
                                 RequestUsage, UserMessage)
from autogen_core.tools import FunctionTool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bridge import Bridge  # noqa: E402
from policy import OID  # noqa: E402


def to_visible(messages):
    vis, calls = [], {}
    for m in messages:
        if isinstance(m, UserMessage):
            vis.append({"role": "user", "content": m.content if isinstance(m.content, str) else str(m.content)})
        elif isinstance(m, AssistantMessage):
            if isinstance(m.content, str):
                vis.append({"role": "assistant", "content": m.content})
            else:
                for fc in m.content:
                    calls[fc.id] = (fc.name, json.loads(fc.arguments or "{}"))
                vis.append({"role": "assistant", "content": "[tool calls]"})
        elif isinstance(m, FunctionExecutionResultMessage):
            for r in m.content:
                name, args = calls.get(r.call_id, (r.name, {}))
                vis.append({"role": "tool", "name": name, "args": args, "content": r.content})
    return vis


class PolicyClient(ChatCompletionClient):
    def __init__(self, policy, bridge):
        self.policy, self.bridge, self.n = policy, bridge, 0

    async def create(self, messages, *, tools=(), tool_choice="auto", json_output=None, extra_create_args={}, cancellation_token=None):
        vis = to_visible(messages)
        act = self.policy.decide(vis, dict(self.bridge.state.facts))
        self.n += 1
        usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        if act.kind == "final":
            return CreateResult(finish_reason="stop", content=act.text, usage=usage, cached=False)
        fcs = [FunctionCall(id=f"call-{self.n}-{i}", name=t, arguments=json.dumps(a)) for i, (t, a) in enumerate(act.calls)]
        return CreateResult(finish_reason="function_calls", content=fcs, usage=usage, cached=False)

    def create_stream(self, *a, **k):
        raise NotImplementedError

    async def close(self): ...
    def actual_usage(self): return RequestUsage(prompt_tokens=0, completion_tokens=0)
    def total_usage(self): return RequestUsage(prompt_tokens=0, completion_tokens=0)
    def count_tokens(self, messages, *, tools=()): return 0
    def remaining_tokens(self, messages, *, tools=()): return 10**6
    @property
    def capabilities(self): return {"vision": False, "function_calling": True, "json_output": False}
    @property
    def model_info(self):
        return {"vision": False, "function_calling": True, "json_output": False, "family": "unknown", "structured_output": False}


async def run_case(case, tools, policy, window=10, entity_memory=False, safe_state=False, approval_gate=False):
    await tools.call("reset", {})
    b = Bridge(tools, case, safe_state, approval_gate)

    async def lookup_order(order_id: str) -> str: return await b.call("lookup_order", order_id=order_id)
    async def get_shipping_status(order_id: str) -> str: return await b.call("get_shipping_status", order_id=order_id)
    async def search_kb(query: str) -> str: return await b.call("search_kb", query=query)
    async def request_approval(action: str, order_id: str, amount: float) -> str:
        return await b.call("request_approval", action=action, order_id=order_id, amount=amount)
    async def issue_refund(order_id: str, amount: float, approval_token: str = "") -> str:
        return await b.call("issue_refund", order_id=order_id, amount=amount, approval_token=approval_token)

    fts = [FunctionTool(f, description=f.__name__, name=f.__name__) for f in
           (lookup_order, get_shipping_status, search_kb, request_approval, issue_refund)]
    agent = AssistantAgent("support", model_client=PolicyClient(policy, b), tools=fts,
                           model_context=BufferedChatCompletionContext(buffer_size=window),
                           reflect_on_tool_use=False, max_tool_iterations=10)
    answers = []
    for turn in case["turns"]:
        if entity_memory:
            f = OID.findall(turn)
            if f:
                b.state.facts = {**b.state.facts, "last_order": f[-1]}
        res = await agent.run(task=turn)
        final = res.messages[-1]
        txt = getattr(final, "content", "")
        answers.append(txt if isinstance(txt, str) else str(txt))
        if answers[-1].startswith("Which order"):
            b.events["asked_for_info"] = True
    log = await tools.call("get_action_log", {})
    return b.result(answers, log)
