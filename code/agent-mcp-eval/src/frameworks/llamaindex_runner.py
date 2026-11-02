"""Run the evaluation cases through REAL LlamaIndex (FunctionAgent workflow).

Real parts: FunctionAgent's workflow loop, its concurrent tool-call dispatch
(call_tool workflow steps), its scratchpad/tool-message format, and
ChatMemoryBuffer (its token-budget context window).
Stand-in part: the LLM -- a FunctionCallingLLM subclass that delegates the
decision to the deterministic policy instead of calling a model.

Note on context windows: AutoGen's buffer is counted in MESSAGES, LlamaIndex's
in TOKENS (here whitespace words, because the default tokenizer downloads data
from the internet). Defaults are chosen to be roughly comparable (~5 short turns).
"""
import json
import sys
from pathlib import Path
from typing import Any, List, Sequence

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, LLMMetadata
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import MessageRole
from llama_index.core.llms.llm import ToolSelection
from pydantic import ConfigDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bridge import Bridge  # noqa: E402
from policy import OID  # noqa: E402


def to_visible(messages):
    vis, calls = [], {}
    for m in messages:
        role = m.role.value if hasattr(m.role, "value") else str(m.role)
        text = m.content or ""
        if role == "user":
            vis.append({"role": "user", "content": text})
        elif role == "assistant":
            for tc in m.additional_kwargs.get("tool_calls", []):
                calls[tc["id"]] = (tc["name"], tc["args"])
            vis.append({"role": "assistant", "content": text or "[tool calls]"})
        elif role == "tool":
            name, args = calls.get(m.additional_kwargs.get("tool_call_id"), ("unknown", {}))
            vis.append({"role": "tool", "name": name, "args": args, "content": text})
    return vis


class PolicyLLM(FunctionCallingLLM):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    policy: Any = None
    bridge: Any = None
    n: int = 0

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=True, is_chat_model=True, context_window=10**6, num_output=256)

    def _prepare_chat_with_tools(self, tools, user_msg=None, chat_history=None, verbose=False, allow_parallel_tool_calls=False, **kw):
        msgs = list(chat_history or [])
        if user_msg:
            msgs.append(user_msg if isinstance(user_msg, ChatMessage) else ChatMessage(role="user", content=user_msg))
        return {"messages": msgs}

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        act = self.policy.decide(to_visible(messages), dict(self.bridge.state.facts))
        self.n += 1
        if act.kind == "final":
            return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=act.text))
        tcs = [{"id": f"call-{self.n}-{i}", "name": t, "args": a} for i, (t, a) in enumerate(act.calls)]
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content="", additional_kwargs={"tool_calls": tcs}))

    def chat(self, messages, **kwargs):
        raise NotImplementedError

    def get_tool_calls_from_response(self, response, error_on_no_tool_call=True, **kwargs) -> List[ToolSelection]:
        return [ToolSelection(tool_id=t["id"], tool_name=t["name"], tool_kwargs=t["args"])
                for t in response.message.additional_kwargs.get("tool_calls", [])]

    # unused abstract surface
    def complete(self, *a, **k): raise NotImplementedError
    async def acomplete(self, *a, **k): raise NotImplementedError
    def stream_chat(self, *a, **k): raise NotImplementedError
    def stream_complete(self, *a, **k): raise NotImplementedError
    async def astream_chat(self, *a, **k): raise NotImplementedError
    async def astream_complete(self, *a, **k): raise NotImplementedError


async def run_case(case, tools, policy, token_window=60, entity_memory=False, safe_state=False, approval_gate=False):
    await tools.call("reset", {})
    b = Bridge(tools, case, safe_state, approval_gate)

    async def lookup_order(order_id: str) -> str:
        """Look up order details."""
        return await b.call("lookup_order", order_id=order_id)

    async def get_shipping_status(order_id: str) -> str:
        """Get shipping status for an order."""
        return await b.call("get_shipping_status", order_id=order_id)

    async def search_kb(query: str) -> str:
        """Search the knowledge base."""
        return await b.call("search_kb", query=query)

    async def request_approval(action: str, order_id: str, amount: float) -> str:
        """Request approval for an action."""
        return await b.call("request_approval", action=action, order_id=order_id, amount=amount)

    async def issue_refund(order_id: str, amount: float, approval_token: str = "") -> str:
        """Refund an order (needs approval token)."""
        return await b.call("issue_refund", order_id=order_id, amount=amount, approval_token=approval_token)

    fts = [FunctionTool.from_defaults(async_fn=f) for f in (lookup_order, get_shipping_status, search_kb, request_approval, issue_refund)]
    llm = PolicyLLM(policy=policy, bridge=b)
    agent = FunctionAgent(tools=fts, llm=llm, streaming=False, system_prompt="support agent")
    memory = ChatMemoryBuffer.from_defaults(token_limit=token_window, tokenizer_fn=lambda s: s.split())
    answers = []
    for turn in case["turns"]:
        if entity_memory:
            f = OID.findall(turn)
            if f:
                b.state.facts = {**b.state.facts, "last_order": f[-1]}
        out = await agent.run(user_msg=turn, memory=memory)
        txt = out.response.content or ""
        answers.append(txt)
        if txt.startswith("Which order"):
            b.events["asked_for_info"] = True
    log = await tools.call("get_action_log", {})
    return b.result(answers, log)
