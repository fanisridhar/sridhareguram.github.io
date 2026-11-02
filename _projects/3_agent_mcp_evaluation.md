---
layout: page
title: Agent and MCP Tool-Calling Evaluation
description: Mercor | Langfuse, MCP, LlamaIndex, AutoGen, asyncio
importance: 3
category: work
---

Evaluated how reliably multi-turn agents call tools, and where their failures originate.

- Evaluated multi-turn agent output and MCP tool and function calling with Langfuse traces, reproducing context loss and actions proposed without approval.
- Compared LlamaIndex and AutoGen workflows on the same cases, using asyncio to reproduce concurrent tool-call failures and recording whether retrieval, conversation state or the tool response caused them.

**Stack:** Langfuse, MCP, LlamaIndex, AutoGen, asyncio

## Reproducible build

A real **MCP server** (support-desk tools) and 40 seeded multi-turn cases run through the **real LlamaIndex `FunctionAgent` and AutoGen `AssistantAgent`**, with identical tools, a deterministic decision policy standing in for the LLM (no LLM API was available), and a grader that attributes every failure to **retrieval, conversation state, tool response or an unapproved action**.

| finding (measured) | result |
|---|---|
| concurrent tool calls writing shared state (asyncio lost update) | 0 / 8 multi-order cases pass → **8 / 8** with a lock |
| unapproved refunds reaching the server | 12 → **0** with a runtime approval gate (the agent still *proposes* them) |
| context loss vs window size | AutoGen survives a 3-turn gap at `buffer_size=10`; needs 18 messages for 7 turns. LlamaIndex needs ~120 words |
| AutoGen vs LlamaIndex, same cases | identical pass/fail at comparable settings; they differ in the unit of the memory window (messages vs tokens) |
| all three mitigations together | 5 / 40 → 25 / 40 |

Mitigations interact: adding entity memory alone raised unapproved actions (12 → 16) because the agent could now proceed with a refund it previously asked about.

Source code, results and full write-up: [`code/agent-mcp-eval`](https://github.com/fanisridhar/sridhareguram.github.io/tree/master/code/agent-mcp-eval).
