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
