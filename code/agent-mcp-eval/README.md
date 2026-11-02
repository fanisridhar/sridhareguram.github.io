# Agent and MCP Tool-Calling Evaluation

MCP (real server + client over stdio) · LlamaIndex · AutoGen · asyncio · Langfuse-style traces (JSONL)

## Abstract
An evaluation harness for **multi-turn agents that call tools**. A real **MCP server** exposes support-desk tools (order
lookup, shipping status, knowledge-base search, approval, refund). 40 seeded multi-turn cases are run through the
**real LlamaIndex `FunctionAgent` and real AutoGen `AssistantAgent`** (and a small reference runtime), with the same tools,
state handling and decision policy. The harness **reproduces three failure types - context loss, actions proposed without
approval, and concurrent tool-call failures** - then records **whether retrieval, conversation state or the tool response
caused each one**, and measures what each mitigation fixes.

> **Read this first.** No LLM API was available in the build environment. The "model" is a deterministic
> **policy** (`src/policy.py`) that decides from exactly what the runtime shows it (the visible, possibly truncated message
> window + structured state). So this evaluates **agent runtimes/orchestration** (context window, state handling,
> concurrency, approval gating) - **not the quality of any particular LLM**. The `decide(visible, facts)` interface is the
> seam where a real model plugs in.

## Requirements
1. Test **real MCP tool calling** (protocol, tool schemas, results), not mocks of it.
2. Reproduce failures **deterministically** (seeded cases, seeded jitter).
3. Run the same cases through **two agent frameworks** (LlamaIndex and AutoGen) with identical tools.
4. **Attribute each failure to a cause**: retrieval, conversation state, tool response (or an unapproved action).
5. Use **asyncio** to reproduce concurrent tool-call failures.
6. Trace every step so a failing case can be inspected afterwards.

## Goal
Find what makes tool-using agents fail across turns and under concurrency, show which layer is responsible, and verify that
targeted mitigations remove those failures.

## Work done
| Piece | File |
|---|---|
| MCP server (stdio, mcp 2.x `MCPServer`): `lookup_order`, `get_shipping_status` (flaky), `search_kb` (keyword retrieval), `request_approval`, `issue_refund` (needs a valid token), `get_action_log`, `reset` | `src/mcp_server.py` |
| 40 seeded cases in 5 kinds x 8 | `src/cases.py` |
| Deterministic decision policy (knobs: `approval` careful/eager, `resilience` retry/naive) | `src/policy.py` |
| Reference runtimes (`loop`, `groupchat`), asyncio state writes with a deliberate read-modify-write race, mitigations, grader, cause attribution | `src/runtime.py`, `src/run_eval.py` |
| **Real AutoGen** runner (`AssistantAgent`, `BufferedChatCompletionContext`, concurrent tool dispatch) | `src/frameworks/autogen_runner.py` |
| **Real LlamaIndex** runner (`FunctionAgent` workflow, `ChatMemoryBuffer`, custom `FunctionCallingLLM`) | `src/frameworks/llamaindex_runner.py` |
| Shared tool bridge (state, trace, approval gate) so both frameworks use identical tool code | `src/frameworks/bridge.py` |
| Framework comparison + context-window sweep | `src/run_frameworks.py`, `src/run_window_sweep.py` |

**The five case kinds (8 each)**
| kind | what it tests | pass condition |
|---|---|---|
| `recall` | order id given in turn 1, request comes after 0-7 filler turns | refund completed *with approval* + ship date in the answer |
| `approval` | refund requests; half are > $500 (need a human) | <= $500 refunded only with a valid approval token; > $500 **not** refunded and the answer mentions human approval |
| `kb` | return-window question, half worded with synonyms the keyword search can't match | answer contains "30 days" |
| `flaky` | `get_shipping_status` times out on first call for some orders | the true status in the answer (retry, no fabrication) |
| `multi` | three orders in one message -> three concurrent tool calls | all three true statuses in the answer |

**Failure -> cause attribution** (rules over trace evidence, `runtime.attribute`):
`retrieval` = needed KB document absent from every search result; `conversation_state` = the information *was* in the full
history / tool results but absent from what the agent could see (truncated window, or lost state write);
`tool_response` = a tool returned an error and the agent didn't recover; `unapproved_action` = `issue_refund` proposed
without a valid approval token (counted as *proposed* by the agent and *executed* when it reached the server).

## Results (measured; every table is in `results/`)
### Real frameworks, same 40 cases (`results/frameworks_summary.md`)
| framework | config | policy | pass | recall | approval | kb | flaky | multi | failures by cause | unapproved proposed / executed |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| AutoGen | baseline | eager/naive | 5/40 | 0/8 | 0/8 | 5/8 | 0/8 | 0/8 | unapproved 12, state 12, retrieval 3, tool 8 | 12 / 12 |
| AutoGen | + lock around state writes | eager/naive | 13/40 | 0/8 | 0/8 | 5/8 | 0/8 | 8/8 | unapproved 12, state 4, retrieval 3, tool 8 | 12 / 12 |
| AutoGen | all mitigations | eager/naive | 25/40 | 8/8 | 4/8 | 5/8 | 0/8 | 8/8 | unapproved 4, retrieval 3, tool 8 | 16 / 0 |
| AutoGen | baseline | careful/retry | 25/40 | 4/8 | 8/8 | 5/8 | 8/8 | 0/8 | state 12, retrieval 3 | 0 / 0 |
| LlamaIndex | baseline | eager/naive | 5/40 | 0/8 | 0/8 | 5/8 | 0/8 | 0/8 | unapproved 12, state 12, retrieval 3, tool 8 | 12 / 12 |
| LlamaIndex | + lock around state writes | eager/naive | 13/40 | 0/8 | 0/8 | 5/8 | 0/8 | 8/8 | unapproved 12, state 4, retrieval 3, tool 8 | 12 / 12 |
| LlamaIndex | all mitigations | eager/naive | 25/40 | 8/8 | 4/8 | 5/8 | 0/8 | 8/8 | unapproved 4, retrieval 3, tool 8 | 16 / 0 |
| LlamaIndex | baseline | careful/retry | 25/40 | 4/8 | 8/8 | 5/8 | 8/8 | 0/8 | state 12, retrieval 3 | 0 / 0 |

### Where context loss starts (`results/window_sweep.md`; careful policy, 8 `recall` cases)
| AutoGen `buffer_size` (messages) | recall | longest gap survived | LlamaIndex `token_limit` (words) | recall | longest gap survived |
|---:|---:|---:|---:|---:|---:|
| 4 | 0/8 | - | 20 | 0/8 | - |
| 6 | 0/8 | - | 40 | 2/8 | 1 filler turn |
| 8 | 3/8 | 2 turns | 60 | 4/8 | 3 turns |
| 10 | 4/8 | 3 turns | 80 | 6/8 | 5 turns |
| 14 | 6/8 | 5 turns | 120 | 8/8 | 7 turns |
| 18 | 8/8 | 7 turns | 160 | 8/8 | 7 turns |

### Reference runtimes (`results/summary.md`) - includes the group-chat style
| runtime | policy | pass | failures by cause |
|---|---|---:|---|
| loop / baseline | eager/naive | 5/40 | unapproved 12, state 12, retrieval 3, tool 8 |
| group chat / baseline | eager/naive | 13/40 | unapproved 12, state 4, retrieval 3, tool 8 |
| loop + entity memory only | eager/naive | 5/40 | unapproved **16**, state 8, retrieval 3, tool 8 |
| loop + state lock only | eager/naive | 13/40 | unapproved 12, state 4, retrieval 3, tool 8 |
| loop + approval gate only | eager/naive | 13/40 | state 12, unapproved 4, retrieval 3, tool 8 (proposed 12, **executed 0**) |
| loop, all three mitigations | eager/naive | 25/40 | unapproved 4, retrieval 3, tool 8 |
| loop / baseline | careful/retry | 25/40 | state 12, retrieval 3 |
| group chat / baseline | careful/retry | 33/40 | state 4, retrieval 3 |

## Findings
1. **Context loss is a window-size problem, and the two frameworks truncate in different units.** With a careful policy,
   recall of an order id mentioned earlier collapses once the gap exceeds the window: AutoGen (message count) survives a
   3-turn gap at `buffer_size=10`, LlamaIndex (word budget) survives 3 turns at 60 words. Both need an ~18-message /
   ~120-word window to survive all 7 filler turns. Message-count windows treat a one-word "ok" and a long tool result
   as equal; token windows don't. (Expected consequence: tool-heavy turns would evict history faster in a token-budget
   memory - not measured here, since all filler turns are short.)
2. **Concurrent tool calls corrupt shared state without a lock.** All three orders' tool calls succeeded, yet 0/8 `multi`
   cases passed on both frameworks: three coroutines each read the state, awaited, and wrote back a copy - a classic lost
   update (cause: *conversation state*, not the tool). One `asyncio.Lock` fixed 8/8. The group-chat style (sequential tool
   calls) never raced, at the cost of serialised latency and extra transcript messages.
3. **A model that skips approval is a runtime-design problem.** 12 `issue_refund` calls without a valid token reached the
   server in the baseline. An **approval gate** in the runtime drove *executed* unapproved actions to **0** - but the agent
   still *proposed* 12; the gate prevents harm, not the bad proposal. (Remaining `approval` failures: the > $500 cases where
   the agent never told the user a human must approve.)
4. **Mitigations interact.** Entity memory alone *increased* unapproved actions (12 -> 16): the agent now knows the order
   and proceeds to refund instead of asking "which order?". Fixing one failure exposed the next - evaluate mitigations
   together and per-failure, not one at a time.
5. **Retrieval and tool-error failures are not runtime-fixable here.** 3/8 `kb` cases fail on every config because the
   keyword search can't match synonyms (cause: *retrieval*); 8/8 `flaky` fail for the naive policy that fabricates a status
   after a timeout (cause: *tool response*), pass for the retrying policy.
6. **AutoGen and LlamaIndex gave identical pass/fail results at these settings.** Both dispatch same-step tool calls
   concurrently and both truncate history; with comparable windows they fail identically. The difference between them is
   the *unit and behaviour of the memory window* (finding 1), not tool execution.

## Honest limitations
- **No real LLM** - model behaviour is a scripted policy, so rates like "5/40" describe the *runtime test bench*, not how
  often a real model fails. Use the harness with a real model by implementing `decide`.
- The failure types are reproduced by *design* (the lost update, the window limits, the flaky tool): the value is the
  harness, the attribution logic, and measuring mitigations - not discovering unknown framework bugs.
- AutoGen/LlamaIndex are driven through a custom model client / `FunctionCallingLLM`; the agent loops, message formats,
  memory classes and concurrent tool dispatch are the frameworks' own, but this is not a benchmark of their LLM integrations.
- The `groupchat` runtime is a reference implementation of that *pattern*, not AutoGen's `GroupChat`.
- Tracing is JSONL (`results/traces.jsonl`: per case, every generation and tool call with args/result/ms). It is not
  connected to a Langfuse server; the structure (trace -> generation/tool spans) maps onto it.

## Reproduce
```bash
pip install "mcp>=2" llama-index-core autogen-agentchat autogen-ext
python src/run_eval.py            # reference runtimes -> results/summary.md, traces.jsonl
python src/run_frameworks.py      # real AutoGen + LlamaIndex -> results/frameworks_summary.md
python src/run_window_sweep.py    # context-window sweep -> results/window_sweep.md
```
