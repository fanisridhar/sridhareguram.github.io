# Project code

Four runnable projects. Each folder has its own README with abstract, requirements, goal, work done, measured results,
findings and limitations. Every number in those READMEs was produced by the code in that folder.

| folder | what it is | what is real vs simulated |
|---|---|---|
| `semantic-retrieval-benchmark/` | 50-query benchmark + latency tuning of a semantic search service | real code and measurements on a synthetic corpus |
| `llm-inference-benchmark/` | load-test / profiling / quantization-check harness for vLLM and TensorRT-LLM | harness is real; **numbers come from a simulated server** (no GPU available) |
| `agent-mcp-eval/` | tool-calling evaluation on a real MCP server with real LlamaIndex and AutoGen agents | real MCP + frameworks; the **LLM is a scripted policy** |
| `inventory-anomaly-poc/` | anomaly detection model comparison, Flask review site, SQL reconciliation | real models on **synthetic labelled data**; Azure ML files are untested templates |

These reproduce *methods* in self-contained settings. Numbers here are not the numbers from any employer engagement.
