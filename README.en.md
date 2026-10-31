# astrbot_plugin_tool_context

Tool-call log archiving and context optimization for AstrBot.

[中文](README.md)

## Status

Under development. The local core implements session-scoped append-only JSONL records, a separate provider summarization adapter, text-result projection on request copies, persistent summary reuse, and scoped record retrieval. Nine offline tests pass using temporary files and a mock provider; no paid API calls were made.

The core is not connected to the AstrBot request loop. The plugin entry point remains a registration skeleton, so installation does not automatically clean context or expose log retrieval to the model. Runtime integration has not been validated.

## Goal

Preserve native tool invocation, archive complete tool arguments and results, and use a separate API to produce concise English action summaries that replace unnecessary verbose records in subsequent model requests.

The plugin targets tool-operation context only. It does not summarize ordinary conversation, modify personas or long-term memory, or speak on the agent's behalf.

## Constraints

- Keep original logs separate from model-request projections.
- Archive successfully before replacing content; preserve original results if archiving or summarization fails.
- Preserve tool-call pairing, errors, unknown outcomes, and necessary artifact locations.
- Do not unconditionally remove file contents, code, or search results before the agent has seen them.
- Do not globally monkey-patch providers.
- Treat logs as potentially sensitive data and explicitly control access and summarization API disclosure.

## Integration

The per-request projection seam in AstrBot is still being evaluated. A tool-response event alone does not provide a reliable way to replace input throughout the active tool loop.

Initial work targets text tool results. Images, signed reasoning, and background-task status receipts are not implemented.

## Reference

The design draws on log offloading and request summarization mechanisms in LangChain Deep Agents without migrating the full agent framework. No source from that project has been incorporated yet. Any future substantial source reuse must preserve its MIT copyright and license notice.
