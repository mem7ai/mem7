# @mem7ai/openclaw-mem7

[OpenClaw](https://github.com/nicepkg/openclaw) memory plugin powered by [mem7](https://github.com/mem7ai/mem7).

Replaces the built-in `memory-core` with LLM-powered fact extraction, vector + graph dual-path recall, automatic deduplication, and an Ebbinghaus forgetting curve — all running on mem7's Rust core via napi-rs.

## Install

```bash
openclaw plugins install @mem7ai/openclaw-mem7
```

## Configure

Add the plugin to `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "slots": { "memory": "openclaw-mem7" },
    "entries": {
      "openclaw-mem7": {
        "enabled": true,
        "config": {
          "llm": {
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "model": "qwen2.5:7b"
          },
          "embedding": {
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "model": "mxbai-embed-large",
            "dims": 1024
          },
          "graph": { "provider": "flat" },
          "decay": { "enabled": true }
        }
      }
    }
  }
}
```

If your OpenClaw config already has an OpenAI provider key (`config.models.providers.openai.apiKey`), you can omit `api_key` from the `llm` and `embedding` sections — the plugin will resolve them automatically.

## Config Reference

| Key               | Type    | Default              | Description                                                     |
| ----------------- | ------- | -------------------- | --------------------------------------------------------------- |
| `llm`             | object  | *required*           | LLM for fact extraction and dedup (OpenAI-compatible)           |
| `embedding`       | object  | *required*           | Embedding provider config                                       |
| `vector`          | object  | `{ provider: "flat" }` | Vector store backend (`flat` or `upstash`)                   |
| `graph`           | object  | *disabled*           | Graph store backend (`flat`, `neo4j`, or `kuzu`)                |
| `decay`           | object  | `{ enabled: true }`  | Forgetting curve parameters                                     |
| `autoRecall`      | boolean | `true`               | Inject relevant memories before each agent turn                 |
| `autoRecallLimit` | integer | `5`                  | Max memories to inject via auto-recall                          |
| `topK`            | integer | `5`                  | Default result limit for tool-driven search/list                |
| `searchThreshold` | number  | unset                | Optional minimum score for plugin search/recall                 |
| `autoCapture`     | boolean | `true`               | Extract and store facts after each agent turn                   |
| `userId`          | string  | `"default"`          | Base user namespace for long-term memory                        |
| `dbPath`          | string  | `~/.openclaw/mem7`   | Base directory for SQLite history DB and graph data              |

## How It Works

### Auto-Recall (`before_prompt_build` / `before_agent_start`)

Before each agent turn, the plugin:

1. Extracts the latest user message
2. Searches both the current session scope and the broader long-term scope
3. Deduplicates the merged recall set
4. Formats the top memories and graph relations into a context block
5. Injects it as a system prompt prepend

The injected context looks like:

```
<mem7_context>
## Relevant memories about this user:
- [2026-03-15] Alice loves playing tennis (score: 0.92)
- [2026-03-10] Alice's coach is Sarah (score: 0.87)

## Known relations:
- Alice -[loves_playing]-> tennis
- Alice -[coached_by]-> Sarah
</mem7_context>
```

### Auto-Capture (`agent_end`)

After each successful turn, the plugin:

1. Extracts the user + assistant messages from the turn
2. Stores them into the current session scope (`runId = sessionKey` when available)
3. Tags stored memories with `metadata.source = "OPENCLAW"`
4. New facts are stored; duplicates are merged; stale facts are updated or replaced

This runs as fire-and-forget — errors are logged but never block the response.

### Tools

| Tool             | Description                                                       |
| ---------------- | ----------------------------------------------------------------- |
| `memory_search`  | Search memories by semantic query, with `scope` / `longTerm`      |
| `memory_list`    | List stored memories for the selected scope                       |
| `memory_store`   | Explicitly store a fact into session or long-term scope           |
| `memory_get`     | Retrieve a specific memory by ID; `path="all"` still lists all    |
| `memory_forget`  | Delete a specific memory by ID, or find delete candidates by query |

### Scope Model

The plugin supports three routing modes on search/list/forget tools:

- `scope: "session"` routes reads/writes to the current session when `sessionKey` is available.
- `scope: "long-term"` routes reads/writes to the broader user namespace without a session `runId`.
- `scope: "all"` searches or lists both scopes and merges the results.
- `longTerm: true` is a convenience alias for `scope: "long-term"`.

For `memory_store`, the default is long-term storage unless you pass `longTerm: false` or `scope: "session"`.

If the runtime `sessionKey` matches `agent:<agentId>:...`, the plugin automatically derives `agentId` for per-agent isolation. Session-scoped operations use the configured base `userId` plus the current `runId`; long-term operations omit `runId` but preserve the same `agentId`.

### Forgetting Curve

Unlike standalone mem7 where decay is opt-in, the plugin **enables decay by default** because OpenClaw's long-running sessions accumulate stale facts quickly.

Memories that are frequently recalled decay slower (spaced-repetition effect). Memories that haven't been accessed in weeks are deprioritized but never fully removed.

See the [mem7 README](../../README.md#memory-decay-forgetting-curve) for the full mathematical model.

## Development

```bash
# From the repository root
just openclaw-build
just lint
just typecheck
```

## License

Apache-2.0
