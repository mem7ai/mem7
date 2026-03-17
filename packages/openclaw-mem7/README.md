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
| `autoCapture`     | boolean | `true`               | Extract and store facts after each agent turn                   |
| `userId`          | string  | session-derived      | Override user ID for memory isolation                           |
| `dbPath`          | string  | `~/.openclaw/mem7`   | Base directory for SQLite history DB and graph data              |

## How It Works

### Auto-Recall (`before_prompt_build`)

Before each agent turn, the plugin:

1. Extracts the latest user message
2. Calls `mem7.search()` with the message as query
3. Formats the top memories and graph relations into a context block
4. Injects it as a system prompt prepend

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
2. Sends them through mem7's fact extraction pipeline
3. New facts are stored; duplicates are merged; stale facts are updated or replaced

This runs as fire-and-forget — errors are logged but never block the response.

### Tools

| Tool             | Description                                          |
| ---------------- | ---------------------------------------------------- |
| `memory_search`  | Search memories by semantic query (with decay scoring) |
| `memory_get`     | Retrieve a specific memory by ID, or list all        |
| `memory_store`   | Explicitly store a fact (bypasses auto-capture)      |

### Forgetting Curve

Unlike standalone mem7 where decay is opt-in, the plugin **enables decay by default** because OpenClaw's long-running sessions accumulate stale facts quickly.

Memories that are frequently recalled decay slower (spaced-repetition effect). Memories that haven't been accessed in weeks are deprioritized but never fully removed.

See the [mem7 README](../../README.md#memory-decay-forgetting-curve) for the full mathematical model.

## Development

```bash
cd packages/openclaw-mem7
npm install
npm run build
```

## License

Apache-2.0
