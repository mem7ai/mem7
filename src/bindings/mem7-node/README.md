# @mem7ai/mem7

LLM-powered long-term memory engine — Rust core with Node.js bindings.

## Install

```bash
npm install @mem7ai/mem7
```

## Usage

```typescript
import { MemoryEngine } from '@mem7ai/mem7'

const engine = await MemoryEngine.create(JSON.stringify({
  llm: {
    base_url: "http://localhost:11434/v1",
    api_key: "ollama",
    model: "qwen2.5:7b",
  },
  embedding: {
    base_url: "http://localhost:11434/v1",
    api_key: "ollama",
    model: "mxbai-embed-large",
    dims: 1024,
  },
}))

// Add memories from conversation
const result = await engine.add(
  [{ role: "user", content: "I love playing tennis and my coach is Sarah." }],
  "alice",
)

// Semantic search
const searchResult = await engine.search("What sports does Alice play?", "alice")
console.log(searchResult.memories)
```

## Supported Platforms

| Platform          | Architecture |
|-------------------|-------------|
| macOS             | x64, ARM64  |
| Linux (glibc)    | x64, ARM64  |
| Windows           | x64         |

## License

Apache-2.0
