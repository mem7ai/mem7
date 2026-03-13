# mem7

LLM-powered long-term memory engine — Rust core with multi-language bindings.

mem7 extracts factual statements from conversations, deduplicates them against existing memories, and stores the results in a vector database with full audit history.

## Install

```bash
pip install mem7          # Python
npm install @mem7ai/mem7  # Node.js / TypeScript
cargo add mem7            # Rust
```

## Architecture

```
Python / TypeScript / Rust API
    │  PyO3 (sync + async) / napi-rs / native
    ▼
Rust Core (tokio async runtime)
    ├── mem7-llm        — OpenAI-compatible LLM client
    ├── mem7-embedding  — OpenAI-compatible embedding client
    ├── mem7-vector     — Vector index (FlatIndex / Upstash)
    ├── mem7-graph      — Graph store (FlatGraph / Kuzu / Neo4j)
    ├── mem7-history    — SQLite audit trail
    ├── mem7-dedup      — LLM-driven memory deduplication
    ├── mem7-reranker   — Search reranking (Cohere / LLM-based)
    └── mem7-store      — Pipeline orchestrator (MemoryEngine)
```

## Quick Start (Python — Sync)

```python
from mem7 import Memory
from mem7.config import MemoryConfig, LlmConfig, EmbeddingConfig

config = MemoryConfig(
    llm=LlmConfig(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen2.5:7b",
    ),
    embedding=EmbeddingConfig(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="mxbai-embed-large",
        dims=1024,
    ),
)

m = Memory(config=config)
m.add("I love playing tennis and my coach is Sarah.", user_id="alice")
results = m.search("What sports does Alice play?", user_id="alice")
```

## Quick Start (Python — Async)

```python
import asyncio
from mem7 import AsyncMemory
from mem7.config import MemoryConfig, LlmConfig, EmbeddingConfig

async def main():
    config = MemoryConfig(
        llm=LlmConfig(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="qwen2.5:7b",
        ),
        embedding=EmbeddingConfig(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="mxbai-embed-large",
            dims=1024,
        ),
    )

    m = await AsyncMemory.create(config=config)
    await m.add("I love playing tennis and my coach is Sarah.", user_id="alice")
    results = await m.search("What sports does Alice play?", user_id="alice")

asyncio.run(main())
```

## Quick Start (TypeScript)

```typescript
import { MemoryEngine } from "@mem7ai/mem7";

const engine = await MemoryEngine.create(JSON.stringify({
  llm: { base_url: "http://localhost:11434/v1", api_key: "ollama", model: "qwen2.5:7b" },
  embedding: { base_url: "http://localhost:11434/v1", api_key: "ollama", model: "mxbai-embed-large", dims: 1024 },
}));

await engine.add([{ role: "user", content: "I love playing tennis and my coach is Sarah." }], "alice");
const results = await engine.search("What sports does Alice play?", "alice");
```

## Supported Providers

mem7 uses a single **OpenAI-compatible client** for both LLM and Embedding, which covers any service that exposes the OpenAI API format. This includes most major providers out of the box.

### LLMs

| Provider | mem0 | mem7 | Notes |
|----------|:----:|:----:|-------|
| OpenAI | :white_check_mark: | :white_check_mark: | Native support |
| Ollama | :white_check_mark: | :white_check_mark: | Via OpenAI-compatible API |
| vLLM | :white_check_mark: | :white_check_mark: | Via OpenAI-compatible API |
| Groq | :white_check_mark: | :white_check_mark: | Via OpenAI-compatible API |
| Together | :white_check_mark: | :white_check_mark: | Via OpenAI-compatible API |
| DeepSeek | :white_check_mark: | :white_check_mark: | Via OpenAI-compatible API |
| xAI (Grok) | :white_check_mark: | :white_check_mark: | Via OpenAI-compatible API |
| LM Studio | :white_check_mark: | :white_check_mark: | Via OpenAI-compatible API |
| Azure OpenAI | :white_check_mark: | :white_check_mark: | Via OpenAI-compatible API |
| Anthropic | :white_check_mark: | :x: | Requires native SDK |
| Gemini | :white_check_mark: | :x: | Requires native SDK |
| Vertex AI | :white_check_mark: | :x: | Requires native SDK |
| AWS Bedrock | :white_check_mark: | :x: | Requires native SDK |
| LiteLLM | :white_check_mark: | :x: | Python proxy |
| Sarvam | :white_check_mark: | :x: | Requires native SDK |
| LangChain | :white_check_mark: | :x: | Python framework |

### Embeddings

| Provider | mem0 | mem7 | Notes |
|----------|:----:|:----:|-------|
| OpenAI | :white_check_mark: | :white_check_mark: | Native support |
| Ollama | :white_check_mark: | :white_check_mark: | Via OpenAI-compatible API |
| Together | :white_check_mark: | :white_check_mark: | Via OpenAI-compatible API |
| LM Studio | :white_check_mark: | :white_check_mark: | Via OpenAI-compatible API |
| Azure OpenAI | :white_check_mark: | :white_check_mark: | Via OpenAI-compatible API |
| Hugging Face | :white_check_mark: | :x: | Requires native SDK |
| Gemini | :white_check_mark: | :x: | Requires native SDK |
| Vertex AI | :white_check_mark: | :x: | Requires native SDK |
| AWS Bedrock | :white_check_mark: | :x: | Requires native SDK |
| FastEmbed | :white_check_mark: | :x: | Python-only (ONNX) |
| LangChain | :white_check_mark: | :x: | Python framework |

### Vector Stores

| Provider | mem0 | mem7 | Notes |
|----------|:----:|:----:|-------|
| In-memory (FlatIndex) | — | :white_check_mark: | Built-in, good for dev |
| Upstash Vector | :white_check_mark: | :white_check_mark: | REST API, serverless |
| Qdrant | :white_check_mark: | :x: | |
| Chroma | :white_check_mark: | :x: | |
| pgvector | :white_check_mark: | :x: | |
| Milvus | :white_check_mark: | :x: | |
| Pinecone | :white_check_mark: | :x: | |
| Redis | :white_check_mark: | :x: | |
| Weaviate | :white_check_mark: | :x: | |
| Elasticsearch | :white_check_mark: | :x: | |
| OpenSearch | :white_check_mark: | :x: | |
| FAISS | :white_check_mark: | :x: | |
| MongoDB | :white_check_mark: | :x: | |
| Supabase | :white_check_mark: | :x: | |
| Azure AI Search | :white_check_mark: | :x: | |
| Vertex AI Vector Search | :white_check_mark: | :x: | |
| Databricks | :white_check_mark: | :x: | |
| Cassandra | :white_check_mark: | :x: | |
| S3 Vectors | :white_check_mark: | :x: | |
| Baidu | :white_check_mark: | :x: | |
| Neptune | :white_check_mark: | :x: | |
| Valkey | :white_check_mark: | :x: | |
| LangChain | :white_check_mark: | :x: | |

### Rerankers

| Provider | mem0 | mem7 | Notes |
|----------|:----:|:----:|-------|
| Cohere | :white_check_mark: | :white_check_mark: | Cohere v2 rerank API |
| LLM-based | :white_check_mark: | :white_check_mark: | Any OpenAI-compatible LLM |
| Jina AI | :white_check_mark: | :x: | Planned |
| Cross-encoder | :white_check_mark: | :x: | Planned |

### Graph Stores

| Provider | mem0 | mem7 | Notes |
|----------|:----:|:----:|-------|
| In-memory (FlatGraph) | — | :white_check_mark: | Built-in, good for dev/testing |
| Kuzu (embedded) | :white_check_mark: | :white_check_mark: | Cypher-based, no server needed (feature flag `kuzu`) |
| Neo4j | :white_check_mark: | :white_check_mark: | Production-grade, Bolt protocol |
| Memgraph | :white_check_mark: | :x: | Planned |
| Amazon Neptune | :white_check_mark: | :x: | Planned |

### Language Bindings

| Language | Status |
|----------|--------|
| Python (sync + async) | :white_check_mark: PyPI: `pip install mem7` |
| TypeScript / Node.js | :white_check_mark: npm: `npm install @mem7ai/mem7` |
| Rust | :white_check_mark: crates.io: `cargo add mem7` |
| Go | Planned |

## Vector Store Backends

**Built-in FlatIndex** (default) — in-memory brute-force, good for development:

```python
from mem7.config import VectorConfig

VectorConfig(provider="flat", dims=1024)
```

**Upstash Vector** — managed cloud vector database:

```python
VectorConfig(
    provider="upstash",
    collection_name="my-namespace",
    dims=1024,
    upstash_url="https://your-index.upstash.io",
    upstash_token="your-token",
)
```

## Graph Memory (Dual-Path Recall)

When `graph` is configured, mem7 runs **dual-path recall**: vector search and graph search execute concurrently via `tokio::join!`, returning both factual memories and entity relations.

On `add()`, the engine extracts entities and relations from conversations using LLM (JSON mode) and stores them in the graph alongside the vector memories.

**FlatGraph** (in-memory, for development):

```python
from mem7 import Memory
from mem7.config import MemoryConfig, LlmConfig, EmbeddingConfig, GraphConfig

config = MemoryConfig(
    llm=LlmConfig(base_url="http://localhost:11434/v1", api_key="ollama", model="qwen2.5:7b"),
    embedding=EmbeddingConfig(base_url="http://localhost:11434/v1", api_key="ollama", model="mxbai-embed-large", dims=1024),
    graph=GraphConfig(provider="flat"),
)

m = Memory(config=config)
m.add("I love playing tennis and my coach is Sarah.", user_id="alice")

results = m.search("What sports does Alice play?", user_id="alice")
# results["memories"]   -> vector search results
# results["relations"]  -> graph relations (e.g. USER -[loves_playing]-> tennis)
```

**Neo4j** (production):

```python
GraphConfig(
    provider="neo4j",
    neo4j_url="bolt://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="password",
)
```

**Kuzu** (embedded, requires `kuzu` feature flag):

```python
GraphConfig(provider="kuzu", kuzu_db_path="./my_graph.kuzu")
```

The graph LLM can be configured separately (e.g. use a cheaper model for extraction):

```python
GraphConfig(
    provider="flat",
    llm=LlmConfig(base_url="http://localhost:11434/v1", api_key="ollama", model="qwen2.5:3b"),
)
```

## Examples

See the [`examples/`](examples/) directory:
- [`mem7_demo.ipynb`](examples/mem7_demo.ipynb) — Python notebook demo
- [`mem7_demo.ts`](examples/mem7_demo.ts) — TypeScript demo

## Development

### Prerequisites

- Rust 1.85+ (stable)
- Python 3.10+
- Node.js 22+
- [maturin](https://github.com/PyO3/maturin)

### Build

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin pydantic

# Development build (debug, fast iteration)
maturin develop

# Release build
maturin develop --release
```

### Test

```bash
# Rust tests
cargo test --workspace

# Clippy
cargo clippy --workspace --all-targets -- -D warnings
```

## License

Apache-2.0
