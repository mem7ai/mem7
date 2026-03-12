# mem7

LLM-powered long-term memory engine — Rust core with Python bindings.

mem7 extracts factual statements from conversations, deduplicates them against existing memories, and stores the results in a vector database with full audit history.

## Architecture

```
Python API (mem7.Memory)
    │  JSON over PyO3
    ▼
Rust Core (tokio async runtime)
    ├── mem7-llm        — OpenAI-compatible LLM client
    ├── mem7-embedding  — OpenAI-compatible embedding client
    ├── mem7-vector     — Vector index (FlatIndex / Upstash)
    ├── mem7-history    — SQLite audit trail
    ├── mem7-dedup      — LLM-driven memory deduplication
    └── mem7-store      — Pipeline orchestrator (MemoryEngine)
```

## Quick Start

```bash
pip install mem7
```

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

# Add memories from conversation
m.add("I love playing tennis and my coach is Sarah.", user_id="alice")

# Semantic search
results = m.search("What sports does Alice play?", user_id="alice")
print(results)
```

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

## Development

### Prerequisites

- Rust 1.85+ (stable)
- Python 3.10+
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
