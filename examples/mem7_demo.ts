/**
 * mem7 Demo — Ollama + Upstash Vector (TypeScript)
 *
 * Prerequisites:
 *   1. Ollama running with models pulled:
 *        ollama serve
 *        ollama pull qwen2.5:7b
 *        ollama pull mxbai-embed-large
 *   2. Environment variables set:
 *        UPSTASH_VECTOR_REST_URL=https://your-index.upstash.io
 *        UPSTASH_VECTOR_REST_TOKEN=your-token
 *   3. Install the package:
 *        npm install @mem7ai/mem7
 *
 * Run:
 *   npx tsx examples/mem7_demo.ts
 */

import { MemoryEngine } from "@mem7ai/mem7";

function pp(obj: unknown): void {
  console.log(JSON.stringify(obj, null, 2));
}

async function main() {
  // ── 1. Configure mem7 ────────────────────────────────────────────
  console.log("=== Initializing mem7 ===\n");

  const engine = await MemoryEngine.create(
    JSON.stringify({
      llm: {
        provider: "ollama",
        base_url: "http://localhost:11434/v1",
        api_key: "ollama",
        model: "qwen2.5:7b",
        temperature: 0.0,
        max_tokens: 2000,
      },
      embedding: {
        provider: "ollama",
        base_url: "http://localhost:11434/v1",
        api_key: "ollama",
        model: "mxbai-embed-large",
        dims: 1024,
      },
      vector: {
        provider: "upstash",
        collection_name: "mem7-ts-test",
        dims: 1024,
        upstash_url: process.env.UPSTASH_VECTOR_REST_URL!,
        upstash_token: process.env.UPSTASH_VECTOR_REST_TOKEN!,
      },
      history: {
        db_path: ":memory:",
      },
    })
  );

  console.log("mem7 initialized\n");

  // ── 2. Reset (clean slate) ───────────────────────────────────────
  await engine.reset();
  console.log("reset done\n");

  // ── 3. Add memories from conversations ───────────────────────────
  console.log("=== Add memories ===\n");

  const result = await engine.add(
    [
      {
        role: "user",
        content:
          "I'm working on improving my tennis skills. I play twice a week.",
      },
      {
        role: "assistant",
        content:
          "That's great! Consistent practice is key. What areas are you focusing on?",
      },
      {
        role: "user",
        content:
          "I'm focusing on my backhand and serve. I also recently started taking lessons from a coach named Sarah.",
      },
    ],
    "alice"
  );
  console.log("Add result:");
  pp(result);
  console.log();

  const result2 = await engine.add(
    [
      {
        role: "user",
        content:
          "I recently moved to San Francisco and I love hiking in Marin.",
      },
    ],
    "alice"
  );
  console.log("Add result 2:");
  pp(result2);
  console.log();

  // ── 4. Search memories ───────────────────────────────────────────
  console.log("=== Search memories ===\n");

  const searchResults = await engine.search(
    "What sports does Alice play?",
    "alice",
    undefined,
    undefined,
    3
  );
  console.log("Search results:");
  pp(searchResults);
  console.log();

  // ── 5. Get all memories ──────────────────────────────────────────
  console.log("=== All memories ===\n");

  const allMemories = await engine.getAll("alice");
  console.log(`Total memories for alice: ${allMemories.length}`);
  pp(allMemories);
  console.log();

  // ── 6. Update with deduplication ─────────────────────────────────
  console.log("=== Add with deduplication ===\n");

  const result3 = await engine.add(
    [
      {
        role: "user",
        content:
          "I now play tennis three times a week and my coach Sarah says my backhand is improving a lot.",
      },
    ],
    "alice"
  );
  console.log("Add with dedup:");
  pp(result3);
  console.log();

  const allAfter = await engine.getAll("alice");
  console.log(`Total memories after dedup: ${allAfter.length}`);
  pp(allAfter);
  console.log();

  // ── 7. View history ──────────────────────────────────────────────
  console.log("=== History ===\n");

  if (allAfter.length > 0) {
    const firstId = allAfter[0].id;
    const history = await engine.history(firstId);
    console.log(`History for memory ${firstId}:`);
    pp(history);
    console.log();
  }

  // ── 8. Manual update & delete ────────────────────────────────────
  console.log("=== Manual update & delete ===\n");

  if (allAfter.length > 0) {
    const targetId = allAfter[0].id;

    console.log("Before update:");
    pp(await engine.get(targetId));

    await engine.update(
      targetId,
      "Alice is an advanced tennis player who trains daily."
    );
    console.log("\nAfter update:");
    pp(await engine.get(targetId));

    console.log("\nHistory after manual update:");
    pp(await engine.history(targetId));
    console.log();
  }

  if (allAfter.length > 1) {
    const delId = allAfter[allAfter.length - 1].id;
    console.log(`Deleting memory ${delId}`);
    await engine.delete(delId);
    const remaining = await engine.getAll("alice");
    console.log(`Remaining memories: ${remaining.length}\n`);
  }

  // ── 9. Multi-user isolation ──────────────────────────────────────
  console.log("=== Multi-user isolation ===\n");

  await engine.add(
    [
      {
        role: "user",
        content:
          "I'm a software engineer working on distributed systems in Rust.",
      },
    ],
    "bob"
  );

  console.log("Bob's memories:");
  pp(await engine.getAll("bob"));

  console.log("\nAlice's memories (unchanged):");
  pp(await engine.getAll("alice"));

  console.log('\nSearch "tennis" as Bob (expect nothing):');
  pp(await engine.search("tennis", "bob"));
  console.log();

  // ── 10. Cleanup ──────────────────────────────────────────────────
  await engine.reset();
  console.log("All data cleared");
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
