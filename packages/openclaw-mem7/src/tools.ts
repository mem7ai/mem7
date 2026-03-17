import { Type } from "@sinclair/typebox";
import type { OpenClawPluginApi, PluginConfig } from "./config.js";
import { resolveUserId } from "./config.js";
import {
  formatSearchResult,
  formatMemoryItem,
  formatMemoryList,
  formatAddResult,
} from "./format.js";
import type { Mem7Engine, EngineGetter } from "./engine.js";

function requireEngine(getEngine: EngineGetter): Mem7Engine {
  const engine = getEngine();
  if (!engine) {
    throw new Error("mem7 engine is not initialized yet");
  }
  return engine;
}

export function registerTools(
  api: OpenClawPluginApi,
  getEngine: EngineGetter,
  cfg: PluginConfig
): void {
  api.registerTool({
    id: "memory_search",
    name: "memory_search",
    description:
      "Search long-term memory for facts, preferences, and relations relevant to a query. " +
      "Returns scored memories and graph relations.",
    parameters: Type.Object({
      query: Type.String({ description: "The search query" }),
      limit: Type.Optional(
        Type.Integer({
          description: "Max number of results",
          default: 5,
          minimum: 1,
          maximum: 50,
        })
      ),
    }),
    async execute(args: { query: string; limit?: number }) {
      const engine = requireEngine(getEngine);
      const userId = resolveUserId(cfg, api);
      const result = await engine.search(
        args.query,
        userId,
        null,
        null,
        args.limit ?? 5,
        null,
        true,
        null
      );
      return formatSearchResult(result);
    },
  });

  api.registerTool({
    id: "memory_get",
    name: "memory_get",
    description:
      'Retrieve a specific memory by ID, or pass path="all" to list all stored memories.',
    parameters: Type.Object({
      path: Type.String({
        description: 'Memory ID (UUID) or "all" to list all memories',
      }),
      from: Type.Optional(
        Type.Integer({ description: "Pagination offset", minimum: 0 })
      ),
      lines: Type.Optional(
        Type.Integer({
          description: "Number of results to return",
          minimum: 1,
          maximum: 100,
        })
      ),
    }),
    async execute(args: { path: string; from?: number; lines?: number }) {
      const engine = requireEngine(getEngine);
      const userId = resolveUserId(cfg, api);

      if (args.path === "all") {
        const items = await engine.getAll(
          userId,
          null,
          null,
          null,
          args.lines ?? 50
        );
        return formatMemoryList(items);
      }

      const item = await engine.get(args.path);
      return formatMemoryItem(item);
    },
  });

  api.registerTool({
    id: "memory_store",
    name: "memory_store",
    description:
      "Explicitly store a fact or piece of information into long-term memory. " +
      "The input is processed through LLM-powered fact extraction and dedup.",
    parameters: Type.Object({
      text: Type.String({
        description: "The text to extract facts from and store",
      }),
      userId: Type.Optional(
        Type.String({ description: "Override user ID for this operation" })
      ),
    }),
    async execute(args: { text: string; userId?: string }) {
      const engine = requireEngine(getEngine);
      const userId = args.userId ?? resolveUserId(cfg, api);
      const result = await engine.add(
        [{ role: "user", content: args.text }],
        userId,
        null,
        null,
        null,
        true
      );
      return formatAddResult(result);
    },
  });
}
