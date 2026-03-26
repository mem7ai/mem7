import { Type } from "@sinclair/typebox";
import type { OpenClawPluginApi, PluginConfig } from "./config.js";
import { resolveScopeIds, type MemoryScope } from "./config.js";
import {
  formatSearchResult,
  formatMemoryItem,
  formatMemoryList,
  formatAddResult,
  type MemoryItem,
  type SearchResult,
} from "./format.js";
import type { Mem7Engine, EngineGetter } from "./engine.js";

function requireEngine(getEngine: EngineGetter): Mem7Engine {
  const engine = getEngine();
  if (!engine) {
    throw new Error("mem7 engine is not initialized yet");
  }
  return engine;
}

const DEFAULT_SOURCE = "OPENCLAW";

const ScopeSchema = Type.Optional(
  Type.Union([Type.Literal("session"), Type.Literal("long-term"), Type.Literal("all")]),
);

type ToolScopeArgs = {
  userId?: string;
  agentId?: string;
  runId?: string;
  scope?: MemoryScope;
  longTerm?: boolean;
};

function normalizeScope(
  scope: MemoryScope | undefined,
  longTerm: boolean | undefined,
  defaultScope: MemoryScope,
): MemoryScope {
  if (scope) return scope;
  if (longTerm != null) return longTerm ? "long-term" : "session";
  return defaultScope;
}

function resolveToolScope(
  cfg: PluginConfig,
  api: OpenClawPluginApi,
  args: ToolScopeArgs,
  scope: Exclude<MemoryScope, "all">,
): { userId: string; agentId: string | null; runId: string | null } {
  return resolveScopeIds(cfg, api, {
    userId: args.userId,
    agentId: args.agentId,
    runId: args.runId,
    scope,
  });
}

function metadataJson(metadata?: Record<string, unknown>, agentId?: string | null): string {
  const payload: Record<string, unknown> = {
    source: DEFAULT_SOURCE,
    ...(metadata ?? {}),
  };
  if (agentId && payload.actor_id == null) {
    payload.actor_id = `agent:${agentId}`;
  }
  return JSON.stringify(payload);
}

function sourceFiltersJson(): string {
  return JSON.stringify({ source: DEFAULT_SOURCE });
}

function emptySearchResult(): SearchResult {
  return { memories: [], relations: [] };
}

function mergeSearchResults(results: SearchResult[], limit: number): SearchResult {
  const memoryMap = new Map<string, MemoryItem>();
  const relationKeys = new Set<string>();
  const relations: SearchResult["relations"] = [];

  for (const result of results) {
    for (const memory of result.memories) {
      if (!memoryMap.has(memory.id)) {
        memoryMap.set(memory.id, memory);
      }
    }
    for (const relation of result.relations) {
      const key = `${relation.source}|${relation.relationship}|${relation.destination}`;
      if (!relationKeys.has(key)) {
        relationKeys.add(key);
        relations.push(relation);
      }
    }
  }

  const memories = Array.from(memoryMap.values()).slice(0, limit);
  return {
    memories,
    relations,
  };
}

function mergeMemoryLists(lists: MemoryItem[][], limit: number): MemoryItem[] {
  const memoryMap = new Map<string, MemoryItem>();
  for (const list of lists) {
    for (const memory of list) {
      if (!memoryMap.has(memory.id)) {
        memoryMap.set(memory.id, memory);
      }
    }
  }
  return Array.from(memoryMap.values()).slice(0, limit);
}

async function searchMemories(
  engine: Mem7Engine,
  cfg: PluginConfig,
  api: OpenClawPluginApi,
  args: ToolScopeArgs & { query: string; limit?: number },
  defaultScope: MemoryScope = "all",
): Promise<SearchResult> {
  const scope = normalizeScope(args.scope, args.longTerm, defaultScope);
  const limit = args.limit ?? cfg.topK ?? 5;
  const filters = sourceFiltersJson();
  const threshold = cfg.searchThreshold ?? null;

  if (scope === "all") {
    const longTermScope = resolveToolScope(cfg, api, args, "long-term");
    const sessionScope = resolveToolScope(cfg, api, args, "session");

    const results: SearchResult[] = [
      await engine.search(
        args.query,
        longTermScope.userId,
        longTermScope.agentId,
        longTermScope.runId,
        limit,
        filters,
        true,
        threshold,
      ),
    ];

    if (sessionScope.runId) {
      results.push(
        await engine.search(
          args.query,
          sessionScope.userId,
          sessionScope.agentId,
          sessionScope.runId,
          limit,
          filters,
          true,
          threshold,
        ),
      );
    }

    return mergeSearchResults(results, limit);
  }

  const scopeIds = resolveToolScope(cfg, api, args, scope);
  if (scope === "session" && !scopeIds.runId) {
    return emptySearchResult();
  }

  return engine.search(
    args.query,
    scopeIds.userId,
    scopeIds.agentId,
    scopeIds.runId,
    limit,
    filters,
    true,
    threshold,
  );
}

async function listMemories(
  engine: Mem7Engine,
  cfg: PluginConfig,
  api: OpenClawPluginApi,
  args: ToolScopeArgs & { limit?: number },
  defaultScope: MemoryScope = "all",
): Promise<MemoryItem[]> {
  const scope = normalizeScope(args.scope, args.longTerm, defaultScope);
  const limit = args.limit ?? cfg.topK ?? 50;
  const filters = sourceFiltersJson();

  if (scope === "all") {
    const longTermScope = resolveToolScope(cfg, api, args, "long-term");
    const sessionScope = resolveToolScope(cfg, api, args, "session");

    const lists: MemoryItem[][] = [
      await engine.getAll(
        longTermScope.userId,
        longTermScope.agentId,
        longTermScope.runId,
        filters,
        limit,
      ),
    ];

    if (sessionScope.runId) {
      lists.push(
        await engine.getAll(
          sessionScope.userId,
          sessionScope.agentId,
          sessionScope.runId,
          filters,
          limit,
        ),
      );
    }

    return mergeMemoryLists(lists, limit);
  }

  const scopeIds = resolveToolScope(cfg, api, args, scope);
  if (scope === "session" && !scopeIds.runId) {
    return [];
  }

  return engine.getAll(scopeIds.userId, scopeIds.agentId, scopeIds.runId, filters, limit);
}

export function registerTools(
  api: OpenClawPluginApi,
  getEngine: EngineGetter,
  cfg: PluginConfig,
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
        }),
      ),
      userId: Type.Optional(Type.String({ description: "Override user ID" })),
      agentId: Type.Optional(Type.String({ description: "Optional agent scope" })),
      runId: Type.Optional(Type.String({ description: "Optional run/session scope" })),
      scope: ScopeSchema,
      longTerm: Type.Optional(
        Type.Boolean({ description: "Shortcut for scope=long-term when true" }),
      ),
    }),
    async execute(args: {
      query: string;
      limit?: number;
      userId?: string;
      agentId?: string;
      runId?: string;
      scope?: MemoryScope;
      longTerm?: boolean;
    }) {
      try {
        const engine = requireEngine(getEngine);
        const result = await searchMemories(engine, cfg, api, args, "all");
        return formatSearchResult(result);
      } catch (err) {
        return {
          content: [{ type: "text", text: `Memory search failed: ${String(err)}` }],
          details: { error: String(err) },
        };
      }
    },
  });

  api.registerTool({
    id: "memory_get",
    name: "memory_get",
    description:
      'Retrieve a specific memory by ID, or pass path="all" to list all stored memories.',
    parameters: Type.Object({
      path: Type.Optional(
        Type.String({
          description: 'Legacy memory path or "all" to list all memories',
        }),
      ),
      memoryId: Type.Optional(Type.String({ description: "Specific memory ID (preferred)" })),
      from: Type.Optional(Type.Integer({ description: "Pagination offset", minimum: 0 })),
      lines: Type.Optional(
        Type.Integer({
          description: "Number of results to return",
          minimum: 1,
          maximum: 100,
        }),
      ),
      userId: Type.Optional(Type.String({ description: "Override user ID" })),
      agentId: Type.Optional(Type.String({ description: "Optional agent scope" })),
      runId: Type.Optional(Type.String({ description: "Optional run/session scope" })),
      scope: ScopeSchema,
      longTerm: Type.Optional(
        Type.Boolean({ description: "Shortcut for scope=long-term when true" }),
      ),
    }),
    async execute(args: {
      path?: string;
      memoryId?: string;
      from?: number;
      lines?: number;
      userId?: string;
      agentId?: string;
      runId?: string;
      scope?: MemoryScope;
      longTerm?: boolean;
    }) {
      try {
        const engine = requireEngine(getEngine);
        const target = args.memoryId ?? args.path;

        if (!target) {
          return {
            content: [{ type: "text", text: "Provide memoryId or path." }],
            details: { error: "missing_param" },
          };
        }

        if (target === "all") {
          const items = await listMemories(
            engine,
            cfg,
            api,
            {
              userId: args.userId,
              agentId: args.agentId,
              runId: args.runId,
              scope: args.scope,
              longTerm: args.longTerm,
              limit: args.lines ?? 50,
            },
            "all",
          );
          return formatMemoryList(items);
        }

        const item = await engine.get(target);
        return formatMemoryItem(item);
      } catch (err) {
        return {
          content: [{ type: "text", text: `Memory get failed: ${String(err)}` }],
          details: { error: String(err) },
        };
      }
    },
  });

  api.registerTool({
    id: "memory_list",
    name: "memory_list",
    description: "List stored memories for the selected user/session scope.",
    parameters: Type.Object({
      limit: Type.Optional(
        Type.Integer({
          description: "Maximum number of memories to return",
          minimum: 1,
          maximum: 100,
        }),
      ),
      userId: Type.Optional(Type.String({ description: "Override user ID" })),
      agentId: Type.Optional(Type.String({ description: "Optional agent scope" })),
      runId: Type.Optional(Type.String({ description: "Optional run/session scope" })),
      scope: ScopeSchema,
      longTerm: Type.Optional(
        Type.Boolean({ description: "Shortcut for scope=long-term when true" }),
      ),
    }),
    async execute(args: {
      limit?: number;
      userId?: string;
      agentId?: string;
      runId?: string;
      scope?: MemoryScope;
      longTerm?: boolean;
    }) {
      try {
        const engine = requireEngine(getEngine);
        const items = await listMemories(engine, cfg, api, args, "all");
        return formatMemoryList(items);
      } catch (err) {
        return {
          content: [{ type: "text", text: `Memory list failed: ${String(err)}` }],
          details: { error: String(err) },
        };
      }
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
      userId: Type.Optional(Type.String({ description: "Override user ID for this operation" })),
      agentId: Type.Optional(Type.String({ description: "Optional agent scope" })),
      runId: Type.Optional(Type.String({ description: "Optional run/session scope" })),
      metadata: Type.Optional(
        Type.Record(Type.String(), Type.Unknown(), {
          description: "Optional metadata to attach to this memory",
        }),
      ),
      scope: ScopeSchema,
      longTerm: Type.Optional(
        Type.Boolean({ description: "Store as long-term when true; session when false" }),
      ),
    }),
    async execute(args: {
      text: string;
      userId?: string;
      agentId?: string;
      runId?: string;
      metadata?: Record<string, unknown>;
      scope?: MemoryScope;
      longTerm?: boolean;
    }) {
      try {
        const engine = requireEngine(getEngine);
        const scope = normalizeScope(args.scope, args.longTerm, "long-term");
        const scopeIds = resolveToolScope(
          cfg,
          api,
          { ...args, scope },
          scope === "all" ? "long-term" : scope,
        );
        const result = await engine.add(
          [{ role: "user", content: args.text }],
          scopeIds.userId,
          scopeIds.agentId,
          scopeIds.runId,
          metadataJson(args.metadata, scopeIds.agentId),
          true,
        );
        return formatAddResult(result);
      } catch (err) {
        return {
          content: [{ type: "text", text: `Memory store failed: ${String(err)}` }],
          details: { error: String(err) },
        };
      }
    },
  });

  api.registerTool({
    id: "memory_forget",
    name: "memory_forget",
    description: "Delete a specific memory by ID, or search for candidate memories to forget.",
    parameters: Type.Object({
      memoryId: Type.Optional(Type.String({ description: "Specific memory ID (UUID) to delete" })),
      query: Type.Optional(
        Type.String({ description: "Search query used to find memory candidates" }),
      ),
      userId: Type.Optional(Type.String({ description: "Override user ID" })),
      agentId: Type.Optional(Type.String({ description: "Optional agent scope" })),
      runId: Type.Optional(Type.String({ description: "Optional run/session scope" })),
      scope: ScopeSchema,
      longTerm: Type.Optional(
        Type.Boolean({ description: "Shortcut for scope=long-term when true" }),
      ),
    }),
    async execute(args: {
      memoryId?: string;
      query?: string;
      userId?: string;
      agentId?: string;
      runId?: string;
      scope?: MemoryScope;
      longTerm?: boolean;
    }) {
      try {
        const engine = requireEngine(getEngine);

        if (args.memoryId) {
          await engine.delete(args.memoryId);
          return {
            content: [{ type: "text", text: `Memory ${args.memoryId} forgotten.` }],
            details: { action: "deleted", id: args.memoryId },
          };
        }

        if (!args.query) {
          return {
            content: [{ type: "text", text: "Provide a query or memoryId." }],
            details: { error: "missing_param" },
          };
        }

        const result = await searchMemories(
          engine,
          cfg,
          api,
          {
            query: args.query,
            limit: 5,
            userId: args.userId,
            agentId: args.agentId,
            runId: args.runId,
            scope: args.scope,
            longTerm: args.longTerm,
          },
          "all",
        );

        if (result.memories.length === 0) {
          return {
            content: [{ type: "text", text: "No matching memories found." }],
            details: { found: 0 },
          };
        }

        if (result.memories.length === 1 || (result.memories[0].score ?? 0) > 0.9) {
          await engine.delete(result.memories[0].id);
          const forgotten = result.memories[0].text;
          return {
            content: [{ type: "text", text: `Forgotten: "${forgotten}"` }],
            details: { action: "deleted", id: result.memories[0].id },
          };
        }

        const candidates = result.memories.map((memory) => ({
          id: memory.id,
          memory: memory.text,
          score: memory.score,
        }));

        const list = candidates
          .map(
            (memory) =>
              `- [${memory.id}] ${memory.memory.slice(0, 80)}${memory.memory.length > 80 ? "..." : ""} (score: ${((memory.score ?? 0) * 100).toFixed(0)}%)`,
          )
          .join("\n");

        return {
          content: [
            {
              type: "text",
              text: `Found ${candidates.length} candidates. Specify memoryId to delete:\n${list}`,
            },
          ],
          details: { action: "candidates", candidates },
        };
      } catch (err) {
        return {
          content: [{ type: "text", text: `Memory forget failed: ${String(err)}` }],
          details: { error: String(err) },
        };
      }
    },
  });
}
