import { resolve, join } from "node:path";
import { homedir } from "node:os";

export interface PluginConfig {
  llm?: {
    base_url?: string;
    api_key?: string;
    model?: string;
    temperature?: number;
    max_tokens?: number;
    enable_vision?: boolean;
  };
  embedding?: {
    provider?: string;
    base_url?: string;
    api_key?: string;
    model?: string;
    dims?: number;
  };
  vector?: {
    provider?: string;
    collection_name?: string;
    upstash_url?: string;
    upstash_token?: string;
  };
  graph?: {
    provider?: string;
    neo4j_url?: string;
    neo4j_username?: string;
    neo4j_password?: string;
    neo4j_database?: string;
    kuzu_db_path?: string;
  } | null;
  decay?: {
    enabled?: boolean;
    base_half_life_secs?: number;
    decay_shape?: number;
    min_retention?: number;
    rehearsal_factor?: number;
  };
  autoRecall?: boolean;
  autoRecallLimit?: number;
  autoCapture?: boolean;
  userId?: string;
  dbPath?: string;
}

export interface ToolDefinition {
  id: string;
  name: string;
  description: string;
  parameters: unknown;
  execute(args: Record<string, unknown>): Promise<unknown>;
}

export interface ServiceDefinition {
  id: string;
  start(): Promise<void>;
  stop(): Promise<void>;
}

export interface OpenClawPluginApi {
  pluginConfig?: Record<string, unknown>;
  config?: {
    models?: {
      providers?: Record<string, { apiKey?: string; baseUrl?: string }>;
    };
  };
  logger: {
    info(msg: string, ...args: unknown[]): void;
    warn(msg: string, ...args: unknown[]): void;
    error(msg: string, ...args: unknown[]): void;
    debug(msg: string, ...args: unknown[]): void;
  };
  registerService(service: ServiceDefinition): void;
  registerTool(tool: ToolDefinition): void;
  on(event: string, handler: (event: unknown) => Promise<unknown> | void): void;
  runtime?: {
    sessionKey?: string;
  };
}

function expandPath(p: string): string {
  if (p.startsWith("~/")) {
    return join(homedir(), p.slice(2));
  }
  return resolve(p);
}

function resolveApiKey(
  api: OpenClawPluginApi,
  provider: string
): string | undefined {
  return api.config?.models?.providers?.[provider]?.apiKey;
}

function resolveBaseUrl(
  api: OpenClawPluginApi,
  provider: string
): string | undefined {
  return api.config?.models?.providers?.[provider]?.baseUrl;
}

export function buildMem7Config(
  cfg: PluginConfig,
  api: OpenClawPluginApi
): Record<string, unknown> {
  const dbBase = expandPath(cfg.dbPath ?? "~/.openclaw/mem7");

  const llmApiKey =
    cfg.llm?.api_key ?? resolveApiKey(api, "openai") ?? "";
  const llmBaseUrl =
    cfg.llm?.base_url ??
    resolveBaseUrl(api, "openai") ??
    "https://api.openai.com/v1";

  const embProvider = cfg.embedding?.provider ?? "openai";
  const embApiKey =
    cfg.embedding?.api_key ?? resolveApiKey(api, embProvider) ?? "";
  const embBaseUrl =
    cfg.embedding?.base_url ??
    resolveBaseUrl(api, embProvider) ??
    "https://api.openai.com/v1";

  const mem7Cfg: Record<string, unknown> = {
    llm: {
      provider: "openai",
      base_url: llmBaseUrl,
      api_key: llmApiKey,
      model: cfg.llm?.model ?? "gpt-4.1-nano",
      temperature: cfg.llm?.temperature ?? 0.0,
      max_tokens: cfg.llm?.max_tokens ?? 1000,
      enable_vision: cfg.llm?.enable_vision ?? false,
    },
    embedding: {
      provider: embProvider,
      base_url: embBaseUrl,
      api_key: embApiKey,
      model: cfg.embedding?.model ?? "text-embedding-3-small",
      dims: cfg.embedding?.dims ?? 1536,
    },
    vector: {
      provider: cfg.vector?.provider ?? "flat",
      collection_name: cfg.vector?.collection_name ?? "mem7_memories",
      dims: cfg.embedding?.dims ?? 1536,
      upstash_url: cfg.vector?.upstash_url ?? null,
      upstash_token: cfg.vector?.upstash_token ?? null,
    },
    history: {
      db_path: join(dbBase, "history.db"),
    },
    decay: cfg.decay
      ? { enabled: cfg.decay.enabled ?? true, ...cfg.decay }
      : { enabled: true },
  };

  if (cfg.graph) {
    const g = cfg.graph;
    mem7Cfg.graph = {
      provider: g.provider ?? "flat",
      kuzu_db_path: g.kuzu_db_path ?? join(dbBase, "graph.kuzu"),
      neo4j_url: g.neo4j_url ?? null,
      neo4j_username: g.neo4j_username ?? null,
      neo4j_password: g.neo4j_password ?? null,
      neo4j_database: g.neo4j_database ?? null,
    };
  }

  return mem7Cfg;
}

export function resolveUserId(
  cfg: PluginConfig,
  api: OpenClawPluginApi
): string {
  if (cfg.userId) return cfg.userId;
  if (api.runtime?.sessionKey) return api.runtime.sessionKey;
  return "default";
}
