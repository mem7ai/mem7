import type {
  SearchResult,
  AddResult,
  MemoryItem,
} from "./format.js";

export interface Mem7Engine {
  add(
    messages: Array<{ role: string; content: string }>,
    userId?: string | null,
    agentId?: string | null,
    runId?: string | null,
    metadata?: string | null,
    infer?: boolean | null,
  ): Promise<AddResult>;

  search(
    query: string,
    userId?: string | null,
    agentId?: string | null,
    runId?: string | null,
    limit?: number | null,
    filters?: string | null,
    rerank?: boolean | null,
    threshold?: number | null,
  ): Promise<SearchResult>;

  get(memoryId: string): Promise<MemoryItem>;

  getAll(
    userId?: string | null,
    agentId?: string | null,
    runId?: string | null,
    filters?: string | null,
    limit?: number | null,
  ): Promise<MemoryItem[]>;

  update(memoryId: string, newText: string): Promise<void>;
  delete(memoryId: string): Promise<void>;
  deleteAll(
    userId?: string | null,
    agentId?: string | null,
    runId?: string | null,
  ): Promise<void>;
  reset(): Promise<void>;
}

export type EngineGetter = () => Mem7Engine | null;
