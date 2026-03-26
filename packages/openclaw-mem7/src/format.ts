import type {
  JsAddResult,
  JsGraphRelation,
  JsMemoryActionResult,
  JsMemoryItem,
  JsSearchResult,
} from "@mem7ai/mem7";

export type AddResult = JsAddResult;
export type ActionResult = JsMemoryActionResult;
export type GraphRelation = JsGraphRelation;
export type MemoryItem = JsMemoryItem;
export type SearchResult = JsSearchResult;

type SnakeCaseMemoryItem = {
  id: string;
  text: string;
  user_id?: string | null;
  agent_id?: string | null;
  run_id?: string | null;
  actor_id?: string | null;
  role?: string | null;
  metadata: string | Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
  score?: number | null;
  last_accessed_at?: string | null;
  access_count: number;
  memory_type?: string | null;
};

type SnakeCaseActionResult = {
  id: string;
  action: string;
  old_value?: string;
  new_value?: string;
};

export interface ToolContent {
  type: "text";
  text: string;
}

export interface ToolReturn {
  content: ToolContent[];
  details?: Record<string, unknown>;
}

function formatDate(iso: string): string {
  if (!iso) return "";
  return iso.slice(0, 10);
}

function memoryText(item: { text: string }): string {
  return item.text;
}

function toSnakeCaseMemoryItem(item: MemoryItem): SnakeCaseMemoryItem {
  return {
    id: item.id,
    text: item.text,
    user_id: item.userId ?? null,
    agent_id: item.agentId ?? null,
    run_id: item.runId ?? null,
    actor_id: item.actorId ?? null,
    role: item.role ?? null,
    metadata: item.metadata,
    created_at: item.createdAt,
    updated_at: item.updatedAt,
    score: item.score ?? null,
    last_accessed_at: item.lastAccessedAt ?? null,
    access_count: item.accessCount,
    memory_type: item.memoryType ?? null,
  };
}

function toSnakeCaseActionResult(item: ActionResult): SnakeCaseActionResult {
  return {
    id: item.id,
    action: item.action,
    old_value: item.oldValue,
    new_value: item.newValue,
  };
}

function actionMemoryText(item: ActionResult): string {
  return item.newValue ?? item.oldValue ?? item.id;
}

function formatMemoryLine(m: MemoryItem): string {
  const date = formatDate(m.createdAt);
  const score = m.score != null ? ` (score: ${m.score.toFixed(2)})` : "";
  return `- [${date}] ${memoryText(m)}${score}`;
}

function formatRelationLine(r: GraphRelation): string {
  const score = r.score != null ? ` (${r.score.toFixed(2)})` : "";
  return `- ${r.source} -[${r.relationship}]-> ${r.destination}${score}`;
}

export function formatSearchResult(result: SearchResult): ToolReturn {
  const memories = result.memories.map(toSnakeCaseMemoryItem);
  const lines: string[] = [];

  if (memories.length === 0 && result.relations.length === 0) {
    return {
      content: [{ type: "text", text: "No relevant memories found." }],
      details: {
        count: 0,
        memories: [],
        relations: [],
      },
    };
  }

  if (memories.length > 0) {
    lines.push(`Found ${memories.length} memories:`);
    lines.push("");
    memories.forEach((memory, index) => {
      const score =
        memory.score != null ? ` (score: ${((memory.score ?? 0) * 100).toFixed(0)}%)` : "";
      lines.push(`${index + 1}. ${memoryText(memory)}${score} (id: ${memory.id})`);
    });
  }

  if (result.relations.length > 0) {
    if (lines.length > 0) {
      lines.push("");
    }
    lines.push("Relations:");
    result.relations.forEach((relation) => {
      lines.push(formatRelationLine(relation));
    });
  }

  return {
    content: [{ type: "text", text: lines.join("\n") }],
    details: {
      count: memories.length,
      memories,
      relations: result.relations,
    },
  };
}

export function formatMemoryItem(item: MemoryItem): ToolReturn {
  const normalized = toSnakeCaseMemoryItem(item);
  const lines = [
    `**ID:** ${normalized.id}`,
    `**Text:** ${item.text}`,
    `**Created:** ${formatDate(item.createdAt)}`,
    `**Updated:** ${formatDate(item.updatedAt)}`,
  ];

  if (item.score != null) {
    lines.push(`**Score:** ${item.score.toFixed(2)}`);
  }
  if (item.lastAccessedAt) {
    lines.push(`**Last Accessed:** ${formatDate(item.lastAccessedAt)}`);
  }
  lines.push(`**Access Count:** ${item.accessCount}`);

  return {
    content: [{ type: "text", text: lines.join("\n") }],
    details: { ...normalized },
  };
}

export function formatMemoryList(items: MemoryItem[]): ToolReturn {
  const normalized = items.map(toSnakeCaseMemoryItem);

  if (normalized.length === 0) {
    return {
      content: [{ type: "text", text: "No memories stored." }],
      details: { count: 0, memories: [] },
    };
  }

  const lines = [`${normalized.length} memories:`, ""];
  normalized.forEach((memory, index) => {
    lines.push(`${index + 1}. ${memoryText(memory)} (id: ${memory.id})`);
  });

  return {
    content: [{ type: "text", text: lines.join("\n") }],
    details: { count: normalized.length, memories: normalized },
  };
}

export function formatAddResult(result: AddResult): ToolReturn {
  const normalized = result.results.map(toSnakeCaseActionResult);
  const actionable = result.results
    .filter((r) => r.action !== "None")
    .map((r) => `[${r.action}] ${actionMemoryText(r)}`);

  const added = normalized.filter((r) => r.action === "Add").length;
  const updated = normalized.filter((r) => r.action === "Update").length;
  const deleted = normalized.filter((r) => r.action === "Delete").length;

  const summary: string[] = [];
  if (added > 0) {
    summary.push(`${added} new memor${added === 1 ? "y" : "ies"} added`);
  }
  if (updated > 0) {
    summary.push(`${updated} memor${updated === 1 ? "y" : "ies"} updated`);
  }
  if (deleted > 0) {
    summary.push(`${deleted} memor${deleted === 1 ? "y" : "ies"} deleted`);
  }
  if (summary.length === 0) {
    summary.push("No new memories extracted");
  }

  const text =
    actionable.length > 0
      ? `Stored: ${summary.join(", ")}. ${actionable.join("; ")}`
      : `Stored: ${summary.join(", ")}.`;

  return {
    content: [{ type: "text", text }],
    details: {
      results: normalized,
      relations: result.relations,
    },
  };
}

export function formatRecallContext(result: SearchResult): string {
  const lines: string[] = [];

  if (result.memories.length > 0) {
    lines.push("## Relevant memories about this user:");
    for (const m of result.memories) {
      lines.push(formatMemoryLine(m));
    }
  }

  if (result.relations.length > 0) {
    lines.push("");
    lines.push("## Known relations:");
    for (const r of result.relations) {
      lines.push(formatRelationLine(r));
    }
  }

  if (lines.length === 0) return "";

  return `<mem7_context>\n${lines.join("\n")}\n</mem7_context>`;
}
