export interface MemoryItem {
  id: string;
  text: string;
  userId?: string | null;
  agentId?: string | null;
  runId?: string | null;
  metadata: Record<string, unknown> | null;
  createdAt: string;
  updatedAt: string;
  score?: number | null;
  lastAccessedAt?: string | null;
  accessCount: number;
}

export interface GraphRelation {
  source: string;
  relationship: string;
  destination: string;
  score?: number;
}

export interface ActionResult {
  id: string;
  action: string;
  oldValue?: string;
  newValue?: string;
}

export interface SearchResult {
  memories: MemoryItem[];
  relations: GraphRelation[];
}

export interface AddResult {
  results: ActionResult[];
  relations: GraphRelation[];
}

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

function formatMemoryLine(m: MemoryItem): string {
  const date = formatDate(m.createdAt);
  const score = m.score != null ? ` (score: ${m.score.toFixed(2)})` : "";
  return `- [${date}] ${m.text}${score}`;
}

function formatRelationLine(r: GraphRelation): string {
  const score = r.score != null ? ` (${r.score.toFixed(2)})` : "";
  return `- ${r.source} -[${r.relationship}]-> ${r.destination}${score}`;
}

export function formatSearchResult(result: SearchResult): ToolReturn {
  const lines: string[] = [];

  if (result.memories.length > 0) {
    lines.push("## Memories");
    for (const m of result.memories) {
      lines.push(formatMemoryLine(m));
    }
  }

  if (result.relations.length > 0) {
    if (lines.length > 0) lines.push("");
    lines.push("## Relations");
    for (const r of result.relations) {
      lines.push(formatRelationLine(r));
    }
  }

  const text =
    lines.length > 0 ? lines.join("\n") : "No relevant memories found.";

  return {
    content: [{ type: "text", text }],
    details: {
      count: result.memories.length,
      memories: result.memories,
      relations: result.relations,
    },
  };
}

export function formatMemoryItem(item: MemoryItem): ToolReturn {
  const lines = [
    `**ID:** ${item.id}`,
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
    details: { ...item },
  };
}

export function formatMemoryList(items: MemoryItem[]): ToolReturn {
  if (items.length === 0) {
    return {
      content: [{ type: "text", text: "No memories stored." }],
      details: { count: 0, memories: [] },
    };
  }

  const lines = items.map(
    (m) => `- **${m.id}**: ${m.text} [${formatDate(m.createdAt)}]`
  );

  return {
    content: [{ type: "text", text: lines.join("\n") }],
    details: { count: items.length, memories: items },
  };
}

export function formatAddResult(result: AddResult): ToolReturn {
  const actions = result.results
    .filter((r) => r.action !== "None")
    .map((r) => `- ${r.action}: ${r.newValue ?? r.oldValue ?? r.id}`);

  const text =
    actions.length > 0
      ? actions.join("\n")
      : "No new facts extracted from the input.";

  return {
    content: [{ type: "text", text }],
    details: {
      results: result.results,
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
