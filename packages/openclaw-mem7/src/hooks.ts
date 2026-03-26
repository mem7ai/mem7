import type { OpenClawPluginApi, PluginConfig } from "./config.js";
import { resolveScopeIds } from "./config.js";
import { formatRecallContext } from "./format.js";
import type { EngineGetter } from "./engine.js";
import type { SearchResult } from "./format.js";

interface ChatMessage {
  role: string;
  content?: string;
}

interface PromptBuildEvent {
  messages?: ChatMessage[];
  prompt?: string;
}

interface AgentEndEvent {
  messages?: ChatMessage[];
  success?: boolean;
}

const DEFAULT_SOURCE = "OPENCLAW";

function extractLastUserMessage(messages?: ChatMessage[]): string | null {
  if (!messages || messages.length === 0) return null;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user" && messages[i].content) {
      return messages[i].content!;
    }
  }
  return null;
}

function extractRecallQuery(event: PromptBuildEvent): string | null {
  if (typeof event.prompt === "string" && event.prompt.trim().length > 0) {
    return event.prompt;
  }
  return extractLastUserMessage(event.messages);
}

function extractRecentMessages(messages?: ChatMessage[]): Array<{ role: string; content: string }> {
  if (!messages || messages.length === 0) return [];

  return messages
    .filter(
      (msg): msg is { role: string; content: string } =>
        (msg.role === "assistant" || msg.role === "user") &&
        typeof msg.content === "string" &&
        msg.content.length > 0,
    )
    .slice(-10)
    .map((msg) => ({ role: msg.role, content: msg.content }));
}

function mergeSearchResults(results: SearchResult[], limit: number): SearchResult {
  const memoryMap = new Map<string, SearchResult["memories"][number]>();
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

export function registerHooks(
  api: OpenClawPluginApi,
  getEngine: EngineGetter,
  cfg: PluginConfig,
): void {
  const autoRecall = cfg.autoRecall !== false;
  const autoRecallLimit = cfg.autoRecallLimit ?? cfg.topK ?? 5;
  const autoCapture = cfg.autoCapture !== false;
  const searchThreshold = cfg.searchThreshold ?? null;
  const filters = JSON.stringify({ source: DEFAULT_SOURCE });

  if (autoRecall) {
    const recallHandler = async (
      event: unknown,
    ): Promise<{ prependSystemContext?: string; prependContext?: string } | void> => {
      const engine = getEngine();
      if (!engine) return;

      const ev = event as PromptBuildEvent;
      const query = extractRecallQuery(ev);
      if (!query) return;

      try {
        const sessionScope = resolveScopeIds(cfg, api, { scope: "session" });
        const longTermScope = resolveScopeIds(cfg, api, { scope: "long-term" });

        const recallResults: SearchResult[] = [
          await engine.search(
            query,
            longTermScope.userId,
            longTermScope.agentId,
            longTermScope.runId,
            autoRecallLimit,
            filters,
            true,
            searchThreshold,
          ),
        ];

        if (sessionScope.runId) {
          recallResults.push(
            await engine.search(
              query,
              sessionScope.userId,
              sessionScope.agentId,
              sessionScope.runId,
              autoRecallLimit,
              filters,
              true,
              searchThreshold,
            ),
          );
        }

        const result = mergeSearchResults(recallResults, autoRecallLimit);
        const context = formatRecallContext(result);
        if (!context) return;

        return {
          prependSystemContext: context,
          prependContext: context,
        };
      } catch (err) {
        api.logger.warn("mem7 auto-recall failed: %s", String(err));
      }
    };

    api.on("before_prompt_build", recallHandler);
    api.on("before_agent_start", recallHandler);
  }

  if (autoCapture) {
    api.on("agent_end", async (event: unknown): Promise<void> => {
      const engine = getEngine();
      if (!engine) return;

      const ev = event as AgentEndEvent;
      if (ev.success === false) return;

      const turnMsgs = extractRecentMessages(ev.messages);
      if (turnMsgs.length === 0) return;

      try {
        const sessionScope = resolveScopeIds(cfg, api, { scope: "session" });
        if (!sessionScope.runId) return;
        const metadata = {
          source: DEFAULT_SOURCE,
          ...(sessionScope.agentId ? { actor_id: `agent:${sessionScope.agentId}` } : {}),
        };
        const result = await engine.add(
          turnMsgs,
          sessionScope.userId,
          sessionScope.agentId,
          sessionScope.runId,
          JSON.stringify(metadata),
          true,
        );

        const actions = result.results?.filter((r) => r.action !== "None");
        if (actions && actions.length > 0) {
          api.logger.info(
            "mem7 auto-capture: %d actions (%s)",
            actions.length,
            actions.map((a) => a.action).join(", "),
          );
        } else {
          api.logger.debug("mem7 auto-capture: no new facts");
        }
      } catch (err) {
        api.logger.warn("mem7 auto-capture failed: %s", String(err));
      }
    });
  }
}
