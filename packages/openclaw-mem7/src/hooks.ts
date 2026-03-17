import type { OpenClawPluginApi, PluginConfig } from "./config.js";
import { resolveUserId } from "./config.js";
import { formatRecallContext } from "./format.js";
import type { MemoryEngine } from "@mem7ai/mem7";

type EngineGetter = () => MemoryEngine | null;

interface ChatMessage {
  role: string;
  content?: string;
}

interface PromptBuildEvent {
  messages?: ChatMessage[];
}

interface AgentEndEvent {
  messages?: ChatMessage[];
  success?: boolean;
}

function extractLastUserMessage(messages?: ChatMessage[]): string | null {
  if (!messages || messages.length === 0) return null;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user" && messages[i].content) {
      return messages[i].content!;
    }
  }
  return null;
}

function extractTurnMessages(
  messages?: ChatMessage[]
): Array<{ role: string; content: string }> {
  if (!messages || messages.length === 0) return [];

  const turn: Array<{ role: string; content: string }> = [];
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (!msg.content) continue;

    if (msg.role === "assistant" || msg.role === "user") {
      turn.unshift({ role: msg.role, content: msg.content });
    }

    if (msg.role === "user" && turn.length >= 2) break;
  }
  return turn;
}

export function registerHooks(
  api: OpenClawPluginApi,
  getEngine: EngineGetter,
  cfg: PluginConfig
): void {
  const autoRecall = cfg.autoRecall !== false;
  const autoRecallLimit = cfg.autoRecallLimit ?? 5;
  const autoCapture = cfg.autoCapture !== false;

  if (autoRecall) {
    api.on(
      "before_prompt_build",
      async (event: unknown): Promise<{ prependSystemContext?: string } | void> => {
        const engine = getEngine();
        if (!engine) return;

        const ev = event as PromptBuildEvent;
        const query = extractLastUserMessage(ev.messages);
        if (!query) return;

        try {
          const userId = resolveUserId(cfg, api);
          const result = await engine.search(
            query,
            userId,
            null,
            null,
            autoRecallLimit,
            null,
            true,
            null
          );

          const context = formatRecallContext(result as any);
          if (context) {
            return { prependSystemContext: context };
          }
        } catch (err) {
          api.logger.warn("mem7 auto-recall failed: %s", String(err));
        }
      }
    );
  }

  if (autoCapture) {
    api.on("agent_end", async (event: unknown): Promise<void> => {
      const engine = getEngine();
      if (!engine) return;

      const ev = event as AgentEndEvent;
      if (ev.success === false) return;

      const turnMsgs = extractTurnMessages(ev.messages);
      if (turnMsgs.length === 0) return;

      try {
        const userId = resolveUserId(cfg, api);
        const result = await engine.add(turnMsgs, userId, null, null, null, true);

        const actions = (result as any).results?.filter(
          (r: any) => r.action !== "None"
        );
        if (actions && actions.length > 0) {
          api.logger.info(
            "mem7 auto-capture: %d actions (%s)",
            actions.length,
            actions.map((a: any) => a.action).join(", ")
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
