import type { OpenClawPluginApi, PluginConfig } from "./config.js";
import { buildMem7Config } from "./config.js";
import { registerTools } from "./tools.js";
import { registerHooks } from "./hooks.js";
import type { Mem7Engine } from "./engine.js";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";

export interface PluginEntry {
  id: string;
  name: string;
  description: string;
  kind: string;
  register(api: OpenClawPluginApi): void;
}

export default {
  id: "openclaw-mem7",
  name: "Memory (mem7)",
  description:
    "LLM-powered long-term memory with fact extraction, graph relations, dedup, and Ebbinghaus forgetting curve.",
  kind: "memory",

  register(api: OpenClawPluginApi): void {
    const pluginCfg = (api.pluginConfig ?? {}) as PluginConfig;
    const mem7ConfigJson = buildMem7Config(pluginCfg, api);

    let engine: Mem7Engine | null = null;

    const historyDb = (mem7ConfigJson.history as { db_path?: string })?.db_path;
    if (historyDb) {
      try {
        mkdirSync(dirname(historyDb), { recursive: true });
      } catch {
        // directory may already exist
      }
    }

    api.registerService({
      id: "openclaw-mem7",
      async start() {
        // Dynamic import -- types are provided by our local Mem7Engine interface,
        // not the npm package (which may lag behind the current Rust API).
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const mod: any = await import("@mem7ai/mem7");
        engine = await mod.MemoryEngine.create(
          JSON.stringify(mem7ConfigJson)
        );
        api.logger.info("mem7 engine initialized");
      },
      async stop() {
        engine = null;
        api.logger.info("mem7 engine stopped");
      },
    });

    registerTools(api, () => engine, pluginCfg);
    registerHooks(api, () => engine, pluginCfg);
  },
} satisfies PluginEntry;
