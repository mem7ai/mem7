import type { MemoryEngine as Mem7Engine } from "@mem7ai/mem7";

export type { Mem7Engine };

export type EngineGetter = () => Mem7Engine | null;
