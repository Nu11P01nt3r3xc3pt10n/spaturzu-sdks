// Minimal ambient declarations for the OpenClaw plugin-sdk surface we use.
// Mirrors src/plugin-sdk/plugin-entry.ts + src/plugins/{hook-types,hook-
// decision-types,types}.ts in openclaw/openclaw. `openclaw` is a peer-
// Dependency at runtime — this shim only exists so this package typechecks
// without pulling the full openclaw tree into devDependencies. Bump it when
// the upstream signatures change.

declare module "openclaw/plugin-sdk/plugin-entry" {
  export type InputGateDecision =
    | { outcome: "pass" }
    | {
        outcome: "block";
        reason: string;
        message?: string;
        category?: string;
        metadata?: Record<string, unknown>;
      };

  export type PluginHookContextWindowSource = string;

  export type PluginHookLlmOutputEvent = {
    runId: string;
    sessionId: string;
    provider: string;
    model: string;
    contextTokenBudget?: number;
    contextWindowSource?: PluginHookContextWindowSource;
    contextWindowReferenceTokens?: number;
    resolvedRef?: string;
    harnessId?: string;
    prompt?: string;
    assistantTexts: string[];
    lastAssistant?: unknown;
    usage?: {
      input?: number;
      output?: number;
      cacheRead?: number;
      cacheWrite?: number;
      total?: number;
    };
  };

  export type PluginHookBeforeAgentRunEvent = {
    prompt: string;
    messages: unknown[];
    systemPrompt?: string;
    accountId?: string;
    channelId?: string;
    senderId?: string;
    senderIsOwner?: boolean;
  };

  export type PluginHookAgentContext = {
    runId?: string;
    agentId?: string;
    sessionKey?: string;
    sessionId?: string;
    channelId?: string;
    modelProviderId?: string;
    modelId?: string;
    trigger?: string;
    contextTokenBudget?: number;
    contextWindowSource?: PluginHookContextWindowSource;
    contextWindowReferenceTokens?: number;
  };

  export type PluginLogger = {
    debug?: (message: string) => void;
    info: (message: string) => void;
    warn: (message: string) => void;
    error: (message: string) => void;
  };

  export type OpenClawPluginApi = {
    id: string;
    pluginConfig?: Record<string, unknown>;
    logger: PluginLogger;
    on(
      hook: "llm_output",
      handler: (
        event: PluginHookLlmOutputEvent,
        ctx: PluginHookAgentContext,
      ) => void | Promise<void>,
      opts?: { priority?: number; timeoutMs?: number },
    ): void;
    on(
      hook: "before_agent_run",
      handler: (
        event: PluginHookBeforeAgentRunEvent,
        ctx: PluginHookAgentContext,
      ) => InputGateDecision | void | Promise<InputGateDecision | void>,
      opts?: { priority?: number; timeoutMs?: number },
    ): void;
    on(
      hook: "gateway_stop",
      handler: () => void | Promise<void>,
      opts?: { priority?: number; timeoutMs?: number },
    ): void;
  };

  export type DefinePluginEntryOptions = {
    id: string;
    name: string;
    description: string;
    configSchema?: unknown;
    register: (api: OpenClawPluginApi) => void;
  };

  export function definePluginEntry(opts: DefinePluginEntryOptions): unknown;
}
