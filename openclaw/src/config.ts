import { z } from "zod";

/** Plugin config validated against `api.pluginConfig` at register time. */
export const ConfigSchema = z.object({
  apiKey: z.string().min(1, "apiKey is required"),
  baseUrl: z.string().url().default("https://spaturzu-api.superchiu.org"),
  agentPrefix: z.string().default("openclaw"),
  enforceBudgets: z.boolean().default(true),
  tags: z
    .record(z.string(), z.union([z.string(), z.number(), z.boolean()]))
    .default({}),
});

export type SpaturzuOpenClawConfig = z.infer<typeof ConfigSchema>;

/** Parse `api.pluginConfig` (typed as `unknown`) into a fully-defaulted
 *  config. Throws a zod error with a descriptive message on missing
 *  apiKey or malformed input — surfaced to the operator at startup. */
export function parseConfig(raw: unknown): SpaturzuOpenClawConfig {
  return ConfigSchema.parse(raw ?? {});
}
