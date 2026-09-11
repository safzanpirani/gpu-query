/**
 * parse(text, schema) -> filter clauses.
 *
 * The compiler is ported from spike/compile_query.py. It is the deterministic
 * half: the model proposes roles, and this code decides whether they form a
 * legal clause. A sequence that cannot is dropped rather than guessed at.
 *
 * The schema never enters the model. It enters the featurizer as identity-free
 * match rows, and it enters here.
 */

import { featurize } from "./featurize";
import * as gpu from "./webgpu";
import { LABELS, PARAMETERS, TRANSFER_EXACT_AST, argmax, forward } from "./model";
import { buildIndex, enumOwners, resolve, tokenize } from "./schema";
import type { Match, Schema } from "./schema";

export { LABELS, PARAMETERS, TRANSFER_EXACT_AST };
export type { Field, Kind, Schema } from "./schema";

export interface Clause {
  field: string;
  cmp: "eq" | "lt" | "gt" | "contains" | "is";
  value: string | true;
  neg: boolean;
}

export interface ParseResult {
  tokens: string[];
  roles: string[];
  scores: number[][];
  boundary: number[];
  resolved: Array<Match | null>;
  clauses: Clause[];
  micros: number;
}

const COMPARISON_FOR_ROLE: Record<string, Clause["cmp"]> = {
  OP_EQ: "eq", OP_LT: "lt", OP_GT: "gt", OP_CONTAINS: "contains",
};
const CONJUNCTIONS = new Set(["AND", "OR"]);
const VALUE_ROLES = new Set(["VALUE", "VALUE_CONT"]);
const FIELD_ROLES = new Set(["FIELD", "FIELD_CONT"]);

function splitClauses(roles: string[]): Array<[number, number]> {
  const spans: Array<[number, number]> = [];
  let start = 0;
  for (let i = 0; i < roles.length; i++) {
    if (CONJUNCTIONS.has(roles[i])) {
      if (i > start) spans.push([start, i]);
      start = i + 1;
    }
  }
  if (start < roles.length) spans.push([start, roles.length]);
  return spans;
}

export function compile(tokens: string[], roles: string[], schema: Schema): Clause[] {
  const index = buildIndex(schema);
  const owners = enumOwners(schema);
  const byName = new Map(schema.map((f) => [f.name, f]));
  const clauses: Clause[] = [];

  for (const [start, end] of splitClauses(roles)) {
    let neg = false;
    let fieldName: string | null = null;
    let fieldKind: string | null = null;
    let comparison: Clause["cmp"] | null = null;
    const valueParts: string[] = [];

    for (let i = start; i < end; i++) {
      const role = roles[i];
      const token = tokens[i];
      if (role === "NEG") neg = true;
      else if (FIELD_ROLES.has(role) && fieldName === null) {
        const found = resolve(token, index);
        if (found) {
          fieldName = found.field;
          fieldKind = found.kind;
        }
      } else if (role in COMPARISON_FOR_ROLE && comparison === null) {
        comparison = COMPARISON_FOR_ROLE[role];
      } else if (VALUE_ROLES.has(role)) valueParts.push(token);
    }

    if (fieldName === null) {
      // A bare value names its own field when exactly one field owns it.
      const candidates = new Set<string>();
      for (const part of valueParts) {
        for (const owner of owners.get(part.toLowerCase()) ?? []) candidates.add(owner);
      }
      if (candidates.size !== 1) continue;
      fieldName = [...candidates][0];
      fieldKind = byName.get(fieldName)?.kind ?? null;
    }

    if (fieldKind === "bool" && valueParts.length === 0) {
      clauses.push({ field: fieldName, cmp: "is", value: true, neg });
      continue;
    }
    if (valueParts.length === 0) continue;

    clauses.push({
      field: fieldName,
      cmp: comparison ?? "eq",
      value: valueParts.join(" "),
      neg,
    });
  }

  return clauses;
}

export function parse(text: string, schema: Schema): ParseResult {
  const started = performance.now();
  const tokens = tokenize(text);
  if (tokens.length === 0) {
    return {
      tokens: [], roles: [], scores: [], boundary: [], resolved: [],
      clauses: [], micros: 0,
    };
  }

  const { rows, neighbors, resolved } = featurize(tokens, schema);
  const out = forward(rows, neighbors);
  const roles = out.logits.map((row) => LABELS[argmax(row)]);
  const clauses = compile(tokens, roles, schema);

  return {
    tokens,
    roles,
    scores: out.logits,
    boundary: out.boundary,
    resolved,
    clauses,
    micros: Math.round((performance.now() - started) * 1000),
  };
}


export type Backend = "auto" | "cpu" | "webgpu";

/** WebGPU only repays its dispatch once there is a batch. gpu-time draws the
 *  same line at 32 inputs. Below it, the CPU path wins outright. */
export const GPU_THRESHOLD = 32;

export interface BatchResult {
  backend: "cpu" | "webgpu";
  millis: number;
  results: Array<{ tokens: string[]; roles: string[]; clauses: Clause[] }>;
}

export function gpuSupported(): boolean {
  return gpu.supported();
}

export async function warmGpu(): Promise<boolean> {
  return gpu.init();
}

/** Parse many queries at once. This is the case the architecture is for. */
export async function parseBatch(
  texts: string[],
  schema: Schema,
  backend: Backend = "auto",
): Promise<BatchResult> {
  const prepared = texts.map((text) => {
    const tokens = tokenize(text);
    const { rows, neighbors } = featurize(tokens, schema);
    return { tokens, rows, neighbors };
  });

  const wantGpu =
    backend === "webgpu" ||
    (backend === "auto" && texts.length >= GPU_THRESHOLD && gpu.supported());

  const started = performance.now();

  if (wantGpu && (await gpu.init())) {
    const raw = await gpu.runBatch({
      rows: prepared.map((p) => p.rows),
      neighbors: prepared.map((p) => p.neighbors),
    });
    const results = prepared.map((p, i) => {
      const roles = raw[i].map((row) => LABELS[argmax(row.slice(0, LABELS.length))]);
      return { tokens: p.tokens, roles, clauses: compile(p.tokens, roles, schema) };
    });
    return { backend: "webgpu", millis: performance.now() - started, results };
  }

  if (backend === "webgpu") throw new Error("WebGPU requested but unavailable");

  const results = prepared.map((p) => {
    const out = forward(p.rows, p.neighbors);
    const roles = out.logits.map((row) => LABELS[argmax(row)]);
    return { tokens: p.tokens, roles, clauses: compile(p.tokens, roles, schema) };
  });
  return { backend: "cpu", millis: performance.now() - started, results };
}

/** Run the same inputs through both backends and report the worst drift.
 *  A kernel nobody checked is worse than no kernel. */
export async function checkBackends(
  texts: string[],
  schema: Schema,
): Promise<{ ok: boolean; maxLogitDelta: number; roleMismatches: number }> {
  const prepared = texts.map((text) => {
    const tokens = tokenize(text);
    return { tokens, ...featurize(tokens, schema) };
  });
  if (!(await gpu.init())) return { ok: false, maxLogitDelta: NaN, roleMismatches: -1 };

  const raw = await gpu.runBatch({
    rows: prepared.map((p) => p.rows),
    neighbors: prepared.map((p) => p.neighbors),
  });

  let maxLogitDelta = 0;
  let roleMismatches = 0;
  prepared.forEach((p, i) => {
    const cpu = forward(p.rows, p.neighbors);
    for (let t = 0; t < p.tokens.length; t++) {
      for (let r = 0; r < LABELS.length; r++) {
        maxLogitDelta = Math.max(maxLogitDelta, Math.abs(cpu.logits[t][r] - raw[i][t][r]));
      }
      maxLogitDelta = Math.max(
        maxLogitDelta, Math.abs(cpu.boundary[t] - raw[i][t][LABELS.length]),
      );
      const a = LABELS[argmax(cpu.logits[t])];
      const b = LABELS[argmax(raw[i][t].slice(0, LABELS.length))];
      if (a !== b) roleMismatches++;
    }
  });

  return { ok: roleMismatches === 0 && maxLogitDelta < 1e-3, maxLogitDelta, roleMismatches };
}
