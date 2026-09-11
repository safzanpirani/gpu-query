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
