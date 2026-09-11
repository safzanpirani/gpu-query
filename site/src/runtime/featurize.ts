/**
 * Sparse feature rows and neighbour indices.
 *
 * Ported from spike/featurize.py and `neighbors_for` in spike/train.py. Row
 * offsets must match the Python exactly, including the two word-identity blocks
 * that this checkpoint never emits: the blocks still occupy their offsets, they
 * are simply skipped. The shipped model was trained with --no-word-identity, so
 * nothing here carries a field name into the weights.
 */

import { buildIndex, enumOwners, resolve, surfaceForms } from "./schema";
import type { Index, Kind, Match, Schema } from "./schema";

export const SLOTS = 16;

const OPERATOR_WORDS = [
  "and", "or", "not", "no", "without", "excluding", "is", "before", "after",
  "since", "over", "under", "above", "below", "than", "least", "most", "more",
  "less", "fewer", "prior", "later", "earlier", "containing", "mentioning",
  "matching", "with", ":", "=",
];
const OPERATOR_INDEX = new Map(OPERATOR_WORDS.map((w, i) => [w, i]));

const BLOCKS: Array<[string, number]> = [
  ["kind", 3],
  ["length", 8],
  ["first_char", 40],
  ["last_char", 40],
  ["word_hash", 256],
  ["devowel_hash", 128],
  ["flags", 4],
  ["field_match", 6],
  ["match_quality", 5],
  ["enum_any", 1],
  ["enum_preceding", 1],
  ["distance", 6],
  ["operator", 30],
  ["position", 4],
];

export const OFFSET: Record<string, number> = {};
let cursor = 0;
for (const [name, size] of BLOCKS) {
  OFFSET[name] = cursor;
  cursor += size;
}
export const FEATURE_ROWS = cursor;
export const PADDING_ROW = FEATURE_ROWS;

const KIND_OF_FIELD: Record<Kind, number> = {
  bool: 1, enum: 2, date: 3, number: 4, text: 5,
};
const LENGTH_BUCKETS = [1, 2, 3, 4, 6, 8, 12, 1e9];
const DISTANCE_ROWS = 6;

function characterClass(character: string): number {
  const code = character.toLowerCase().charCodeAt(0);
  if (code >= 97 && code <= 122) return code - 97;
  if (code >= 48 && code <= 57) return code - 48 + 26;
  return 36 + (code % 4);
}

function lengthBucket(text: string): number {
  for (let i = 0; i < LENGTH_BUCKETS.length; i++) {
    if (text.length <= LENGTH_BUCKETS[i]) return i;
  }
  return LENGTH_BUCKETS.length - 1;
}

function isDigits(text: string): boolean {
  const stripped = text.replace(/\./g, "");
  return stripped.length > 0 && /^[0-9]+$/.test(stripped);
}

export interface Featurized {
  rows: number[][];
  neighbors: Array<[number, number]>;
  resolved: Array<Match | null>;
}

export function featurize(tokens: string[], schema: Schema): Featurized {
  const index: Index = buildIndex(schema);
  const owners = enumOwners(schema);
  const lowered = tokens.map((t) => t.toLowerCase());
  const resolved = lowered.map((t) => resolve(t, index));

  // Nearest resolved field before each token, by canonical name.
  const precedingField: Array<string | null> = [];
  const distanceToField: number[] = [];
  let current: string | null = null;
  let distance = 99;
  for (let i = 0; i < lowered.length; i++) {
    precedingField.push(current);
    distanceToField.push(distance);
    if (resolved[i]) {
      current = resolved[i]!.field;
      distance = 0;
    } else if (distance < 99) {
      distance += 1;
    }
  }

  const rows: number[][] = [];
  for (let position = 0; position < tokens.length; position++) {
    const token = tokens[position];
    const folded = lowered[position];
    const active: number[] = [];
    const add = (block: string, value: number) => {
      // word_hash and devowel_hash are the ablated blocks; never emitted.
      if (block === "word_hash" || block === "devowel_hash") return;
      active.push(OFFSET[block] + value);
    };

    let kind: number;
    if (isDigits(folded)) kind = 1;
    else if (/[A-Za-z_]/.test(folded[0])) kind = 0;
    else kind = 2;
    add("kind", kind);
    add("length", lengthBucket(token));
    add("first_char", characterClass(token[0]));
    add("last_char", characterClass(token[token.length - 1]));

    if (token[0] === token[0].toUpperCase() && /[A-Za-z]/.test(token[0])) add("flags", 0);
    if (/[0-9]/.test(token)) add("flags", 1);
    if (token.includes("-") || token.includes("_")) add("flags", 2);

    const found = resolved[position];
    add("field_match", found ? KIND_OF_FIELD[found.kind] : 0);
    add("match_quality", found ? found.quality + 1 : 0);

    const owning = owners.get(folded);
    if (owning) {
      add("enum_any", 0);
      const previous = precedingField[position];
      if (previous !== null && owning.has(previous)) add("enum_preceding", 0);
    }

    const gap = distanceToField[position];
    add("distance", gap === 99 ? DISTANCE_ROWS - 1 : Math.min(gap, DISTANCE_ROWS - 2));
    add("operator", OPERATOR_INDEX.get(folded) ?? OPERATOR_WORDS.length);

    const span = Math.max(1, tokens.length - 1);
    add("position", Math.min(3, Math.floor((4 * position) / (span + 1))));

    const padded = active.slice(0, SLOTS);
    while (padded.length < SLOTS) padded.push(PADDING_ROW);
    rows.push(padded);
  }

  return { rows, neighbors: neighborsFor(tokens, schema), resolved };
}

/**
 * Nearest field-matching token on each side.
 *
 * Note this uses EXACT surface matching, not the tolerant resolver, because the
 * Python it mirrors builds its index from surface forms directly.
 */
export function neighborsFor(tokens: string[], schema: Schema): Array<[number, number]> {
  const surfaces = new Set<string>();
  for (const field of schema) {
    for (const surface of surfaceForms(field)) surfaces.add(surface.toLowerCase());
  }
  const isField = tokens.map((t) => surfaces.has(t.toLowerCase()));

  const previous: number[] = new Array(tokens.length).fill(-1);
  let last = -1;
  for (let i = 0; i < tokens.length; i++) {
    previous[i] = last;
    if (isField[i]) last = i;
  }

  const following: number[] = new Array(tokens.length).fill(-1);
  let upcoming = -1;
  for (let i = tokens.length - 1; i >= 0; i--) {
    following[i] = upcoming;
    if (isField[i]) upcoming = i;
  }

  return tokens.map((_, i) => [previous[i], following[i]] as [number, number]);
}
