/**
 * The tagger forward pass, in plain TypeScript.
 *
 * Ported from spike/model.py. At 29,597 parameters over a handful of tokens the
 * whole pass is well under a millisecond on a CPU, so this runs as plain arrays
 * rather than through WebGPU. The architecture is the reason a GPU would help at
 * scale: step 5 is a prefix scan, which parallelises. At one short query it does
 * not need to.
 *
 * Verified against PyTorch by `npm run parity`.
 */

import weightsBlob from "./weights.json";

const HIDDEN = 32;

interface Packed {
  shape: number[];
  scale: number;
  data: string;
}

interface Blob {
  featureRows: number;
  hidden: number;
  slots: number;
  labels: string[];
  parameters: number;
  bits: number;
  transferExactAst: number;
  tensors: Record<string, Packed>;
}

const BLOB = weightsBlob as unknown as Blob;

function decode(packed: Packed): Float32Array {
  const binary = atob(packed.data);
  const out = new Float32Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    // int8 two's complement, then the tensor scale.
    const byte = binary.charCodeAt(i);
    out[i] = ((byte << 24) >> 24) * packed.scale;
  }
  return out;
}

const W: Record<string, Float32Array> = {};
for (const [name, packed] of Object.entries(BLOB.tensors)) W[name] = decode(packed);

export const LABELS = BLOB.labels;
export const PARAMETERS = BLOB.parameters;
export const FEATURE_ROWS = BLOB.featureRows;
export const TRANSFER_EXACT_AST = BLOB.transferExactAst;

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/** y = x @ Wᵀ + b, with W shaped (out, in). */
function linear(x: Float32Array, name: string, outSize: number, inSize: number): Float32Array {
  const weight = W[`${name}_weight`];
  const bias = W[`${name}_bias`];
  const y = new Float32Array(outSize);
  for (let o = 0; o < outSize; o++) {
    let sum = bias[o];
    const base = o * inSize;
    for (let i = 0; i < inSize; i++) sum += x[i] * weight[base + i];
    y[o] = sum;
  }
  return y;
}

export interface Forward {
  logits: number[][];
  boundary: number[];
  embedded: number[][];
  forward: number[][];
  backward: number[][];
  combined: number[][];
}

export function forward(
  rows: number[][],
  neighbors: Array<[number, number]>,
): Forward {
  const T = rows.length;
  const padding = BLOB.featureRows;

  // 1. Sum the embedding rows that fired for each token.
  const embedded: Float32Array[] = [];
  for (let t = 0; t < T; t++) {
    const vec = new Float32Array(HIDDEN);
    for (const row of rows[t]) {
      if (row === padding) continue;
      const base = row * HIDDEN;
      for (let c = 0; c < HIDDEN; c++) vec[c] += W.embedding[base + c];
    }
    embedded.push(vec);
  }

  // 2. Depthwise convolution over a five-token window, plus neighbour gathers.
  const encoded: Float32Array[] = [];
  for (let t = 0; t < T; t++) {
    const vec = new Float32Array(HIDDEN);
    for (let k = 0; k < 5; k++) {
      const source = t + k - 2;
      if (source < 0 || source >= T) continue;
      for (let c = 0; c < HIDDEN; c++) {
        vec[c] += W.convolution[k * HIDDEN + c] * embedded[source][c];
      }
    }
    for (let c = 0; c < HIDDEN; c++) vec[c] += W.encoder_bias[c];
    for (let side = 0; side < 2; side++) {
      const index = neighbors[t][side];
      if (index < 0) continue;
      for (let c = 0; c < HIDDEN; c++) {
        vec[c] += embedded[index][c] * W.neighbor_weights[side * HIDDEN + c];
      }
    }
    for (let c = 0; c < HIDDEN; c++) vec[c] = Math.tanh(vec[c]);
    encoded.push(vec);
  }

  // 3. Gates and candidates.
  const gate: Float32Array[] = [];
  const candidate: Float32Array[] = [];
  for (let t = 0; t < T; t++) {
    const g = linear(encoded[t], "gate", HIDDEN, HIDDEN);
    const c = linear(encoded[t], "candidate", HIDDEN, HIDDEN);
    const gv = new Float32Array(HIDDEN);
    const cv = new Float32Array(HIDDEN);
    for (let i = 0; i < HIDDEN; i++) {
      gv[i] = sigmoid(g[i]);
      cv[i] = (1 - gv[i]) * Math.tanh(c[i]);
    }
    gate.push(gv);
    candidate.push(cv);
  }

  // 4. state[t] = gate[t] * state[t-1] + candidate[t], both directions.
  const scan = (reverse: boolean): Float32Array[] => {
    const out: Float32Array[] = new Array(T);
    let state = new Float32Array(HIDDEN);
    for (let step = 0; step < T; step++) {
      const t = reverse ? T - 1 - step : step;
      const next = new Float32Array(HIDDEN);
      for (let c = 0; c < HIDDEN; c++) next[c] = gate[t][c] * state[c] + candidate[t][c];
      out[t] = next;
      state = next;
    }
    return out;
  };
  const fwd = scan(false);
  const bwd = scan(true);

  // 5. Combine, then a mean-pooled whole-phrase context.
  const combined: Float32Array[] = [];
  for (let t = 0; t < T; t++) {
    const pair = new Float32Array(HIDDEN * 2);
    pair.set(fwd[t], 0);
    pair.set(bwd[t], HIDDEN);
    const mixed = linear(pair, "combine", HIDDEN, HIDDEN * 2);
    const vec = new Float32Array(HIDDEN);
    for (let c = 0; c < HIDDEN; c++) vec[c] = Math.tanh(encoded[t][c] + mixed[c]);
    combined.push(vec);
  }

  const pooled = new Float32Array(HIDDEN);
  for (let t = 0; t < T; t++) for (let c = 0; c < HIDDEN; c++) pooled[c] += combined[t][c];
  for (let c = 0; c < HIDDEN; c++) pooled[c] /= Math.max(1, T);
  const gated = linear(pooled, "global", HIDDEN, HIDDEN);
  const context = new Float32Array(HIDDEN);
  for (let c = 0; c < HIDDEN; c++) context[c] = sigmoid(gated[c]) * pooled[c];

  // 6. Head.
  const roleCount = BLOB.labels.length;
  const logits: number[][] = [];
  const boundary: number[] = [];
  for (let t = 0; t < T; t++) {
    const joined = new Float32Array(HIDDEN * 2);
    joined.set(combined[t], 0);
    joined.set(context, HIDDEN);
    const hg = linear(joined, "head_gate", 16, HIDDEN * 2);
    const headGate = new Float32Array(16);
    for (let i = 0; i < 16; i++) headGate[i] = sigmoid(hg[i]);

    const wide = new Float32Array(HIDDEN * 2 + 16);
    wide.set(joined, 0);
    wide.set(headGate, HIDDEN * 2);
    const hidden = linear(wide, "head_hidden", 64, HIDDEN * 2 + 16);
    for (let i = 0; i < 64; i++) hidden[i] = Math.tanh(hidden[i]);

    const out = linear(hidden, "output", roleCount + 1, 64);
    logits.push(Array.from(out.slice(0, roleCount)));
    boundary.push(out[roleCount]);
  }

  const toArrays = (xs: Float32Array[]) => xs.map((v) => Array.from(v));
  return {
    logits,
    boundary,
    embedded: toArrays(embedded),
    forward: toArrays(fwd),
    backward: toArrays(bwd),
    combined: toArrays(combined),
  };
}

export function argmax(values: number[]): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}
