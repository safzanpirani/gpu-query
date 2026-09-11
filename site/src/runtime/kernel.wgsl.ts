/**
 * The WGSL compute kernel, with model constants spliced in at build time.
 *
 * One workgroup per query, one thread per hidden channel. Every channel is
 * independent through the scans, so the affine recurrence runs as a plain
 * sequential walk inside each thread while 32 channels and every query in the
 * batch run in parallel. That is the shape the architecture was chosen for.
 *
 * Four workgroup arrays are reused rather than allocated per stage, which keeps
 * shared memory at roughly 13 KiB:
 *
 *   A  embedded -> forward scan
 *   B  encoded  (held throughout)
 *   C  gate     -> combined
 *   D  candidate -> backward scan, written in place
 *
 * Overwriting candidate with the backward scan is safe because thread c only
 * ever touches channel c, and it reads position t before writing it.
 */

export interface KernelShape {
  featureRows: number;
  slots: number;
  roles: number;
  maxTokens: number;
  offsets: Record<string, number>;
}

export function buildKernel(shape: KernelShape): string {
  const o = shape.offsets;
  return /* wgsl */ `
const HIDDEN: u32 = 32u;
const SLOTS: u32 = ${shape.slots}u;
const MAX_TOKENS: u32 = ${shape.maxTokens}u;
const ROLES: u32 = ${shape.roles}u;
const OUTS: u32 = ${shape.roles + 1}u;
const PADDING_ROW: u32 = ${shape.featureRows}u;

const O_EMBEDDING: u32 = ${o.embedding}u;
const O_ENCODER_BIAS: u32 = ${o.encoder_bias}u;
const O_CONVOLUTION: u32 = ${o.convolution}u;
const O_NEIGHBOR: u32 = ${o.neighbor_weights}u;
const O_GATE_W: u32 = ${o.gate_weight}u;
const O_GATE_B: u32 = ${o.gate_bias}u;
const O_CAND_W: u32 = ${o.candidate_weight}u;
const O_CAND_B: u32 = ${o.candidate_bias}u;
const O_COMBINE_W: u32 = ${o.combine_weight}u;
const O_COMBINE_B: u32 = ${o.combine_bias}u;
const O_GLOBAL_W: u32 = ${o.global_weight}u;
const O_GLOBAL_B: u32 = ${o.global_bias}u;
const O_HG_W: u32 = ${o.head_gate_weight}u;
const O_HG_B: u32 = ${o.head_gate_bias}u;
const O_HH_W: u32 = ${o.head_hidden_weight}u;
const O_HH_B: u32 = ${o.head_hidden_bias}u;
const O_OUT_W: u32 = ${o.output_weight}u;
const O_OUT_B: u32 = ${o.output_bias}u;

@group(0) @binding(0) var<storage, read> W: array<f32>;
@group(0) @binding(1) var<storage, read> rows: array<u32>;
@group(0) @binding(2) var<storage, read> neighbors: array<i32>;
@group(0) @binding(3) var<storage, read> lengths: array<u32>;
@group(0) @binding(4) var<storage, read_write> outputs: array<f32>;

var<workgroup> A: array<f32, ${shape.maxTokens * 32}>;
var<workgroup> B: array<f32, ${shape.maxTokens * 32}>;
var<workgroup> C: array<f32, ${shape.maxTokens * 32}>;
var<workgroup> D: array<f32, ${shape.maxTokens * 32}>;
var<workgroup> pooled: array<f32, 32>;
var<workgroup> ctx: array<f32, 32>;
var<workgroup> joined: array<f32, 64>;
var<workgroup> headGate: array<f32, 16>;
var<workgroup> wide: array<f32, 80>;
var<workgroup> hidden: array<f32, 64>;

fn sigmoid(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }

@compute @workgroup_size(32)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let seq = wg.x;
  let c = lid.x;
  let count = min(lengths[seq], MAX_TOKENS);

  // 1. Sum the embedding rows that fired for each token.
  for (var t: u32 = 0u; t < count; t = t + 1u) {
    var total = 0.0;
    let base = (seq * MAX_TOKENS + t) * SLOTS;
    for (var s: u32 = 0u; s < SLOTS; s = s + 1u) {
      let row = rows[base + s];
      if (row != PADDING_ROW) {
        total = total + W[O_EMBEDDING + row * HIDDEN + c];
      }
    }
    A[t * HIDDEN + c] = total;
  }
  workgroupBarrier();

  // 2. Depthwise convolution over five tokens, plus the two neighbour gathers.
  for (var t: u32 = 0u; t < count; t = t + 1u) {
    var total = W[O_ENCODER_BIAS + c];
    for (var k: u32 = 0u; k < 5u; k = k + 1u) {
      let source = i32(t) + i32(k) - 2;
      if (source >= 0 && source < i32(count)) {
        total = total + W[O_CONVOLUTION + k * HIDDEN + c] * A[u32(source) * HIDDEN + c];
      }
    }
    for (var side: u32 = 0u; side < 2u; side = side + 1u) {
      let index = neighbors[(seq * MAX_TOKENS + t) * 2u + side];
      if (index >= 0) {
        total = total + A[u32(index) * HIDDEN + c] * W[O_NEIGHBOR + side * HIDDEN + c];
      }
    }
    B[t * HIDDEN + c] = tanh(total);
  }
  workgroupBarrier();

  // 3. Gate and candidate.
  for (var t: u32 = 0u; t < count; t = t + 1u) {
    var g = W[O_GATE_B + c];
    var k = W[O_CAND_B + c];
    for (var i: u32 = 0u; i < HIDDEN; i = i + 1u) {
      let value = B[t * HIDDEN + i];
      g = g + value * W[O_GATE_W + c * HIDDEN + i];
      k = k + value * W[O_CAND_W + c * HIDDEN + i];
    }
    let gv = sigmoid(g);
    C[t * HIDDEN + c] = gv;
    D[t * HIDDEN + c] = (1.0 - gv) * tanh(k);
  }
  workgroupBarrier();

  // 4. state[t] = gate[t] * state[t-1] + candidate[t], forward into A.
  var state = 0.0;
  for (var t: u32 = 0u; t < count; t = t + 1u) {
    state = C[t * HIDDEN + c] * state + D[t * HIDDEN + c];
    A[t * HIDDEN + c] = state;
  }
  // Backward, overwriting candidate in place. Safe: this thread owns channel c
  // and reads position t before writing it.
  state = 0.0;
  for (var step: u32 = 0u; step < count; step = step + 1u) {
    let t = count - 1u - step;
    state = C[t * HIDDEN + c] * state + D[t * HIDDEN + c];
    D[t * HIDDEN + c] = state;
  }
  workgroupBarrier();

  // 5. Combine into C, then mean-pool the phrase.
  for (var t: u32 = 0u; t < count; t = t + 1u) {
    var mixed = W[O_COMBINE_B + c];
    for (var i: u32 = 0u; i < HIDDEN; i = i + 1u) {
      mixed = mixed + A[t * HIDDEN + i] * W[O_COMBINE_W + c * (HIDDEN * 2u) + i];
      mixed = mixed + D[t * HIDDEN + i] * W[O_COMBINE_W + c * (HIDDEN * 2u) + HIDDEN + i];
    }
    C[t * HIDDEN + c] = tanh(B[t * HIDDEN + c] + mixed);
  }
  workgroupBarrier();

  var total = 0.0;
  for (var t: u32 = 0u; t < count; t = t + 1u) { total = total + C[t * HIDDEN + c]; }
  pooled[c] = total / max(1.0, f32(count));
  workgroupBarrier();

  var gatedGlobal = W[O_GLOBAL_B + c];
  for (var i: u32 = 0u; i < HIDDEN; i = i + 1u) {
    gatedGlobal = gatedGlobal + pooled[i] * W[O_GLOBAL_W + c * HIDDEN + i];
  }
  ctx[c] = sigmoid(gatedGlobal) * pooled[c];
  workgroupBarrier();

  // 6. Head, one token at a time so 32 threads can cover 16/64/13 widths.
  for (var t: u32 = 0u; t < count; t = t + 1u) {
    joined[c] = C[t * HIDDEN + c];
    joined[HIDDEN + c] = ctx[c];
    workgroupBarrier();

    if (c < 16u) {
      var sum = W[O_HG_B + c];
      for (var i: u32 = 0u; i < 64u; i = i + 1u) {
        sum = sum + joined[i] * W[O_HG_W + c * 64u + i];
      }
      headGate[c] = sigmoid(sum);
    }
    workgroupBarrier();

    wide[c] = joined[c];
    wide[32u + c] = joined[32u + c];
    if (c < 16u) { wide[64u + c] = headGate[c]; }
    workgroupBarrier();

    for (var half: u32 = 0u; half < 2u; half = half + 1u) {
      let index = half * 32u + c;
      var sum = W[O_HH_B + index];
      for (var i: u32 = 0u; i < 80u; i = i + 1u) {
        sum = sum + wide[i] * W[O_HH_W + index * 80u + i];
      }
      hidden[index] = tanh(sum);
    }
    workgroupBarrier();

    if (c < OUTS) {
      var sum = W[O_OUT_B + c];
      for (var i: u32 = 0u; i < 64u; i = i + 1u) {
        sum = sum + hidden[i] * W[O_OUT_W + c * 64u + i];
      }
      outputs[(seq * MAX_TOKENS + t) * OUTS + c] = sum;
    }
    workgroupBarrier();
  }
}
`;
}
