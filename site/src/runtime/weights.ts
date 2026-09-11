/**
 * Decoded model weights, shared by the CPU and WebGPU backends.
 *
 * Stored as int6 values in int8 with one scale per tensor, which is the
 * quantisation the model trained under. Decoded once at module load.
 */

import blob from "./weights.json";

export const HIDDEN = 32;

interface Packed { shape: number[]; scale: number; data: string }
interface Blob {
  featureRows: number; hidden: number; slots: number; labels: string[];
  parameters: number; bits: number; transferExactAst: number;
  tensors: Record<string, Packed>;
}

const BLOB = blob as unknown as Blob;

function decode(packed: Packed): Float32Array {
  const binary = atob(packed.data);
  const out = new Float32Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    // int8 two's complement, then the tensor scale.
    out[i] = ((binary.charCodeAt(i) << 24) >> 24) * packed.scale;
  }
  return out;
}

/** Fixed order, so the GPU weight buffer packs identically every run. */
export const TENSOR_ORDER = Object.keys(BLOB.tensors);

export const WEIGHTS: Record<string, Float32Array> = {};
for (const name of TENSOR_ORDER) WEIGHTS[name] = decode(BLOB.tensors[name]);

export const LABELS = BLOB.labels;
export const PARAMETERS = BLOB.parameters;
export const FEATURE_ROWS = BLOB.featureRows;
export const TRANSFER_EXACT_AST = BLOB.transferExactAst;
