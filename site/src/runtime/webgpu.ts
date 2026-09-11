/**
 * WebGPU backend.
 *
 * The device, pipeline, weight buffer and grow-only IO buffers stay resident
 * between calls, so a warm batch pays for dispatch and readback only.
 *
 * One dispatch handles a whole batch: one workgroup per query. A single short
 * query is faster on the CPU because dispatch and readback dominate, which is
 * why `backend: "auto"` keeps small work there. gpu-time draws the same line at
 * 32 inputs or 512 tokens.
 */

import { buildKernel } from "./kernel.wgsl";
import { TENSOR_ORDER, WEIGHTS, HIDDEN, LABELS, FEATURE_ROWS } from "./weights";
import { SLOTS } from "./featurize";

export const MAX_TOKENS = 24;

let device: GPUDevice | null = null;
let pipeline: GPUComputePipeline | null = null;
let layout: GPUBindGroupLayout | null = null;
let weightBuffer: GPUBuffer | null = null;
let failed = false;

interface Slot {
  buffer: GPUBuffer;
  size: number;
}
const slots: Record<string, Slot | null> = {
  rows: null, neighbors: null, lengths: null, outputs: null, readback: null,
};

export function supported(): boolean {
  return typeof navigator !== "undefined" && !!navigator.gpu && !failed;
}

/** Flatten every tensor into one buffer, and record where each one starts. */
function packWeights(): { data: Float32Array; offsets: Record<string, number> } {
  const offsets: Record<string, number> = {};
  let total = 0;
  for (const name of TENSOR_ORDER) {
    offsets[name] = total;
    total += WEIGHTS[name].length;
  }
  const data = new Float32Array(total);
  for (const name of TENSOR_ORDER) data.set(WEIGHTS[name], offsets[name]);
  return { data, offsets };
}

export async function init(): Promise<boolean> {
  if (device) return true;
  if (!supported()) return false;
  try {
    const adapter = await navigator.gpu!.requestAdapter();
    if (!adapter) { failed = true; return false; }
    device = await adapter.requestDevice();
    device.lost.then(() => { device = null; pipeline = null; failed = true; });

    const { data, offsets } = packWeights();
    weightBuffer = device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(weightBuffer, 0, data as BufferSource);

    const code = buildKernel({
      featureRows: FEATURE_ROWS,
      slots: SLOTS,
      roles: LABELS.length,
      maxTokens: MAX_TOKENS,
      offsets,
    });
    const module = device.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === "error");
    if (errors.length) {
      console.error("WGSL compilation failed", errors);
      failed = true;
      return false;
    }

    layout = device.createBindGroupLayout({
      entries: [0, 1, 2, 3, 4].map((binding) => ({
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: binding === 4 ? "storage" : "read-only-storage" } as GPUBufferBindingLayout,
      })),
    });
    pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: "main" },
    });
    return true;
  } catch (error) {
    console.error("WebGPU init failed", error);
    failed = true;
    return false;
  }
}

/** Grow-only buffers: reallocate when a batch needs more room, never shrink. */
function ensure(name: string, size: number, usage: GPUBufferUsageFlags): GPUBuffer {
  const existing = slots[name];
  if (existing && existing.size >= size) return existing.buffer;
  existing?.buffer.destroy();
  const buffer = device!.createBuffer({ size, usage });
  slots[name] = { buffer, size };
  return buffer;
}

export interface Batch {
  rows: number[][][];
  neighbors: Array<Array<[number, number]>>;
}

/** Logits plus the boundary score, per token, for every query in the batch. */
export async function runBatch(batch: Batch): Promise<number[][][]> {
  if (!device || !pipeline) throw new Error("WebGPU not initialised");
  const count = batch.rows.length;
  const outs = LABELS.length + 1;

  const rowData = new Uint32Array(count * MAX_TOKENS * SLOTS).fill(FEATURE_ROWS);
  const neighborData = new Int32Array(count * MAX_TOKENS * 2).fill(-1);
  const lengthData = new Uint32Array(count);

  for (let s = 0; s < count; s++) {
    const tokens = Math.min(batch.rows[s].length, MAX_TOKENS);
    lengthData[s] = tokens;
    for (let t = 0; t < tokens; t++) {
      rowData.set(batch.rows[s][t], (s * MAX_TOKENS + t) * SLOTS);
      neighborData.set(batch.neighbors[s][t], (s * MAX_TOKENS + t) * 2);
    }
  }

  const outBytes = count * MAX_TOKENS * outs * 4;
  const rowsBuf = ensure("rows", rowData.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const neighBuf = ensure("neighbors", neighborData.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const lenBuf = ensure("lengths", Math.max(4, lengthData.byteLength), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const outBuf = ensure("outputs", outBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const readBuf = ensure("readback", outBytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);

  device.queue.writeBuffer(rowsBuf, 0, rowData as BufferSource);
  device.queue.writeBuffer(neighBuf, 0, neighborData as BufferSource);
  device.queue.writeBuffer(lenBuf, 0, lengthData as BufferSource);

  const bindGroup = device.createBindGroup({
    layout: layout!,
    entries: [
      { binding: 0, resource: { buffer: weightBuffer! } },
      { binding: 1, resource: { buffer: rowsBuf } },
      { binding: 2, resource: { buffer: neighBuf } },
      { binding: 3, resource: { buffer: lenBuf } },
      { binding: 4, resource: { buffer: outBuf } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(count);
  pass.end();
  encoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, outBytes);
  device.queue.submit([encoder.finish()]);

  await readBuf.mapAsync(GPUMapMode.READ, 0, outBytes);
  const view = new Float32Array(readBuf.getMappedRange(0, outBytes).slice(0));
  readBuf.unmap();

  const results: number[][][] = [];
  for (let s = 0; s < count; s++) {
    const tokens = Math.min(batch.rows[s].length, MAX_TOKENS);
    const perToken: number[][] = [];
    for (let t = 0; t < tokens; t++) {
      const base = (s * MAX_TOKENS + t) * outs;
      perToken.push(Array.from(view.subarray(base, base + outs)));
    }
    results.push(perToken);
  }
  return results;
}

export { HIDDEN };
