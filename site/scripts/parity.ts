/**
 * Prove the TypeScript forward pass matches PyTorch.
 *
 * The runtime is a reimplementation, so it has to be checked against the model
 * that produced the published numbers, not assumed correct. Any drift here
 * means the site is showing something the checkpoint never predicted.
 */
import cases from "../src/runtime/parity.json";
import { featurize } from "../src/runtime/featurize";
import { forward, argmax, LABELS } from "../src/runtime/model";
import { compile } from "../src/runtime/index";
import type { Schema } from "../src/runtime/schema";

let worstLogit = 0;
let rowMismatch = 0;
let labelMismatch = 0;
let astMismatch = 0;

for (const c of cases as any[]) {
  const { rows, neighbors } = featurize(c.tokens, c.schema as Schema);

  if (JSON.stringify(rows) !== JSON.stringify(c.rows)) rowMismatch++;
  if (JSON.stringify(neighbors) !== JSON.stringify(c.neighbors)) rowMismatch++;

  const out = forward(c.rows, c.neighbors);
  for (let t = 0; t < c.logits.length; t++) {
    for (let r = 0; r < c.logits[t].length; r++) {
      worstLogit = Math.max(worstLogit, Math.abs(out.logits[t][r] - c.logits[t][r]));
    }
  }

  const roles = out.logits.map((row) => LABELS[argmax(row)]);
  if (JSON.stringify(roles) !== JSON.stringify(c.labels)) labelMismatch++;

  const ast = compile(c.tokens, roles, c.schema as Schema);
  const expected = c.ast.map((a: any) => ({ ...a, neg: a.neg }));
  if (JSON.stringify(ast) !== JSON.stringify(expected)) astMismatch++;
}

console.log(`cases            ${cases.length}`);
console.log(`feature-row drift ${rowMismatch}`);
console.log(`label drift       ${labelMismatch}`);
console.log(`filter drift      ${astMismatch}`);
console.log(`max |logit delta| ${worstLogit.toExponential(3)}`);

const ok = rowMismatch === 0 && labelMismatch === 0 && astMismatch === 0 && worstLogit < 1e-3;
console.log(ok ? "\nPARITY OK" : "\nPARITY FAILED");
process.exit(ok ? 0 : 1);
