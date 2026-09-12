/**
 * The workbench.
 *
 * The schema is editable because the claim is falsifiable: the field names never
 * reach the weights, so renaming them should change which field a clause targets
 * and nothing else. "scramble names" is the fastest way for a visitor to check
 * that rather than take it on faith.
 */

import {
  PARAMETERS, parse, parseBatch, checkBackends, gpuSupported, warmGpu,
} from "./runtime";
import type { Field, Kind, Schema } from "./runtime";

const DEFAULT_SCHEMA: Schema = [
  { name: "album", kind: "text", aliases: ["record"], values: [] },
  { name: "artist", kind: "text", aliases: ["performer", "band"], values: [] },
  { name: "genre", kind: "enum", aliases: ["style"], values: ["ambient", "techno", "jazz", "shoegaze"] },
  { name: "plays", kind: "number", aliases: ["listens"], values: [] },
  { name: "released", kind: "date", aliases: ["issued"], values: [] },
  { name: "explicit", kind: "bool", aliases: ["uncensored"], values: [] },
];

const EXAMPLES = [
  "anything record listens and performer containing uncensored please",
  "show me style is jazz and listens over 100",
  "find performer containing aphex and not uncensored",
  "issued before last month and genre shoegaze",
  "albu containing blue",
];

const NONSENSE = [
  "wobbet", "franzle", "kroop", "milvane", "dazzik", "plonth",
  "survish", "yarnyx", "brellow", "quinth",
];

const KINDS: Kind[] = ["text", "enum", "number", "date", "bool"];

let schema: Schema = structuredClone(DEFAULT_SCHEMA);

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const queryInput = $<HTMLInputElement>("query");
const schemaRows = $<HTMLDivElement>("schema-rows");
const strip = $<HTMLDivElement>("strip");
const filterBox = $<HTMLPreElement>("filter");
const microsOut = $<HTMLSpanElement>("micros");

function renderSchema() {
  schemaRows.replaceChildren();
  schema.forEach((field, position) => {
    const row = document.createElement("div");
    row.className = "field-row";

    const name = document.createElement("input");
    name.value = field.name;
    name.spellcheck = false;
    name.addEventListener("input", () => {
      schema[position].name = name.value.trim() || "field";
      run();
    });

    const kind = document.createElement("select");
    for (const option of KINDS) {
      const element = document.createElement("option");
      element.value = option;
      element.textContent = option;
      if (option === field.kind) element.selected = true;
      kind.append(element);
    }
    kind.addEventListener("change", () => {
      schema[position].kind = kind.value as Kind;
      run();
    });

    const drop = document.createElement("button");
    drop.className = "drop";
    drop.textContent = "×";
    drop.title = "remove field";
    drop.addEventListener("click", () => {
      schema.splice(position, 1);
      renderSchema();
      run();
    });

    row.append(name, kind, drop);
    schemaRows.append(row);

    const detail = [
      field.aliases.length ? `aka ${field.aliases.join(", ")}` : "",
      field.values.length ? `{${field.values.join(" ")}}` : "",
    ].filter(Boolean).join("  ");
    if (detail) {
      const line = document.createElement("div");
      line.className = "alias-line";
      line.textContent = detail;
      schemaRows.append(line);
    }
  });
}

function escapeHtml(text: string): string {
  return text.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]!));
}

function renderFilter(clauses: ReturnType<typeof parse>["clauses"]) {
  if (clauses.length === 0) {
    filterBox.innerHTML = '<span class="empty">no clause the compiler would accept</span>';
    return;
  }
  const lines = clauses.map((clause) => {
    const not = clause.neg ? '<span class="n">not </span>' : "";
    const value = clause.value === true
      ? '<span class="v">true</span>'
      : `<span class="v">"${escapeHtml(String(clause.value))}"</span>`;
    return `  { ${not}<span class="f">${escapeHtml(clause.field)}</span> <span class="k">${clause.cmp}</span> ${value} }`;
  });
  filterBox.innerHTML = `[\n${lines.join(",\n")}\n]`;
}

function run() {
  const result = parse(queryInput.value, schema);
  microsOut.textContent = result.micros ? String(result.micros) : "—";

  strip.replaceChildren();
  result.tokens.forEach((token, i) => {
    const role = result.roles[i];
    const cell = document.createElement("div");
    cell.className = "tok";
    cell.dataset.role = role;

    const word = document.createElement("span");
    word.className = "word";
    word.textContent = token;

    const tag = document.createElement("span");
    tag.className = "role";
    tag.textContent = role === "O" ? "" : role;

    const resolved = result.resolved[i];
    if (resolved) {
      word.title = `matches ${resolved.field} (${resolved.kind}), ${
        ["exact", "stem", "prefix", "typo"][resolved.quality]
      }`;
    }

    cell.append(word, tag);
    strip.append(cell);
  });

  renderFilter(result.clauses);
}

function scramble() {
  // Renaming fields alone would leave the query pointing at words that no
  // longer exist, and the compiler would correctly refuse to build anything.
  // That reads as breakage rather than proof, so rewrite the query to the new
  // names at the same time. What the visitor should see is a schema full of
  // words that appear in no training corpus, parsing exactly as before.
  const pool = [...NONSENSE].sort(() => Math.random() - 0.5);
  const rename = new Map<string, string>();

  schema = schema.map((field, i) => {
    const name = pool[i % pool.length];
    const aliases = field.aliases.map((_, j) => pool[(i + j + 3) % pool.length] + "ic");
    rename.set(field.name.toLowerCase(), name);
    field.aliases.forEach((alias, j) => rename.set(alias.toLowerCase(), aliases[j]));
    return { ...field, name, aliases };
  });

  queryInput.value = tokenizeWords(queryInput.value)
    .map((token) => rename.get(token.toLowerCase()) ?? token)
    .join(" ");

  renderSchema();
  run();
}

/** Split on whitespace, keeping tokens intact for substitution. */
function tokenizeWords(text: string): string[] {
  return text.split(/\s+/).filter(Boolean);
}

function addField() {
  schema.push({ name: "newfield", kind: "text", aliases: [], values: [] } as Field);
  renderSchema();
  run();
}

function renderExamples() {
  const host = $<HTMLDivElement>("examples");
  for (const example of EXAMPLES) {
    const button = document.createElement("button");
    button.textContent = example.length > 38 ? example.slice(0, 36) + "…" : example;
    button.title = example;
    button.addEventListener("click", () => {
      queryInput.value = example;
      run();
    });
    host.append(button);
  }
}

$("scramble").addEventListener("click", scramble);
$("reset").addEventListener("click", () => {
  schema = structuredClone(DEFAULT_SCHEMA);
  renderSchema();
  run();
});
$("add-field").addEventListener("click", addField);
queryInput.addEventListener("input", run);

$("param-count").textContent = PARAMETERS.toLocaleString();
queryInput.value = EXAMPLES[0];
renderExamples();
renderSchema();
run();

// Test hook: lets the parity and benchmark checks run against a real device.
(window as any).__gpu = {
  supported: () => gpuSupported(),
  async check(n = 40) {
    const texts = Array.from({ length: n }, (_, i) => EXAMPLES[i % EXAMPLES.length]);
    return checkBackends(texts, schema);
  },
  async bench(n = 512) {
    const texts = Array.from({ length: n }, (_, i) => EXAMPLES[i % EXAMPLES.length]);
    await warmGpu();
    await parseBatch(texts, schema, "webgpu");            // warm
    await parseBatch(texts, schema, "cpu");
    const gpuRun = await parseBatch(texts, schema, "webgpu");
    const cpuRun = await parseBatch(texts, schema, "cpu");
    const same = JSON.stringify(gpuRun.results.map((r) => r.roles))
      === JSON.stringify(cpuRun.results.map((r) => r.roles));
    return {
      n,
      webgpuMs: +gpuRun.millis.toFixed(1),
      cpuMs: +cpuRun.millis.toFixed(1),
      speedup: +(cpuRun.millis / gpuRun.millis).toFixed(2),
      rolesIdentical: same,
    };
  },
};

// Batch benchmark, run on the visitor's own hardware.
const benchButton = $<HTMLButtonElement>("bench");
const benchOut = $<HTMLSpanElement>("bench-out");
benchButton.addEventListener("click", async () => {
  if (!gpuSupported()) {
    benchOut.textContent = "no WebGPU in this browser";
    return;
  }
  benchButton.disabled = true;
  benchOut.textContent = "warming…";
  try {
    const texts = Array.from({ length: 1024 }, (_, i) => EXAMPLES[i % EXAMPLES.length]);
    await warmGpu();
    // Warm both sides equally, twice. A single warm pass leaves the GPU path
    // paying pipeline setup and the JIT still cold, which understates it by
    // roughly 3x and would not match the numbers quoted anywhere else.
    for (let i = 0; i < 2; i++) {
      await parseBatch(texts, schema, "webgpu");
      await parseBatch(texts, schema, "cpu");
    }
    const gpuRun = await parseBatch(texts, schema, "webgpu");
    const cpuRun = await parseBatch(texts, schema, "cpu");
    const speedup = (cpuRun.millis / gpuRun.millis).toFixed(1);
    benchOut.innerHTML =
      `cpu ${cpuRun.millis.toFixed(0)} ms · webgpu ${gpuRun.millis.toFixed(0)} ms ` +
      `· <span class="win">${speedup}× faster</span>`;
  } catch (error) {
    benchOut.textContent = "benchmark failed, see console";
    console.error(error);
  } finally {
    benchButton.disabled = false;
  }
});

// Report backend parity on load, so the page never claims more than it checked.
void (async () => {
  if (!gpuSupported()) return;
  const texts = Array.from({ length: 16 }, (_, i) => EXAMPLES[i % EXAMPLES.length]);
  const check = await checkBackends(texts, schema);
  if (!check.ok) {
    $("bench-note").textContent =
      "The WGSL kernel disagreed with the CPU path on this device, so batches stay on the CPU.";
  }
})();
