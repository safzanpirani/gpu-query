# gpu-query pipeline video

A ~2-minute, 1920×1080 / 60 fps narrated explainer. The storyboard is in
[`scenes.md`](scenes.md). The film follows one query through tokens, sparse
feature rows, the schema boundary, bidirectional gated state, role scores, and
the TypeScript compiler that turns roles into a filter.

It mixes the two reference films. From `gpu-time`: white ground, Geist Mono,
wordmark and scene counter, one short title per scene, and a bottom caption
naming what is real and what is sampled. From `gpu-lexer`: the token strip that
persists across scenes as the same transformed objects, and activation heatmaps
rather than fully-connected edges.

## Render

```sh
uv venv manim-env --python 3.13
uv pip install --python manim-env/bin/python manim manim-voiceover

../spike/.venv/bin/python export_trace.py          # needs model.pt
./manim-env/bin/manim -ql gpu_query_pipeline.py GpuQueryPipeline   # draft
./manim-env/bin/manim -qh --fps 60 gpu_query_pipeline.py GpuQueryPipeline
```

Requires `ffmpeg`, Geist Mono and Geist (both vendored in
`../reference/gpu-lexer/video/assets/fonts/`; copy to `~/Library/Fonts/`), and
Gemini API keys for narration. Keys come from `GEMINI_API_KEY` or
`~/.config/tome/config.json`. Narration is cached by text in `tts-cache/`, so
layout iterations are free as long as the words do not change.

Outputs land in `media/videos/gpu_query_pipeline/1080p60/`.

## Where the numbers come from

`model.pt` is a real checkpoint, trained by:

```sh
cd ../spike
uv run python train.py --device mps --epochs 40 --samples 60000 \
  --no-word-identity --run trace-model --save ../video/model.pt
```

`export_trace.py` then runs one example through that checkpoint with
`record_trace` enabled and writes `trace.json`: the summed embeddings, encoded
states, gates, forward and backward scans, combined states, all twelve role
logits, the boundary logit, the named feature rows that fired, and the schema
resolution for every token. It re-checks the result two ways before writing,
comparing predicted labels against gold labels and the compiled filter against
the gold filter. Both must agree.

Every activation, score and label on screen is read from that file. Nothing is
invented, rounded into a nicer shape, or hand-placed.

## The demo query

```
anything  record  listens  and  performer  containing  uncensored  please
          FIELD   VALUE    AND  FIELD      OP_CONTAINS VALUE
```

Chosen deterministically by `pick_demo`, which requires a two-clause example
carrying both hard cases at once. In this schema:

- `record` and `performer` are **aliases** for `album` and `artist`. The model
  never saw either word; the compiler resolves them afterward from the schema.
- `listens` and `uncensored` are **values that are also field names**.
  `listens` is an alias of the `plays` field, `uncensored` of `explicit`. A
  lookup calls both FIELD. The model calls both VALUE, from context alone.

The schema is drawn from the evaluation bank, whose vocabulary never appears in
training. That is the claim the film makes, so the trace honours it.

## What the film does not claim

- **No speed claims in the film.** The animation is a schematic and contains no
  profiling footage. The project does have a WebGPU backend and measured browser
  timings; they live on the site and in the root README, where a reader can rerun
  them.
- **The quoted 0.9888 is a generated-corpus score**, on four thousand queries
  over evaluation-bank schemas. The caption says so on screen. It is not a
  measurement of real user phrasing, and both reference model cards are
  explicit about the same distinction.
- **Channels and rows are sampled** where showing all of them would be
  illegible. Eight of thirty-two state channels per token, five of up to
  sixteen feature rows. Each caption says which.
- The closing states experimental status.

## Model depicted

29,597 parameters, 532 feature rows, 32 state channels, 16 head gates, 64 head
hidden values, 12 role outputs plus one clause-boundary output. Word-identity
feature rows are disabled in this checkpoint, which is why "the model never sees
a field name" is literal rather than a simplification. It is also the better
model, scoring 0.9888 against 0.9865 with those rows enabled.
