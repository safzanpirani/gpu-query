# gpu-query: a model that never learns your field names

## Overview

- **Visual thesis**: A 29,000-parameter model turns a typed phrase into an app's
  filter query without ever seeing the field names, because it reads structure
  rather than vocabulary.
- **Hook**: Your search bar currently needs either remembered DSL syntax or a
  round trip to an LLM. Neither is necessary.
- **Audience**: Web developers. No machine-learning background assumed.
- **Length**: ~80 seconds, 1920×1080, 60 fps, narrated.
- **Key insight**: The schema enters the featurizer as identity-free *match
  rows* and enters the compiler afterward. It never enters the weights. That
  separation is why the model works on a schema it has never seen.

## Look

Mixes the two reference films. From `gpu-time`: white background, dark Geist
Mono, blue/teal/violet accents, wordmark top-left, `NN / 08` scene counter
top-right, one short title per scene, and a bottom caption naming what is real
and what is sampled. From `gpu-lexer`: the thin outlined container that holds
the data flow, the token strip that persists across scenes, and activation
heatmaps rather than fully-connected edge spaghetti.

Narration is generated, matching gpu-time's approach. Segment boundaries drive
scene duration.

## Honesty contract

Both reference projects export real traces and say so on screen. This film does
the same.

- Every activation, state, and logit comes from `trace.json`, exported from a
  trained checkpoint by `export_trace.py`.
- Where nodes or channels are sampled for legibility, the bottom caption says
  so.
- The measured transfer result is stated once, with its corpus named.
- No speed claims. Nothing has been benchmarked in a browser yet.
- The closing states experimental status.

## Scenes

| # | Seconds | Beat | What is on screen | Real data |
|---|---|---|---|---|
| 1 | 0–9 | The ask | A search bar. `unread from sarah last week with attachments` types on. Two failed paths flash: a DSL cheatsheet, then a network request to an LLM that greys out. Caption: `no request leaves the page`. | — |
| 2 | 9–19 | One mechanical scan | The phrase separates into tokens. Under three representative tokens, compact feature marks appear: kind, length bucket, shape, case. Captions: `one scan` → `sparse feature rows`. | Real row IDs from `featurize.rows_for` |
| 3 | 19–32 | The schema stays outside | **The scene that carries the film.** The app's schema panel slides in: `unread: bool`, `sender: text`, `received: date`. Arrows carry it toward the model and are *cut* at the boundary — the field names visibly drop away. What passes through instead: `matched a field · kind bool`, `matched a value of the nearest preceding field`, `distance to field: 1`. Caption: `the model never sees a field name`. | Real match rows |
| 4 | 32–44 | Context, both directions | 32-channel state grids rise from the token strip. A gated signal sweeps left to right, then right to left, grids updating as it arrives. Caption: `state[t] = gate · state[t-1] + candidate`. | Real forward/backward states |
| 5 | 44–54 | Scores | For one token, the twelve role scores grow from zero. `VALUE` wins. A separate clause-boundary score is shown apart. | Real logits |
| 6 | 54–64 | Roles land, code takes over | Roles settle onto the token strip. The model half greys out; a `typescript` panel lights up. `band` resolves to `artist` — an alias the weights never saw. | Real predictions |
| 7 | 64–76 | The result | Two schema families face each other: trained on `issues · mail · files · commits`, evaluated on `contacts · music · recipes · shipments`, with a struck-through line showing zero shared vocabulary. The number lands: `0.9865 exact match`. Then the ablation: removing every word-identity row *raised* the score. | Measured |
| 8 | 76–82 | Lockup | `gpu-query` wordmark. `29,597 parameters · runs on your machine · experimental`. | — |

## Scene notes

**Scene 3 must not be rushed.** It is the only idea in the film that is not
already in `gpu-lexer` or `gpu-time`, and it is the reason the project works.
Hold on the severed arrows.

**Scene 7 states one number, once.** Both reference films resist stacking
metrics, and gpu-time's model card is explicit that a generated-distribution
score is not real-user accuracy. The caption names the corpus so the number
cannot be read as a general claim.

**The token strip persists from scene 2 to scene 6.** Same mobjects,
transformed — never rebuilt. That continuity is what makes gpu-lexer's film
read as one pipeline rather than eight slides.
