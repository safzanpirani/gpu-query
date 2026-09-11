# gpu-query

`gpu-query` is an experimental neural parser for app search bars. A 29,597-parameter
model turns a typed phrase into a structured filter, using your schema without ever
learning your field names. It runs locally and sends nothing to a server.

```
anything  record  listens  and  performer  containing  uncensored  please
          FIELD   VALUE    AND  FIELD      OP_CONTAINS VALUE

→ [{ field: "album",  cmp: "eq",       value: "listens"    },
   { field: "artist", cmp: "contains", value: "uncensored" }]
```

This is a feasibility spike. Nothing is published to npm, and the WebGPU kernel
and the quantised export do not exist yet.

## How it works

One mechanical CPU scan splits the phrase into tokens and emits sparse feature
rows for each one: shape, case, length bucket, operator-lexicon membership. Your
schema is added as three more kinds of row, none of which carry a field name:

- this token matched a field, and the field has this kind
- this token matched a value belonging to the nearest preceding field
- this token is one position away from a field

The field words themselves are dropped at the boundary. The model sums those rows
into one vector per token, mixes nearby evidence with a five-token convolution
and two neighbour gathers, then carries context in both directions with gated
affine scans, `state[t] = gate[t] * state[t-1] + candidate[t]`. That scan is a
blocked parallel prefix scan, which is the operation a GPU is built for and the
reason the architecture maps onto one. A two-layer head emits one of twelve roles
per token plus a clause-boundary score.

Ordinary TypeScript then compiles the roles into a filter and resolves surface
forms back to real fields, tolerating aliases, plurals and single-character
typos. The compiler drops a role sequence that cannot form a legal clause rather
than guessing at one.

The architecture is a port of gpu-time's `TimeTagger`. Only the lexicon, the
label set, the generator and the CPU resolver are new.

## Accuracy

Trained on schemas drawn from issue trackers, mail, files and commits. Evaluated
on contacts, music, recipes and shipments, which share no field word, alias or
enum value with training. An assertion enforces that on import.

| | exact filter match |
|---|---|
| heldout (training schemas, fresh seed) | 1.0000 |
| transfer (unseen schemas) | 0.9888 |

The heldout row only checks whether the model memorised its training seed. The
transfer row is the result.

Removing the hashed word-identity feature rows raised transfer from 0.9865 to
0.9888. The model reads structure, and vocabulary hashes crowd it out.

Both figures come from a generated evaluation corpus of four thousand queries.
They do not establish accuracy on real user phrasing, and nothing here has been
benchmarked in a browser.

## Development

```sh
cd spike
uv sync
uv run python schema.py         # sample schemas + the disjointness guard
uv run python generate.py       # rendered queries with labels and gold filters
uv run python featurize.py      # the feature-row layout
uv run python compile_query.py  # resolver round-trip; MUST print 1.0000
uv run python train.py --device mps --epochs 40 --samples 60000 --run baseline
uv run python train.py --device mps --epochs 40 --samples 60000 --no-word-identity --run ablation
```

A 40-epoch run is under two minutes on an M-series Mac. These models do not need
a GPU; Python data generation dominates the loop.

Two invariants keep a result meaningful. The evaluation bank must share no
vocabulary with training, and `compile_query.py` must round-trip gold labels at
exactly 1.0000. Below that you are measuring resolver bugs rather than the model.

## Repository

- `spike/`: generator, featurizer, fuzzy schema matcher, compiler, training
- `video/`: the explainer film and its storyboard

## Video

A narrated explainer is in [`video/`](video). Every activation, score and label
on screen is read from a trace exported from a real checkpoint and cross-checked
against gold before it is written. See [`video/README.md`](video/README.md) for
what the film does and does not claim.

## Prior art

The architecture and the evaluation conventions come from
[gpu-lexer](https://github.com/vercel-labs/gpu-lexer) by Shu Ding and
[gpu-time](https://github.com/arikchakma/gpu-time) by Arik Chakma. This applies
their method to a new domain.

## License

MIT. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
