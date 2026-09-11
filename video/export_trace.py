"""Export a real forward-pass trace for the explainer film.

Follows gpu-time's `export-data.ts`: run one example through the trained model,
capture every intermediate, and independently re-check the result before
writing it. The film asserts these are actual activations, so nothing here may
be invented or rounded into a nicer shape.

The demo schema is drawn from the EVALUATION bank, whose vocabulary never
appears in training. That is the claim the film makes, so the trace has to
honour it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

SPIKE = Path(__file__).resolve().parent.parent / "spike"
sys.path.insert(0, str(SPIKE))

import featurize  # noqa: E402
import generate  # noqa: E402
import match  # noqa: E402
import schema as schema_module  # noqa: E402
from compile_query import compile_query  # noqa: E402
from model import QueryTagger  # noqa: E402
from train import neighbors_for  # noqa: E402

CHECKPOINT = Path(__file__).resolve().parent / "model.pt"
OUTPUT = Path(__file__).resolve().parent / "trace.json"


def pick_demo(limit: int = 6000):
    """An eval-bank example that carries both hard cases at once.

    Wanted, in order: two clauses joined by AND (legible), at least one field
    word that is not its canonical name (an alias or a typo the weights never
    saw), and at least one VALUE token that is ALSO a field word in the same
    schema. That last case is the one lookup cannot solve, so it is the one
    worth putting on screen.
    """
    best = None
    for example in generate.dataset("eval", limit, seed=4242):
        if not 7 <= len(example.tokens) <= 11 or len(example.ast) != 2:
            continue
        if any(generate.LABELS[label] == "OR" for label in example.labels):
            continue
        canonical = {item.name for item in example.schema.fields}
        surfaces = {
            surface.lower()
            for item in example.schema.fields
            for surface in item.surface_forms()
        }
        field_words = [
            token
            for token, label in zip(example.tokens, example.labels)
            if generate.LABELS[label] == "FIELD"
        ]
        alias_like = [w for w in field_words if w.lower() not in canonical]
        collisions = [
            token
            for token, label in zip(example.tokens, example.labels)
            if generate.LABELS[label] in ("VALUE", "VALUE_CONT")
            and token.lower() in surfaces
        ]
        if not alias_like or not collisions:
            continue
        score = len(collisions) * 100 + len(alias_like) * 10 - len(example.tokens)
        if best is None or score > best[0]:
            best = (score, example, alias_like, collisions)
    if best is None:
        raise SystemExit("no example carried both hard cases; widen the search")
    return best[1], best[2], best[3]


def main() -> None:
    schema_module.assert_disjoint()
    if not CHECKPOINT.exists():
        raise SystemExit(f"no checkpoint at {CHECKPOINT}; train with --save first")

    blob = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()

    model = QueryTagger(blob["featureRows"], len(blob["labels"]))
    model.load_state_dict(blob["model"])
    model.eval()
    model.qat = True  # score the model that would ship, not the float one
    model.record_trace = True

    example, alias_like, collisions = pick_demo()
    tokens = example.tokens
    rows = featurize.rows_for(tokens, example.schema, use_word_identity=False)
    neighbors = neighbors_for(tokens, example.schema)

    row_tensor = torch.tensor([rows], dtype=torch.long)
    valid = torch.ones((1, len(tokens)), dtype=torch.bool)
    neighbor_tensor = torch.tensor([neighbors], dtype=torch.long)

    with torch.no_grad():
        logits, boundary = model(row_tensor, valid, neighbor_tensor)

    predicted = logits.argmax(-1)[0].tolist()
    rebuilt = compile_query(tokens, predicted, example.schema)

    # Independent re-check, exactly as gpu-time re-derives its head.
    agree = rebuilt == example.ast
    label_agree = predicted == list(example.labels)

    trace = model.trace
    def grid(name):
        return trace[name][0].tolist()

    # Which feature rows fired, named, so scene 3 can show real rows.
    block_of = {}
    for block, size in featurize._blocks:
        for offset in range(size):
            block_of[featurize.OFFSET[block] + offset] = (block, offset)
    named_rows = []
    for token_rows in rows:
        active = []
        for value in token_rows:
            if value == featurize.PADDING_ROW:
                continue
            block, offset = block_of[value]
            active.append({"row": value, "block": block, "offset": offset})
        named_rows.append(active)

    resolved = [
        (lambda found: {"field": found.field, "kind": found.kind,
                        "quality": ["exact", "stem", "prefix", "typo"][found.quality]}
         if found else None)(match.resolve(token, example.schema))
        for token in tokens
    ]

    payload = {
        "checkpoint": {
            "path": CHECKPOINT.name,
            "sha256": digest,
            "parameters": blob["parameters"],
            "featureRows": blob["featureRows"],
            "wordIdentity": blob["wordIdentity"],
            "transferExactAst": blob["final"]["transfer"]["exactAst"],
            "heldoutExactAst": blob["final"]["heldout"]["exactAst"],
        },
        "demo": {
            "text": example.text,
            "tokens": tokens,
            "goldLabels": [generate.LABELS[label] for label in example.labels],
            "predictedLabels": [generate.LABELS[label] for label in predicted],
            "goldAst": example.ast,
            "compiledAst": rebuilt,
            "aliasLikeFieldWords": alias_like,
            "valueTokensThatAreFieldWords": collisions,
            "schemaDomain": example.schema.domain,
            "schema": [
                {"name": f.name, "kind": f.kind, "aliases": f.aliases, "values": f.values}
                for f in example.schema.fields
            ],
            "resolved": resolved,
            "featureRows": named_rows,
            "neighbors": neighbors,
        },
        "activations": {
            "embedded": grid("embedded"),
            "encoded": grid("encoded"),
            "gate": grid("gate"),
            "forward": grid("forward"),
            "backward": grid("backward"),
            "combined": grid("combined"),
            "logits": logits[0].tolist(),
            "boundary": boundary[0].tolist(),
            "labels": blob["labels"],
        },
        "checks": {
            "compiledMatchesGold": agree,
            "labelsMatchGold": label_agree,
        },
    }

    if not agree:
        print("WARNING: demo example does not compile to its gold AST", file=sys.stderr)

    OUTPUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps({
        "text": example.text,
        "schemaDomain": example.schema.domain,
        "aliasLike": alias_like,
        "valueIsFieldWord": collisions,
        "compiledMatchesGold": agree,
        "labelsMatchGold": label_agree,
        "parameters": blob["parameters"],
        "transferExactAst": blob["final"]["transfer"]["exactAst"],
        "bytes": OUTPUT.stat().st_size,
    }, indent=2))


if __name__ == "__main__":
    main()
