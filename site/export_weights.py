"""Export the trained checkpoint for the browser runtime.

Weights ship as int6 values stored in int8 with one scale per tensor, which is
the quantisation the model was trained under. The browser decodes them once at
load.

Also writes a parity fixture. The TypeScript forward pass is a reimplementation,
so it has to be proved against PyTorch rather than assumed correct; `parity.json`
holds real inputs and the logits PyTorch produced for them.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "spike"))

import featurize  # noqa: E402
import generate  # noqa: E402
import schema as schema_module  # noqa: E402
from model import QueryTagger, quantize  # noqa: E402
from train import neighbors_for  # noqa: E402

CHECKPOINT = ROOT / "video" / "model.pt"
OUT = Path(__file__).resolve().parent / "src" / "runtime"
BITS = 6


def pack(tensor: torch.Tensor) -> dict:
    """Int6 values in int8 storage, with the scale that decodes them."""
    maximum = (1 << (BITS - 1)) - 1
    scale = float(tensor.abs().max() / maximum)
    scale = max(scale, 1e-8)
    codes = torch.round(tensor / scale).clamp(-maximum, maximum).to(torch.int8)
    return {
        "shape": list(tensor.shape),
        "scale": scale,
        "data": base64.b64encode(codes.numpy().tobytes()).decode("ascii"),
    }


def main() -> None:
    schema_module.assert_disjoint()
    blob = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

    model = QueryTagger(blob["featureRows"], len(blob["labels"]))
    model.load_state_dict(blob["model"])
    model.eval()
    model.qat = True

    tensors = {}
    for name, value in model.state_dict().items():
        # Quantise exactly as the forward pass does, then store the codes.
        tensors[name] = pack(quantize(value, BITS).detach())

    payload = {
        "featureRows": blob["featureRows"],
        "hidden": 32,
        "slots": featurize.SLOTS,
        "labels": blob["labels"],
        "parameters": blob["parameters"],
        "bits": BITS,
        "transferExactAst": blob["final"]["transfer"]["exactAst"],
        "tensors": tensors,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "weights.json").write_text(json.dumps(payload))

    # Parity fixture: real inputs and the logits PyTorch gives for them.
    cases = []
    for example in generate.dataset("eval", 40, seed=90210)[:20]:
        rows = featurize.rows_for(example.tokens, example.schema, use_word_identity=False)
        neigh = neighbors_for(example.tokens, example.schema)
        with torch.no_grad():
            logits, boundary = model(
                torch.tensor([rows], dtype=torch.long),
                torch.ones((1, len(example.tokens)), dtype=torch.bool),
                torch.tensor([neigh], dtype=torch.long),
            )
        cases.append({
            "text": example.text,
            "tokens": example.tokens,
            "schema": [
                {"name": f.name, "kind": f.kind, "aliases": f.aliases, "values": f.values}
                for f in example.schema.fields
            ],
            "rows": rows,
            "neighbors": neigh,
            "logits": [[round(v, 5) for v in row] for row in logits[0].tolist()],
            "boundary": [round(v, 5) for v in boundary[0].tolist()],
            "labels": [generate.LABELS[l] for l in logits.argmax(-1)[0].tolist()],
            "ast": example.ast,
        })
    (OUT / "parity.json").write_text(json.dumps(cases, indent=1))

    size = (OUT / "weights.json").stat().st_size
    print(json.dumps({
        "weights": f"{size/1024:.1f} KiB",
        "parameters": blob["parameters"],
        "featureRows": blob["featureRows"],
        "parityCases": len(cases),
    }, indent=2))


if __name__ == "__main__":
    main()
