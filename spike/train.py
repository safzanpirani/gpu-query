"""Train the query tagger and measure transfer to unseen schemas.

Three corpora, and only one of them is the result:

  train     training schemas, training seed
  heldout   training schemas, fresh seed  — memorization check
  transfer  EVALUATION schemas            — the actual question

`transfer` draws from a vocabulary bank that shares no field word with the
training bank, so a model that memorized "status names a field" scores zero
there. Exact AST match is the bar; token accuracy is diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import featurize
import generate
import schema as schema_module
from compile_query import compile_query
from model import QueryTagger

IGNORE = -100


def neighbors_for(tokens: list[str], active) -> list[tuple[int, int]]:
    """Nearest field-matching token on each side.

    Schema-derived, never label-derived. It lets a value token reach its field
    past the five-token convolution window.
    """
    index = featurize.SchemaIndex.build(active)
    lowered = [token.lower() for token in tokens]
    is_field = [token in index.field_kind for token in lowered]

    previous = [-1] * len(tokens)
    last = -1
    for position in range(len(tokens)):
        previous[position] = last
        if is_field[position]:
            last = position

    following = [-1] * len(tokens)
    upcoming = -1
    for position in range(len(tokens) - 1, -1, -1):
        following[position] = upcoming
        if is_field[position]:
            upcoming = position

    return list(zip(previous, following))


class Corpus:
    def __init__(self, split: str, count: int, seed: int, use_word_identity: bool):
        self.examples = generate.dataset(split, count, seed)
        self.rows = []
        self.neighbors = []
        for example in self.examples:
            self.rows.append(
                featurize.rows_for(example.tokens, example.schema, use_word_identity)
            )
            self.neighbors.append(neighbors_for(example.tokens, example.schema))
        self.lengths = np.array([len(item.tokens) for item in self.examples])

    def __len__(self) -> int:
        return len(self.examples)

    def batch(self, indices: np.ndarray, device: str):
        width = int(self.lengths[indices].max())
        size = len(indices)
        rows = np.full(
            (size, width, featurize.SLOTS), featurize.PADDING_ROW, dtype=np.int64
        )
        labels = np.full((size, width), IGNORE, dtype=np.int64)
        boundaries = np.zeros((size, width), dtype=np.float32)
        valid = np.zeros((size, width), dtype=np.bool_)
        neighbors = np.full((size, width, 2), -1, dtype=np.int64)
        for destination, source in enumerate(indices):
            example = self.examples[source]
            span = len(example.tokens)
            rows[destination, :span] = self.rows[source]
            labels[destination, :span] = example.labels
            boundaries[destination, :span] = example.boundaries
            valid[destination, :span] = True
            neighbors[destination, :span] = self.neighbors[source]
        return tuple(
            torch.from_numpy(value).to(device)
            for value in (rows, labels, boundaries, valid, neighbors)
        )

    def batches(self, size: int, rng: np.random.Generator | None = None):
        order = np.argsort(self.lengths, kind="stable")
        groups = [order[start : start + size] for start in range(0, len(order), size)]
        if rng is not None:
            rng.shuffle(groups)
        return groups


@torch.no_grad()
def evaluate(model: QueryTagger, corpus: Corpus, size: int, device: str) -> dict:
    model.eval()
    correct = total = exact_ast = sequences = 0
    for indices in corpus.batches(size):
        rows, labels, _boundaries, valid, neighbors = corpus.batch(indices, device)
        logits, _boundary = model(rows, valid, neighbors)
        predicted = logits.argmax(-1)
        mask = labels >= 0
        correct += ((predicted == labels) & mask).sum().item()
        total += mask.sum().item()
        host = predicted.cpu().numpy()
        for offset, source in enumerate(indices):
            example = corpus.examples[source]
            span = len(example.tokens)
            rebuilt = compile_query(
                example.tokens, list(host[offset, :span]), example.schema
            )
            exact_ast += int(rebuilt == example.ast)
            sequences += 1
    return {
        "tokenAccuracy": correct / max(1, total),
        "exactAst": exact_ast / max(1, sequences),
        "sequences": sequences,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--samples", type=int, default=40000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qat-start", type=int, default=6)
    parser.add_argument("--no-word-identity", action="store_true")
    parser.add_argument("--run", default="baseline")
    parser.add_argument("--save", default=None,
                        help="write the trained weights here, for trace export")
    arguments = parser.parse_args()

    schema_module.assert_disjoint()
    if arguments.device == "cuda" and not torch.cuda.is_available():
        arguments.device = "cpu"
    use_identity = not arguments.no_word_identity

    started = time.perf_counter()
    training = Corpus("train", arguments.samples, arguments.seed, use_identity)
    heldout = Corpus("train", 4000, arguments.seed + 9973, use_identity)
    transfer = Corpus("eval", 4000, arguments.seed + 7717, use_identity)
    prepared = time.perf_counter() - started

    model = QueryTagger(featurize.FEATURE_ROWS, generate.LABEL_COUNT).to(
        arguments.device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate)
    rng = np.random.default_rng(arguments.seed)

    print(
        json.dumps(
            {
                "run": arguments.run,
                "parameters": model.parameter_count(),
                "featureRows": featurize.FEATURE_ROWS,
                "wordIdentity": use_identity,
                "device": arguments.device,
                "prepareSeconds": round(prepared, 1),
            }
        ),
        flush=True,
    )

    history = []
    step = 0
    warmup = 200
    for epoch in range(arguments.epochs):
        model.train()
        model.qat = epoch >= arguments.qat_start
        groups = training.batches(arguments.batch, rng)
        losses = []
        epoch_started = time.perf_counter()
        for number, indices in enumerate(groups):
            step += 1
            progress = (epoch + number / len(groups)) / arguments.epochs
            rate = (
                1e-4
                + (arguments.learning_rate - 1e-4)
                * (1 + math.cos(math.pi * progress))
                / 2
            ) * min(1.0, step / warmup)
            for group in optimizer.param_groups:
                group["lr"] = rate

            rows, labels, boundaries, valid, neighbors = training.batch(
                indices, arguments.device
            )
            logits, boundary_logits = model(rows, valid, neighbors)
            mask = labels >= 0
            role_loss = F.cross_entropy(
                logits.reshape(-1, model.role_classes),
                labels.reshape(-1),
                ignore_index=IGNORE,
                label_smoothing=0.03,
            )
            boundary_loss = F.binary_cross_entropy_with_logits(
                boundary_logits[mask],
                boundaries[mask],
                pos_weight=torch.tensor(4.0, device=arguments.device),
            )
            loss = role_loss + 0.5 * boundary_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.detach())

        was_qat = model.qat
        model.qat = True
        entry = {
            "epoch": epoch + 1,
            "loss": round(torch.stack(losses).mean().item(), 4),
            "seconds": round(time.perf_counter() - epoch_started, 1),
            "heldout": evaluate(model, heldout, arguments.batch, arguments.device),
            "transfer": evaluate(model, transfer, arguments.batch, arguments.device),
        }
        model.qat = was_qat
        history.append(entry)
        print(json.dumps(entry), flush=True)

    if arguments.save:
        # The video claims real activations, so it needs real weights.
        torch.save(
            {
                "model": model.state_dict(),
                "featureRows": featurize.FEATURE_ROWS,
                "labels": generate.LABELS,
                "parameters": model.parameter_count(),
                "wordIdentity": use_identity,
                "config": vars(arguments),
                "final": history[-1],
            },
            arguments.save,
        )
        print(json.dumps({"stage": "saved", "path": arguments.save}), flush=True)

    output = Path(f"results-{arguments.run}.json")
    output.write_text(
        json.dumps(
            {
                "config": vars(arguments),
                "parameters": model.parameter_count(),
                "featureRows": featurize.FEATURE_ROWS,
                "history": history,
                "totalSeconds": round(time.perf_counter() - started, 1),
            },
            indent=2,
        )
        + "\n"
    )
    final = history[-1]
    print(
        json.dumps(
            {
                "stage": "completed",
                "run": arguments.run,
                "heldoutExactAst": final["heldout"]["exactAst"],
                "transferExactAst": final["transfer"]["exactAst"],
                "totalSeconds": round(time.perf_counter() - started, 1),
                "results": str(output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
