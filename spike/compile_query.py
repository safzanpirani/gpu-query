"""Turn predicted token roles into a filter AST.

This is the deterministic half. The model proposes roles; this code decides
whether they form a legal clause and resolves a surface form back to a real
field name using the caller's schema. A role sequence that cannot form a clause
is dropped rather than guessed at, which is the same contract gpu-time uses
when its compiler rejects an impossible schedule.

The schema never enters the model. It enters here.
"""

from __future__ import annotations

import match
from generate import LABEL_ID
from schema import BOOL, Schema

COMPARISON_FOR_LABEL = {
    LABEL_ID["OP_EQ"]: "eq",
    LABEL_ID["OP_LT"]: "lt",
    LABEL_ID["OP_GT"]: "gt",
    LABEL_ID["OP_CONTAINS"]: "contains",
}
CONJUNCTIONS = {LABEL_ID["AND"], LABEL_ID["OR"]}
VALUE_LABELS = {LABEL_ID["VALUE"], LABEL_ID["VALUE_CONT"]}
FIELD_LABELS = {LABEL_ID["FIELD"], LABEL_ID["FIELD_CONT"]}


def split_clauses(labels: list[int]) -> list[tuple[int, int]]:
    """Token spans between conjunctions."""
    spans: list[tuple[int, int]] = []
    start = 0
    for position, label in enumerate(labels):
        if label in CONJUNCTIONS:
            if position > start:
                spans.append((start, position))
            start = position + 1
    if start < len(labels):
        spans.append((start, len(labels)))
    return spans


def compile_query(
    tokens: list[str], labels: list[int], active: Schema
) -> list[dict]:
    entries = match.build_index(active)
    owners = match.enum_owners(active)
    clauses: list[dict] = []

    for start, end in split_clauses(labels):
        negated = False
        field_name: str | None = None
        field_kind: str | None = None
        comparison: str | None = None
        value_parts: list[str] = []

        for position in range(start, end):
            label = labels[position]
            token = tokens[position]
            if label == LABEL_ID["NEG"]:
                negated = True
            elif label in FIELD_LABELS and field_name is None:
                # Tolerant resolution, so an inflected or mistyped field word
                # still lands on the right field. Same matcher the featurizer
                # used, or the model would be scored against a field this
                # compiler never picks.
                found = match.resolve(token, active, entries)
                if found is not None:
                    field_name, field_kind = found.field, found.kind
            elif label in COMPARISON_FOR_LABEL and comparison is None:
                comparison = COMPARISON_FOR_LABEL[label]
            elif label in VALUE_LABELS:
                value_parts.append(token)

        if field_name is None:
            # A bare value can still name its field, when exactly one field owns
            # it: "open" means status:open. Two owners is genuinely ambiguous,
            # so the clause is dropped rather than guessed at.
            candidates = {
                owner
                for part in value_parts
                for owner in owners.get(part.lower(), ())
            }
            if len(candidates) != 1:
                continue
            field_name = candidates.pop()
            field_kind = active.field_by_name(field_name).kind

        if field_kind == BOOL and not value_parts:
            clauses.append(
                {"field": field_name, "cmp": "is", "value": True, "neg": negated}
            )
            continue

        if not value_parts:
            continue

        clauses.append(
            {
                "field": field_name,
                "cmp": comparison or "eq",
                "value": " ".join(value_parts),
                "neg": negated,
            }
        )

    return clauses


def matches(predicted: list[dict], expected: list[dict]) -> bool:
    return predicted == expected


if __name__ == "__main__":
    import generate
    import schema as schema_module

    schema_module.assert_disjoint()
    total = exact = 0
    for split in ("train", "eval"):
        for example in generate.dataset(split, 2000, seed=11):
            rebuilt = compile_query(example.tokens, example.labels, example.schema)
            total += 1
            exact += int(matches(rebuilt, example.ast))
    print(f"compiler round-trip on gold labels: {exact}/{total} = {exact / total:.4f}")
    # Anything below 1.0 is a compiler bug, not a model result.
    if exact != total:
        for split in ("train", "eval"):
            for example in generate.dataset(split, 2000, seed=11):
                rebuilt = compile_query(example.tokens, example.labels, example.schema)
                if rebuilt != example.ast:
                    print("\nMISMATCH")
                    print("  text    ", example.text)
                    print("  expected", example.ast)
                    print("  compiled", rebuilt)
                    break
            else:
                continue
            break
