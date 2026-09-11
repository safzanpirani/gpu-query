"""Sparse feature rows for the gpu-query tagger.

Two families of rows compete here, and the spike exists to find out which one
the model uses.

Mechanical rows carry word identity through hashes. They let the model memorize
that "status" names a field, and they are worthless on a schema it has never
seen.

Schema rows carry no identity at all. They say only that a token matched some
field of some kind, or matched an enum value belonging to the nearest preceding
field. A model leaning on these transfers to any schema.

`use_word_identity=False` folds the hash blocks onto the padding row, which is
the ablation that separates the two stories.
"""

from __future__ import annotations

from dataclasses import dataclass

import match
from schema import BOOL, DATE, ENUM, NUMBER, TEXT, Schema

SLOTS = 16

# Closed English operator lexicon. Not app-specific, so this is not a leak.
OPERATOR_WORDS = [
    "and",
    "or",
    "not",
    "no",
    "without",
    "excluding",
    "is",
    "before",
    "after",
    "since",
    "over",
    "under",
    "above",
    "below",
    "than",
    "least",
    "most",
    "more",
    "less",
    "fewer",
    "prior",
    "later",
    "earlier",
    "containing",
    "mentioning",
    "matching",
    "with",
    ":",
    "=",
]
OPERATOR_INDEX = {word: index for index, word in enumerate(OPERATOR_WORDS)}

KIND_ROWS = 3
LENGTH_ROWS = 8
CHAR_ROWS = 40
WORD_HASH_ROWS = 256
DEVOWEL_ROWS = 128
FLAG_ROWS = 4
FIELD_MATCH_ROWS = 6  # none, bool, enum, date, number, text
MATCH_QUALITY_ROWS = 5  # none, exact, stem, prefix, typo
ENUM_ANY_ROWS = 1
ENUM_PRECEDING_ROWS = 1
DISTANCE_ROWS = 6  # 0..4 tokens since a field, plus a distinct "no field yet"
OPERATOR_ROWS = len(OPERATOR_WORDS) + 1
POSITION_ROWS = 4

_blocks = [
    ("kind", KIND_ROWS),
    ("length", LENGTH_ROWS),
    ("first_char", CHAR_ROWS),
    ("last_char", CHAR_ROWS),
    ("word_hash", WORD_HASH_ROWS),
    ("devowel_hash", DEVOWEL_ROWS),
    ("flags", FLAG_ROWS),
    ("field_match", FIELD_MATCH_ROWS),
    ("match_quality", MATCH_QUALITY_ROWS),
    ("enum_any", ENUM_ANY_ROWS),
    ("enum_preceding", ENUM_PRECEDING_ROWS),
    ("distance", DISTANCE_ROWS),
    ("operator", OPERATOR_ROWS),
    ("position", POSITION_ROWS),
]

OFFSET: dict[str, int] = {}
_cursor = 0
for _name, _size in _blocks:
    OFFSET[_name] = _cursor
    _cursor += _size
FEATURE_ROWS = _cursor
PADDING_ROW = FEATURE_ROWS

IDENTITY_BLOCKS = ("word_hash", "devowel_hash")
KIND_OF_FIELD = {BOOL: 1, ENUM: 2, DATE: 3, NUMBER: 4, TEXT: 5}
LENGTH_BUCKETS = [1, 2, 3, 4, 6, 8, 12, 10**9]


def _hash(text: str) -> int:
    value = 2166136261
    for character in text:
        value = ((value ^ ord(character)) * 16777619) & 0xFFFFFFFF
    return value


def _character_class(character: str) -> int:
    code = ord(character.lower())
    if 97 <= code <= 122:
        return code - 97
    if 48 <= code <= 57:
        return code - 48 + 26
    return 36 + (code % 4)


def _length_bucket(text: str) -> int:
    for index, bound in enumerate(LENGTH_BUCKETS):
        if len(text) <= bound:
            return index
    return len(LENGTH_BUCKETS) - 1


@dataclass
class SchemaIndex:
    """Surface-form lookups for one schema, built once per example."""

    field_kind: dict[str, str]
    enum_owner: dict[str, set[str]]

    @classmethod
    def build(cls, active: Schema) -> "SchemaIndex":
        field_kind: dict[str, str] = {}
        enum_owner: dict[str, set[str]] = {}
        for item in active.fields:
            for surface in item.surface_forms():
                field_kind[surface.lower()] = item.kind
            for value in item.values:
                enum_owner.setdefault(value.lower(), set()).add(item.name)
        return cls(field_kind=field_kind, enum_owner=enum_owner)


def rows_for(
    tokens: list[str], active: Schema, use_word_identity: bool = True
) -> list[list[int]]:
    """One padded row list per token."""
    index = SchemaIndex.build(active)
    entries = match.build_index(active)
    lowered = [token.lower() for token in tokens]
    resolved = [match.resolve(token, active, entries) for token in lowered]

    # Which field, if any, most recently appeared before each token. Canonical
    # names, so "assngnee" anchors the values after it exactly as "assignee" would.
    preceding_field: list[str | None] = []
    distance_to_field: list[int] = []
    current: str | None = None
    distance = 99
    for position in range(len(lowered)):
        preceding_field.append(current)
        distance_to_field.append(distance)
        if resolved[position] is not None:
            current = resolved[position].field
            distance = 0
        elif distance < 99:
            distance += 1

    output: list[list[int]] = []
    for position, token in enumerate(tokens):
        folded = lowered[position]
        rows: list[int] = []

        def add(block: str, value: int) -> None:
            if block in IDENTITY_BLOCKS and not use_word_identity:
                return
            rows.append(OFFSET[block] + value)

        if folded.replace(".", "").isdigit():
            kind = 1
        elif folded[0].isalpha() or folded[0] == "_":
            kind = 0
        else:
            kind = 2
        add("kind", kind)
        add("length", _length_bucket(token))
        add("first_char", _character_class(token[0]))
        add("last_char", _character_class(token[-1]))
        add("word_hash", _hash(folded) & 255)
        stripped = "".join(c for c in folded if c not in "aeiou") or folded
        add("devowel_hash", _hash(stripped) & 127)

        if token[:1].isupper():
            add("flags", 0)
        if any(c.isdigit() for c in token):
            add("flags", 1)
        if "-" in token or "_" in token:
            add("flags", 2)

        found = resolved[position]
        add("field_match", KIND_OF_FIELD[found.kind] if found else 0)
        add("match_quality", found.quality + 1 if found else 0)

        owners = index.enum_owner.get(folded)
        if owners:
            add("enum_any", 0)
            # Does this enum value belong to the field just named? This row is
            # what disambiguates a value two fields share.
            if preceding_field[position] in owners:
                add("enum_preceding", 0)

        gap = distance_to_field[position]
        add("distance", DISTANCE_ROWS - 1 if gap == 99 else min(gap, DISTANCE_ROWS - 2))
        add("operator", OPERATOR_INDEX.get(folded, len(OPERATOR_WORDS)))

        span = max(1, len(tokens) - 1)
        add("position", min(3, int(4 * position / (span + 1))))

        rows = rows[:SLOTS] + [PADDING_ROW] * max(0, SLOTS - len(rows))
        output.append(rows)
    return output


if __name__ == "__main__":
    import random

    import generate
    import schema as schema_module

    schema_module.assert_disjoint()
    print(f"FEATURE_ROWS = {FEATURE_ROWS}  (padding row {PADDING_ROW})")
    for name, size in _blocks:
        print(f"  {name:<16} {OFFSET[name]:>4} .. {OFFSET[name] + size - 1:>4}")
    estimated = FEATURE_ROWS * 32 + 14393
    print(f"\nparameters at HIDDEN=32: {FEATURE_ROWS} * 32 + 14393 = {estimated:,}")

    example = generate.dataset("eval", 1, seed=3)[0]
    print(f"\nexample: {example.text}")
    table = rows_for(example.tokens, example.schema)
    for token, label, row in zip(example.tokens, example.labels, table):
        active = [value for value in row if value != PADDING_ROW]
        named = []
        for value in active:
            for name, size in _blocks:
                if OFFSET[name] <= value < OFFSET[name] + size:
                    named.append(f"{name}+{value - OFFSET[name]}")
                    break
        print(f"  {token:<14} {generate.LABELS[label]:<12} {' '.join(named)}")
