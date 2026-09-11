"""Render natural-language queries from a schema, with labels and a target AST.

Labels come from the renderer's own structure, never from a parser, so the
model is never trained on its own output. Each example carries the schema that
produced it, because featurization needs the schema to emit match features and
the compiler needs it to resolve a slot back to a real field.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import match
import schema as schema_module
from schema import BOOL, DATE, ENUM, NUMBER, TEXT, Field, Schema

LABELS = [
    "O",
    "FIELD",
    "FIELD_CONT",
    "OP_EQ",
    "OP_LT",
    "OP_GT",
    "OP_CONTAINS",
    "VALUE",
    "VALUE_CONT",
    "NEG",
    "AND",
    "OR",
]
LABEL_ID = {name: index for index, name in enumerate(LABELS)}
LABEL_COUNT = len(LABELS)

TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_'-]*|\d+(?:\.\d+)?|[^\sA-Za-z0-9]")

# Carrier prose. The model must learn to ignore it without a stopword list.
PREFIXES = [
    "",
    "show me",
    "find",
    "find all",
    "list",
    "give me",
    "search for",
    "everything",
    "anything",
    "show",
]
SUFFIXES = ["", "please", "thanks"]

PEOPLE = ["sarah", "arik", "shu", "meera", "tomas", "nadia", "ken", "priya"]
FREE_TEXT = [
    "quarterly report",
    "budget",
    "onboarding",
    "roadmap review",
    "invoice",
    "design spec",
    "postmortem",
]
RELATIVE_DATES = [
    ("today", 1),
    ("yesterday", 1),
    ("last week", 2),
    ("this month", 2),
    ("last month", 2),
    ("this quarter", 2),
    ("last friday", 2),
]

LESS_WORDS = ["under", "below", "less than", "fewer than", "at most"]
MORE_WORDS = ["over", "above", "more than", "greater than", "at least"]
BEFORE_WORDS = ["before", "prior to", "earlier than"]
AFTER_WORDS = ["after", "since", "later than"]
NEGATIONS = ["not", "no", "without", "excluding"]
CONTAINS_WORDS = ["containing", "mentioning", "matching", "with"]


@dataclass
class Example:
    tokens: list[str]
    labels: list[int]
    boundaries: list[int]
    ast: list[dict]
    schema: Schema
    text: str


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def emit(pieces: list[tuple[str, str]]) -> tuple[list[str], list[int]]:
    """Expand (phrase, label) pairs into aligned token and label lists.

    The first token of a phrase takes the label; the rest take its
    continuation form when one exists, otherwise the same label.
    """
    tokens: list[str] = []
    labels: list[int] = []
    for phrase, label in pieces:
        parts = tokenize(phrase)
        if not parts:
            continue
        continuation = f"{label}_CONT"
        follow = continuation if continuation in LABEL_ID else label
        for index, part in enumerate(parts):
            tokens.append(part)
            labels.append(LABEL_ID[label if index == 0 else follow])
    return tokens, labels


def corrupt(word: str, rng: random.Random) -> str:
    style = rng.random()
    if style < 0.45:
        return word + ("es" if word.endswith(("s", "x", "ch")) else "s")
    if style < 0.75 and len(word) > match.MIN_PREFIX + 1:
        return word[: len(word) - rng.randint(1, 2)]
    if len(word) >= match.MIN_TYPO_LENGTH:
        position = rng.randrange(1, len(word))
        replacement = rng.choice("abcdefghijklmnopqrstuvwxyz")
        if replacement != word[position]:
            return word[:position] + replacement + word[position + 1 :]
    return word


def field_surface(
    field: Field, rng: random.Random, active: Schema | None = None
) -> str:
    """A surface form, sometimes inflected or mistyped.

    Only returns a corruption that still resolves to this same field, so the
    label stays truthful. Exact matching alone would force the app author to
    enumerate every phrasing, which is the grammar problem these projects exist
    to remove.
    """
    word = rng.choice(field.surface_forms())
    if active is None or rng.random() >= 0.30:
        return word
    candidate = corrupt(word, rng)
    found = match.resolve(candidate, active)
    if found is not None and found.field == field.name:
        return candidate
    return word


def other_field_words(active: Schema, exclude: Field) -> list[str]:
    """Surface forms belonging to some *other* field in the same schema.

    Used as literal text values. A token that matches a field but is really a
    value is the case lookup cannot solve, so it is the case worth generating.
    """
    words = []
    for item in active.fields:
        if item.name != exclude.name:
            words.extend(item.surface_forms())
    return words


def render_bare_enum(
    field: Field, rng: random.Random, active: Schema
) -> tuple[list[tuple[str, str]] | None, dict | None]:
    """A value standing alone, its field implied: "open" meaning status:open.

    Only emitted when exactly one field owns the value. A value shared by two
    fields is genuinely ambiguous without its field token, and the generator
    must not produce a label the compiler cannot justify.
    """
    if field.kind != ENUM or not field.values or rng.random() >= 0.25:
        return None, None
    owners = match.enum_owners(active)
    unique = [value for value in field.values if len(owners.get(value.lower(), ())) == 1]
    if not unique:
        return None, None
    value = rng.choice(unique)
    negated = rng.random() < 0.15
    pieces: list[tuple[str, str]] = []
    if negated:
        pieces.append((rng.choice(NEGATIONS), "NEG"))
    pieces.append((value, "VALUE"))
    return pieces, {"field": field.name, "cmp": "eq", "value": value, "neg": negated}


def render_clause(
    field: Field, rng: random.Random, active: Schema | None = None
) -> tuple[list[tuple[str, str]], dict]:
    """One filter clause as (phrase, label) pairs plus its AST node."""
    negated = rng.random() < 0.18
    pieces: list[tuple[str, str]] = []
    if negated:
        pieces.append((rng.choice(NEGATIONS), "NEG"))

    if field.kind == BOOL:
        # Bare flag: "unread", "not starred". The field word is the whole clause.
        pieces.append((field_surface(field, rng, active), "FIELD"))
        return pieces, {"field": field.name, "cmp": "is", "value": True, "neg": negated}

    if field.kind == ENUM:
        value = rng.choice(field.values)
        style = rng.random()
        if style < 0.35:
            pieces.append((field_surface(field, rng, active), "FIELD"))
            pieces.append((":", "OP_EQ"))
            pieces.append((value, "VALUE"))
        elif style < 0.7:
            pieces.append((field_surface(field, rng, active), "FIELD"))
            pieces.append((rng.choice(["is", "="]), "OP_EQ"))
            pieces.append((value, "VALUE"))
        else:
            # Bare enum value with the field implied by the schema.
            pieces.append((field_surface(field, rng, active), "FIELD"))
            pieces.append((value, "VALUE"))
        return pieces, {"field": field.name, "cmp": "eq", "value": value, "neg": negated}

    if field.kind == NUMBER:
        amount = str(rng.choice([1, 2, 3, 5, 10, 25, 100, 500]))
        roll = rng.random()
        if roll < 0.35:
            pieces.append((field_surface(field, rng, active), "FIELD"))
            pieces.append((rng.choice(LESS_WORDS), "OP_LT"))
            pieces.append((amount, "VALUE"))
            comparison = "lt"
        elif roll < 0.7:
            pieces.append((field_surface(field, rng, active), "FIELD"))
            pieces.append((rng.choice(MORE_WORDS), "OP_GT"))
            pieces.append((amount, "VALUE"))
            comparison = "gt"
        else:
            pieces.append((field_surface(field, rng, active), "FIELD"))
            pieces.append((rng.choice(["is", "="]), "OP_EQ"))
            pieces.append((amount, "VALUE"))
            comparison = "eq"
        return pieces, {
            "field": field.name,
            "cmp": comparison,
            "value": amount,
            "neg": negated,
        }

    if field.kind == DATE:
        phrase, _length = rng.choice(RELATIVE_DATES)
        roll = rng.random()
        if roll < 0.4:
            pieces.append((field_surface(field, rng, active), "FIELD"))
            pieces.append((rng.choice(BEFORE_WORDS), "OP_LT"))
            pieces.append((phrase, "VALUE"))
            comparison = "lt"
        elif roll < 0.8:
            pieces.append((field_surface(field, rng, active), "FIELD"))
            pieces.append((rng.choice(AFTER_WORDS), "OP_GT"))
            pieces.append((phrase, "VALUE"))
            comparison = "gt"
        else:
            pieces.append((field_surface(field, rng, active), "FIELD"))
            pieces.append((phrase, "VALUE"))
            comparison = "eq"
        return pieces, {
            "field": field.name,
            "cmp": comparison,
            "value": phrase,
            "neg": negated,
        }

    # TEXT
    roll_value = rng.random()
    if active is not None and roll_value < 0.22:
        # "subject containing status" — the value IS another field's name.
        candidates = other_field_words(active, field)
        value = rng.choice(candidates) if candidates else rng.choice(PEOPLE)
    elif roll_value < 0.6:
        value = rng.choice(PEOPLE)
    else:
        value = rng.choice(FREE_TEXT)
    roll = rng.random()
    if roll < 0.3:
        pieces.append((field_surface(field, rng, active), "FIELD"))
        pieces.append((":", "OP_EQ"))
        pieces.append((value, "VALUE"))
        comparison = "eq"
    elif roll < 0.6:
        pieces.append((field_surface(field, rng, active), "FIELD"))
        pieces.append((rng.choice(CONTAINS_WORDS), "OP_CONTAINS"))
        pieces.append((value, "VALUE"))
        comparison = "contains"
    else:
        pieces.append((field_surface(field, rng, active), "FIELD"))
        pieces.append((value, "VALUE"))
        comparison = "eq"
    return pieces, {"field": field.name, "cmp": comparison, "value": value, "neg": negated}


def render(active_schema: Schema, rng: random.Random) -> Example:
    clause_count = rng.choices([1, 2, 3], weights=[0.45, 0.4, 0.15])[0]
    available = list(active_schema.fields)
    rng.shuffle(available)
    chosen = available[:clause_count]

    pieces: list[tuple[str, str]] = []
    prefix = rng.choice(PREFIXES)
    if prefix:
        pieces.append((prefix, "O"))

    ast: list[dict] = []
    clause_starts: list[int] = []
    joiner = "AND" if rng.random() < 0.82 else "OR"
    for index, field in enumerate(chosen):
        if index:
            word = "and" if joiner == "AND" else "or"
            pieces.append((word, joiner))
        # Record where this clause begins, in tokens, before extending.
        clause_pieces, node = render_bare_enum(field, rng, active_schema)
        if clause_pieces is None:
            clause_pieces, node = render_clause(field, rng, active_schema)
        start = sum(len(tokenize(text)) for text, _ in pieces)
        clause_starts.append(start)
        pieces.extend(clause_pieces)
        ast.append(node)

    suffix = rng.choice(SUFFIXES)
    if suffix:
        pieces.append((suffix, "O"))

    tokens, labels = emit(pieces)
    boundaries = [0] * len(tokens)
    for start in clause_starts:
        if start < len(boundaries):
            boundaries[start] = 1

    return Example(
        tokens=tokens,
        labels=labels,
        boundaries=boundaries,
        ast=ast,
        schema=active_schema,
        text=" ".join(tokens),
    )


def dataset(split: str, count: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    examples = []
    for _ in range(count):
        active = schema_module.sample(split, rng)
        examples.append(render(active, rng))
    return examples


if __name__ == "__main__":
    schema_module.assert_disjoint()
    for split in ("train", "eval"):
        print(f"===== {split} =====")
        for example in dataset(split, 6, seed=7):
            print(f"  {example.text}")
            pairs = " ".join(
                f"{token}/{LABELS[label]}"
                for token, label in zip(example.tokens, example.labels)
            )
            print(f"    {pairs}")
            print(f"    ast={example.ast}")
