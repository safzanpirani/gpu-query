"""Resolve a token to a schema field, tolerantly.

Exact surface matching makes the app author enumerate every phrasing a user
might type, which is the hand-written grammar these projects exist to kill. This
module widens the match to inflections, prefixes, and single-character typos,
and reports *how* it matched so the model can weigh a shaky match differently
from a certain one.

Featurization and the compiler both call `resolve`. They must agree, or the
model will be scored against a field the compiler never picks.
"""

from __future__ import annotations

from dataclasses import dataclass

from schema import Schema

EXACT = 0
STEM = 1
PREFIX = 2
TYPO = 3
QUALITY_COUNT = 4

MIN_PREFIX = 4
MIN_TYPO_LENGTH = 5


def stem(word: str) -> str:
    for suffix in ("ies", "es", "s", "ed", "ing"):
        if len(word) - len(suffix) >= 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def within_one_edit(left: str, right: str) -> bool:
    """True when one substitution, insertion, or deletion separates the words."""
    if abs(len(left) - len(right)) > 1:
        return False
    if left == right:
        return False
    if len(left) == len(right):
        differences = sum(1 for a, b in zip(left, right) if a != b)
        return differences == 1
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    index = 0
    while index < len(shorter) and shorter[index] == longer[index]:
        index += 1
    return shorter[index:] == longer[index + 1 :]


@dataclass(frozen=True)
class Match:
    field: str
    kind: str
    quality: int


def build_index(active: Schema) -> list[tuple[str, str, str]]:
    """(surface, field name, kind) for every declared form."""
    entries = []
    for item in active.fields:
        for surface in item.surface_forms():
            entries.append((surface.lower(), item.name, item.kind))
    return entries


def resolve(token: str, active: Schema, index=None) -> Match | None:
    """Best field match for one token, or None.

    Quality order is strict: an exact match always beats a stem match, and a
    tier only applies when no better tier produced a unique answer.
    """
    folded = token.lower()
    entries = index if index is not None else build_index(active)

    for surface, name, kind in entries:
        if folded == surface:
            return Match(name, kind, EXACT)

    folded_stem = stem(folded)
    for surface, name, kind in entries:
        if folded_stem == stem(surface):
            return Match(name, kind, STEM)

    if len(folded) >= MIN_PREFIX:
        hits = {
            (name, kind)
            for surface, name, kind in entries
            if len(surface) >= MIN_PREFIX and surface.startswith(folded)
        }
        if len(hits) == 1:
            name, kind = hits.pop()
            return Match(name, kind, PREFIX)

    if len(folded) >= MIN_TYPO_LENGTH:
        hits = {
            (name, kind)
            for surface, name, kind in entries
            if within_one_edit(folded, surface)
        }
        if len(hits) == 1:
            name, kind = hits.pop()
            return Match(name, kind, TYPO)

    return None


def enum_owners(active: Schema) -> dict[str, set[str]]:
    owners: dict[str, set[str]] = {}
    for item in active.fields:
        for value in item.values:
            owners.setdefault(value.lower(), set()).add(item.name)
    return owners


if __name__ == "__main__":
    import random

    import schema as schema_module

    schema_module.assert_disjoint()
    active = schema_module.sample("eval", random.Random(4))
    print(f"domain={active.domain}")
    for item in active.fields:
        print(f"  {'/'.join(item.surface_forms()):<28} {item.kind} {item.values}")
    print()
    probe = active.fields[0]
    word = probe.name
    tests = [
        word,
        word + "s",
        word[: max(MIN_PREFIX, len(word) - 2)],
        word[:-1] + ("x" if word[-1] != "x" else "y"),
        "completelyunrelated",
    ]
    for candidate in tests:
        found = resolve(candidate, active)
        label = ["exact", "stem", "prefix", "typo"][found.quality] if found else "—"
        print(f"  {candidate:<24} -> {found.field if found else 'no match':<16} {label}")
