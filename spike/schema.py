"""Random application schemas for the gpu-query generalization spike.

A schema describes the filterable surface of one app: an issue tracker, a mail
client, a file browser. The tagger never sees a field name. It sees only that a
token matched some field of some type, so the spike must prove that structure
alone carries the parse.

Vocabulary banks are disjoint by design. `TRAIN_DOMAINS` and `EVAL_DOMAINS`
share no field word and no enum word, so an evaluation schema is unfamiliar in
exactly the way a third-party app would be.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field as dataclass_field

BOOL = "bool"
ENUM = "enum"
DATE = "date"
NUMBER = "number"
TEXT = "text"

FIELD_TYPES = (BOOL, ENUM, DATE, NUMBER, TEXT)


@dataclass
class Field:
    name: str
    kind: str
    aliases: list[str]
    values: list[str] = dataclass_field(default_factory=list)

    def surface_forms(self) -> list[str]:
        return [self.name, *self.aliases]


@dataclass
class Schema:
    domain: str
    fields: list[Field]

    def field_by_name(self, name: str) -> Field:
        return next(item for item in self.fields if item.name == name)

    def of_kind(self, kind: str) -> list[Field]:
        return [item for item in self.fields if item.kind == kind]


# Each domain supplies candidate fields as (name, kind, aliases, values).
# Training and evaluation banks are vocabulary-disjoint.
TRAIN_DOMAINS: dict[str, list[tuple]] = {
    "issues": [
        ("assignee", TEXT, ["owner", "assigned"], []),
        ("status", ENUM, ["state"], ["open", "closed", "triage", "blocked"]),
        ("priority", ENUM, ["severity"], ["urgent", "high", "low", "trivial"]),
        ("label", TEXT, ["tag"], []),
        ("milestone", TEXT, ["release"], []),
        ("created", DATE, ["opened", "filed"], []),
        ("comments", NUMBER, ["replies"], []),
        ("draft", BOOL, ["unfinished"], []),
        ("locked", BOOL, ["frozen"], []),
    ],
    "mail": [
        ("sender", TEXT, ["from"], []),
        ("recipient", TEXT, ["to"], []),
        ("subject", TEXT, ["title"], []),
        ("folder", ENUM, ["mailbox"], ["inbox", "archive", "spam", "sent"]),
        ("unread", BOOL, ["new"], []),
        ("starred", BOOL, ["flagged"], []),
        ("attachment", BOOL, ["attached"], []),
        ("received", DATE, ["arrived"], []),
        ("size", NUMBER, ["bytes"], []),
    ],
    "files": [
        ("filename", TEXT, ["name"], []),
        ("extension", ENUM, ["format"], ["pdf", "png", "csv", "mp4"]),
        ("folder", TEXT, ["directory"], []),
        ("modified", DATE, ["changed", "touched"], []),
        ("bytes", NUMBER, ["filesize"], []),
        ("shared", BOOL, ["public"], []),
        ("starred", BOOL, ["pinned"], []),
        ("author", TEXT, ["creator"], []),
    ],
    "commits": [
        ("author", TEXT, ["committer"], []),
        ("branch", TEXT, ["ref"], []),
        ("message", TEXT, ["summary"], []),
        ("verified", BOOL, ["signed"], []),
        ("merged", BOOL, ["landed"], []),
        ("committed", DATE, ["pushed"], []),
        ("additions", NUMBER, ["insertions"], []),
    ],
}

EVAL_DOMAINS: dict[str, list[tuple]] = {
    "contacts": [
        ("company", TEXT, ["employer", "org"], []),
        ("stage", ENUM, ["pipeline"], ["prospect", "qualified", "won", "churned"]),
        ("region", ENUM, ["territory"], ["emea", "apac", "latam", "nordics"]),
        ("phone", TEXT, ["mobile"], []),
        ("contacted", DATE, ["reached"], []),
        ("revenue", NUMBER, ["value"], []),
        ("subscribed", BOOL, ["opted"], []),
        ("confirmed", BOOL, ["screened"], []),
    ],
    "music": [
        ("artist", TEXT, ["performer", "band"], []),
        ("album", TEXT, ["record"], []),
        ("genre", ENUM, ["style"], ["ambient", "techno", "jazz", "shoegaze"]),
        ("duration", NUMBER, ["length"], []),
        ("released", DATE, ["issued"], []),
        ("explicit", BOOL, ["uncensored"], []),
        ("downloaded", BOOL, ["offline"], []),
        ("plays", NUMBER, ["listens"], []),
    ],
    "recipes": [
        ("cuisine", ENUM, ["origin"], ["thai", "basque", "creole", "sichuan"]),
        ("difficulty", ENUM, ["effort"], ["easy", "moderate", "hard"]),
        ("ingredient", TEXT, ["component"], []),
        ("calories", NUMBER, ["energy"], []),
        ("cooked", DATE, ["prepared"], []),
        ("vegetarian", BOOL, ["meatless"], []),
        ("bookmarked", BOOL, ["saved"], []),
    ],
    "shipments": [
        ("carrier", ENUM, ["courier"], ["maersk", "hapag", "evergreen"]),
        ("origin", TEXT, ["departure"], []),
        ("destination", TEXT, ["arrival"], []),
        ("weight", NUMBER, ["tonnage"], []),
        ("dispatched", DATE, ["departed"], []),
        ("insured", BOOL, ["covered"], []),
        ("delayed", BOOL, ["late"], []),
    ],
}


def build(candidates: list[tuple], domain: str, rng: random.Random) -> Schema:
    """Pick a plausible subset of one domain's fields."""
    count = rng.randint(4, min(8, len(candidates)))
    chosen = rng.sample(candidates, count)
    fields = []
    for name, kind, aliases, values in chosen:
        # Drop aliases sometimes so the model cannot assume every field has one.
        kept = [alias for alias in aliases if rng.random() < 0.75]
        fields.append(Field(name=name, kind=kind, aliases=kept, values=list(values)))

    # Sometimes give two enum fields a value in common. A shared value cannot be
    # resolved by lookup alone, so only the preceding field token disambiguates
    # it. Without this the enum-ownership feature is never actually tested.
    enums = [item for item in fields if item.kind == ENUM and item.values]
    if len(enums) >= 2 and rng.random() < 0.5:
        donor, receiver = rng.sample(enums, 2)
        shared = rng.choice(donor.values)
        if shared not in receiver.values:
            receiver.values.append(shared)

    return Schema(domain=domain, fields=fields)


def sample(split: str, rng: random.Random) -> Schema:
    domains = TRAIN_DOMAINS if split == "train" else EVAL_DOMAINS
    domain = rng.choice(sorted(domains))
    return build(domains[domain], domain, rng)


def vocabulary(domains: dict[str, list[tuple]]) -> set[str]:
    words = set()
    for candidates in domains.values():
        for name, _kind, aliases, values in candidates:
            words.update([name, *aliases, *values])
    return words


def assert_disjoint() -> None:
    """Guard the whole premise of the spike."""
    overlap = vocabulary(TRAIN_DOMAINS) & vocabulary(EVAL_DOMAINS)
    if overlap:
        raise AssertionError(f"train/eval vocabulary overlaps: {sorted(overlap)}")


if __name__ == "__main__":
    assert_disjoint()
    generator = random.Random(0)
    for split in ("train", "eval"):
        schema = sample(split, generator)
        print(f"[{split}] domain={schema.domain}")
        for item in schema.fields:
            surface = "/".join(item.surface_forms())
            values = f" values={item.values}" if item.values else ""
            print(f"  {surface:<32} {item.kind}{values}")
