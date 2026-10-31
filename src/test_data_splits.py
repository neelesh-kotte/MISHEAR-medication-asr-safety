"""Leakage-resistant dataset partitions for MISHEAR's two shift evaluations.

Despite the filename, this is a reusable implementation module. It keeps
active-ingredient connected components together for H1 and provides a separate
condition-held-out split for H2. It intentionally has no dependency on sklearn.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_FRACTIONS = {"train": 0.60, "validation": 0.20, "test": 0.20}


def _values(record: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    out: list[str] = []
    for key in keys:
        value = record.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, str):
            if key.endswith("_ids") or "," in value or ";" in value:
                chunks = value.replace(";", ",").split(",")
            else:
                chunks = [value]
        elif isinstance(value, (list, tuple, set)):
            chunks = list(value)
        else:
            chunks = [value]
        out.extend(str(chunk).strip() for chunk in chunks if str(chunk).strip())
    return list(dict.fromkeys(out))


def ingredient_ids(record: Mapping[str, Any]) -> list[str]:
    """Return ingredient IDs from normalized recording or pair metadata."""
    return _values(record, (
        "ingredient_ids", "ingredient_id", "ingredient_rxcui", "active_ingredient_id",
        "ingredient_a", "ingredient_b", "ingredient_a_rxcui", "ingredient_b_rxcui",
    ))


class _DisjointSet:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def add(self, value: str) -> None:
        self.parent.setdefault(value, value)

    def find(self, value: str) -> str:
        self.add(value)
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            low, high = sorted((a, b))
            self.parent[high] = low


def build_ingredient_component_map(
    records: Sequence[Mapping[str, Any]],
    pair_edges: Iterable[Mapping[str, Any]] = (),
) -> dict[str, str]:
    """Map each active ingredient to its connected component in the pair graph."""
    dsu = _DisjointSet()
    for row in records:
        ids = ingredient_ids(row)
        if not ids:
            raise ValueError("Ingredient-disjoint splitting requires ingredient IDs on every row.")
        for identifier in ids:
            dsu.add(identifier)
        for identifier in ids[1:]:
            dsu.union(ids[0], identifier)
    for edge in pair_edges:
        ids = ingredient_ids(edge)
        if len(ids) < 2:
            raise ValueError(f"Pair edge must contain two ingredient IDs: {edge!r}")
        for identifier in ids:
            dsu.add(identifier)
        for identifier in ids[1:]:
            dsu.union(ids[0], identifier)
    components: dict[str, list[str]] = defaultdict(list)
    for identifier in dsu.parent:
        components[dsu.find(identifier)].append(identifier)
    component_label = {member: min(members) for members in components.values() for member in members}
    return component_label


def _normalized_fractions(fractions: Mapping[str, float]) -> dict[str, float]:
    required = ("train", "validation", "test")
    if set(fractions) != set(required):
        raise ValueError(f"fractions must have exactly these keys: {required}")
    if any(value < 0 for value in fractions.values()) or sum(fractions.values()) <= 0:
        raise ValueError("fractions must be non-negative and have a positive sum")
    total = sum(fractions.values())
    return {name: fractions[name] / total for name in required}


def _assign_groups(group_sizes: Mapping[str, int], fractions: Mapping[str, float], seed: int) -> dict[str, str]:
    """Greedily balance whole groups against row-count targets, with seeded ties."""
    rng = random.Random(seed)
    names = list(group_sizes)
    rng.shuffle(names)
    names.sort(key=lambda name: group_sizes[name], reverse=True)
    partitions = ("train", "validation", "test")
    total = sum(group_sizes.values())
    targets = {part: fractions[part] * total for part in partitions}
    current = {part: 0 for part in partitions}
    assignment: dict[str, str] = {}
    for group in names:
        size = group_sizes[group]
        chosen = min(
            partitions,
            key=lambda part: (
                sum(((current[other] + (size if other == part else 0) - targets[other]) / max(targets[other], 1.0)) ** 2
                    for other in partitions),
                partitions.index(part),
            ),
        )
        assignment[group] = chosen
        current[chosen] += size
    return assignment


def create_ingredient_disjoint_splits(
    records: Sequence[Mapping[str, Any]],
    pair_edges: Iterable[Mapping[str, Any]] = (),
    *,
    fractions: Mapping[str, float] = DEFAULT_FRACTIONS,
    seed: int = 20261030,
) -> dict[str, list[dict[str, Any]]]:
    """Create train/validation/test sets grouped by pair-graph components.

    A row that represents both endpoints (``ingredient_a`` and ``ingredient_b``)
    links those endpoints into one component. Otherwise pass the pair manifest as
    ``pair_edges`` so all connected medications stay on a single side of a split.
    Returned records are copied and tagged with ``split`` and ``component_id``.
    """
    if not records:
        raise ValueError("Cannot split an empty dataset.")
    fractions = _normalized_fractions(fractions)
    rows = [dict(row) for row in records]
    pair_edges = list(pair_edges)
    component_map = build_ingredient_component_map(rows, pair_edges)
    group_by_row: list[str] = []
    for index, row in enumerate(rows):
        ids = ingredient_ids(row)
        components = {component_map[identifier] for identifier in ids}
        if len(components) != 1:
            raise ValueError(f"A row spans multiple pair-graph components at row {index}: {sorted(components)}")
        group_by_row.append(next(iter(components)))
    group_sizes: dict[str, int] = defaultdict(int)
    for group in group_by_row:
        group_sizes[group] += 1
    group_assignment = _assign_groups(group_sizes, fractions, seed)
    split_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in ("train", "validation", "test")}
    for row, group in zip(rows, group_by_row):
        part = group_assignment[group]
        row["split"] = part
        row["component_id"] = group
        split_rows[part].append(row)
    validate_ingredient_disjoint_splits(split_rows, pair_edges)
    return split_rows


def _condition_id(record: Mapping[str, Any], condition_id_field: str, family_field: str) -> str:
    direct = record.get(condition_id_field) or record.get("acoustic_condition_id") or record.get("condition_id")
    if direct:
        return str(direct)
    family = record.get(family_field) or record.get("condition_family") or "clean"
    return str(family)


def create_unseen_acoustic_condition_split(
    records: Sequence[Mapping[str, Any]],
    *,
    held_out_condition_ids: Iterable[str],
    held_out_families: Iterable[str] = (),
    condition_id_field: str = "acoustic_condition_id",
    family_field: str = "condition_family",
    group_field: str = "ingredient_id",
    pair_edges: Iterable[Mapping[str, Any]] = (),
    validation_fraction: float = 0.2,
    seed: int = 20261030,
) -> dict[str, list[dict[str, Any]]]:
    """Reserve selected conditions for test; grouped-split remaining development.

    Condition-held-out evaluation deliberately allows the same medication
    component in development and test, because H2 tests unseen conditions.
    Development train/validation remain ingredient/pair-component grouped.
    Pass the complete clipping family through ``held_out_families`` and the
    exact harshest babble+reverb condition ID through ``held_out_condition_ids``.
    """
    if not records:
        raise ValueError("Cannot split an empty dataset.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    held_ids, held_families = set(held_out_condition_ids), set(held_out_families)
    pair_edges = list(pair_edges)
    if not held_ids and not held_families:
        raise ValueError("At least one held-out condition ID or family is required.")
    rows = [dict(row) for row in records]
    train_val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for row in rows:
        condition_id = _condition_id(row, condition_id_field, family_field)
        family = str(row.get(family_field) or row.get("condition_family") or "")
        if condition_id in held_ids or family in held_families:
            row["split"] = "test"
            test.append(row)
        else:
            train_val.append(row)
    if not test:
        raise ValueError("No records matched the held-out condition specification.")
    if not train_val:
        raise ValueError("No development-condition records remain for training/validation.")

    fractions = {"train": 1.0 - validation_fraction, "validation": validation_fraction, "test": 0.0}
    # Component grouping is preferred when medication IDs and pair edges exist.
    has_ingredient_ids = all(ingredient_ids(row) for row in train_val)
    if has_ingredient_ids:
        component_map = build_ingredient_component_map(train_val, pair_edges)
        groups = []
        for index, row in enumerate(train_val):
            components = {component_map[identifier] for identifier in ingredient_ids(row)}
            if len(components) != 1:
                raise ValueError(f"Development row spans multiple ingredient components: {index}")
            group = next(iter(components))
            row["component_id"] = group
            groups.append(group)
    else:
        groups = [str(row.get(group_field) or row.get("pair_id") or row.get("recording_id") or index)
                  for index, row in enumerate(train_val)]
    group_sizes: dict[str, int] = defaultdict(int)
    for group in groups:
        group_sizes[group] += 1
    dev_fractions = {"train": fractions["train"], "validation": fractions["validation"], "test": 0.0}
    # Assign only among the two development partitions.
    train_val_assignment = _assign_two_groups(group_sizes, dev_fractions, seed)
    output: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": test}
    for row, group in zip(train_val, groups):
        part = train_val_assignment[group]
        row["split"] = part
        output[part].append(row)
    validate_unseen_acoustic_condition_split(
        output,
        held_out_condition_ids=held_ids,
        held_out_families=held_families,
        condition_id_field=condition_id_field,
        family_field=family_field,
    )
    return output


def _assign_two_groups(group_sizes: Mapping[str, int], fractions: Mapping[str, float], seed: int) -> dict[str, str]:
    """Assign groups between train and validation while retaining whole groups."""
    rng = random.Random(seed)
    names = list(group_sizes)
    rng.shuffle(names)
    names.sort(key=lambda name: group_sizes[name], reverse=True)
    total = sum(group_sizes.values())
    targets = {"train": fractions["train"] * total, "validation": fractions["validation"] * total}
    current = {"train": 0, "validation": 0}
    assignment: dict[str, str] = {}
    for group in names:
        size = group_sizes[group]
        part = min(("train", "validation"), key=lambda key: (
            ((current["train"] + (size if key == "train" else 0) - targets["train"]) / max(targets["train"], 1)) ** 2
            + ((current["validation"] + (size if key == "validation" else 0) - targets["validation"]) / max(targets["validation"], 1)) ** 2,
            ("train", "validation").index(key),
        ))
        assignment[group] = part
        current[part] += size
    return assignment


def validate_ingredient_disjoint_splits(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    pair_edges: Iterable[Mapping[str, Any]] = (),
) -> None:
    """Raise if ingredient IDs or connected pair-graph components cross splits."""
    all_rows = [row for rows in splits.values() for row in rows]
    component_map = build_ingredient_component_map(all_rows, pair_edges) if all_rows else {}
    component_owner: dict[str, str] = {}
    ingredient_owner: dict[str, str] = {}
    seen_rows: dict[str, str] = {}
    for partition, rows in splits.items():
        for index, row in enumerate(rows):
            ids = ingredient_ids(row)
            if not ids:
                raise ValueError(f"Row in {partition} has no ingredient identifier.")
            for identifier in ids:
                previous = ingredient_owner.setdefault(identifier, partition)
                if previous != partition:
                    raise ValueError(f"Ingredient {identifier!r} appears in {previous} and {partition}.")
                component = component_map[identifier]
                previous_component = component_owner.setdefault(component, partition)
                if previous_component != partition:
                    raise ValueError(f"Pair-graph component {component!r} crosses {previous_component}/{partition}.")
            row_id = str(row.get("utterance_id") or row.get("recording_id") or row.get("audio_path") or f"{partition}:{index}")
            previous_row = seen_rows.setdefault(row_id, partition)
            if previous_row != partition:
                raise ValueError(f"Recording {row_id!r} occurs in multiple splits.")


def validate_unseen_acoustic_condition_split(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    held_out_condition_ids: Iterable[str],
    held_out_families: Iterable[str] = (),
    condition_id_field: str = "acoustic_condition_id",
    family_field: str = "condition_family",
) -> None:
    """Ensure reserved conditions occur only in the held-out test partition."""
    held_ids, held_families = set(held_out_condition_ids), set(held_out_families)
    if not splits.get("test"):
        raise ValueError("Held-out condition test partition is empty.")
    for part in ("train", "validation"):
        for row in splits.get(part, []):
            condition_id = _condition_id(row, condition_id_field, family_field)
            family = str(row.get(family_field) or row.get("condition_family") or "")
            if condition_id in held_ids or family in held_families:
                raise ValueError(f"Held-out condition leaked into {part}: {condition_id!r}, {family!r}")
    for row in splits["test"]:
        condition_id = _condition_id(row, condition_id_field, family_field)
        family = str(row.get(family_field) or row.get("condition_family") or "")
        if condition_id not in held_ids and family not in held_families:
            raise ValueError(f"Non-held-out condition appears in acoustic test set: {condition_id!r}")


def write_split_manifest(splits: Mapping[str, Sequence[Mapping[str, Any]]], output_path: str | Path) -> None:
    """Save the exact recording IDs in each frozen partition as JSON."""
    payload = {
        "splits": {
            name: [str(row.get("utterance_id") or row.get("recording_id") or row.get("audio_path")) for row in rows]
            for name, rows in splits.items()
        },
        "row_counts": {name: len(rows) for name, rows in splits.items()},
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
