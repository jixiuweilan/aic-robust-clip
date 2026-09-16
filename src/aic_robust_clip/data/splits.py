"""Deterministic exact-duplicate grouping and grouped stratified splits."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..contracts import PARTITIONS, ContractError, SampleRecord, SplitRecord, canonical_json, sha256_json, write_json


ALGORITHM_VERSION = "grouped-observed-label-stratified-v2"


class SplitError(ValueError):
    """Raised when a split cannot satisfy stage and lineage constraints."""


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        left, right = self.find(first), self.find(second)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def group_exact_duplicates(records: Iterable[SampleRecord]) -> dict[str, str]:
    """Group exact byte/pixel duplicates without using test data.

    Pixel hashes can join files with different encodings, while byte hashes
    catch exact copies without a decoder.  Unioning both relations also makes
    transitive duplicate groups explicit.
    """

    records = list(records)
    if any(record.role != "train" for record in records):
        raise SplitError("duplicate grouping accepts training records only")
    if len({record.stage for record in records}) > 1:
        raise SplitError("duplicate grouping cannot mix stages")
    ids = [record.sample_id for record in records]
    if len(ids) != len(set(ids)):
        raise SplitError("sample IDs must be unique")
    union_find = _UnionFind(ids)
    for field in ("byte_sha256", "pixel_sha256"):
        by_digest: dict[str, list[str]] = defaultdict(list)
        for record in records:
            digest = getattr(record, field)
            if digest:
                by_digest[digest].append(record.sample_id)
        for members in by_digest.values():
            for member in members[1:]:
                union_find.union(members[0], member)
    by_root: dict[str, list[str]] = defaultdict(list)
    for sample_id in ids:
        by_root[union_find.find(sample_id)].append(sample_id)
    groups: dict[str, str] = {}
    for members in by_root.values():
        group_id = "g-" + hashlib.sha256("\0".join(sorted(members)).encode("utf-8")).hexdigest()[:24]
        for sample_id in members:
            groups[sample_id] = group_id
    return groups


@dataclass
class SplitManifest:
    records: list[SplitRecord]
    parent_manifest_digest: str
    algorithm_version: str = ALGORITHM_VERSION
    schema_version: str = "1.0"

    @property
    def digest(self) -> str:
        return sha256_json([record.to_dict() for record in sorted(self.records, key=lambda item: item.sample_id)])

    def counts(self) -> dict[str, int]:
        return dict(Counter(record.partition for record in self.records))

    def report(self) -> dict[str, Any]:
        by_class: dict[str, Counter[str]] = defaultdict(Counter)
        by_group: dict[str, set[str]] = defaultdict(set)
        for record in self.records:
            by_class[record.class_id or "<none>"][record.partition] += 1
            if record.class_id is not None:
                by_group[record.group_id].add(record.class_id)
        return {
            "schema_version": self.schema_version,
            "algorithm_version": self.algorithm_version,
            "parent_manifest_digest": self.parent_manifest_digest,
            "split_digest": self.digest,
            "counts": self.counts(),
            "per_class": {key: dict(sorted(value.items())) for key, value in sorted(by_class.items())},
            "conflicting_label_groups": {
                group_id: sorted(class_ids)
                for group_id, class_ids in sorted(by_group.items())
                if len(class_ids) > 1
            },
            "label_quality": "noisy_proxy",
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm_version": self.algorithm_version,
            "parent_manifest_digest": self.parent_manifest_digest,
            "split_digest": self.digest,
            "records": [record.to_dict() for record in sorted(self.records, key=lambda item: item.sample_id)],
            "report": self.report(),
        }


def _largest_remainder(total: int, ratios: tuple[float, float, float]) -> list[int]:
    raw = [total * ratio for ratio in ratios]
    result = [int(value) for value in raw]
    for index in sorted(range(3), key=lambda item: (-(raw[item] - result[item]), item))[: total - sum(result)]:
        result[index] += 1
    return result


def _target_counts(records: list[SampleRecord], ratios: tuple[float, float, float]) -> dict[str, list[int]]:
    by_class = Counter(record.class_id for record in records)
    return {class_id: _largest_remainder(count, ratios) for class_id, count in sorted(by_class.items()) if class_id is not None}


def _group_features(records: list[SampleRecord], groups: dict[str, str]) -> dict[str, list[SampleRecord]]:
    result: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        result[groups[record.sample_id]].append(record)
    return result


def make_grouped_split(
    records: Iterable[SampleRecord],
    *,
    seed: int = 17,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> SplitManifest:
    """Create a reproducible 80/10/10 split grouped by exact duplicates."""

    records = sorted(records, key=lambda item: item.sample_id)
    if not records:
        raise SplitError("cannot split an empty manifest")
    if any(record.role != "train" for record in records):
        raise SplitError("split construction accepts current-stage training records only")
    stages = {record.stage for record in records}
    if len(stages) != 1:
        raise SplitError("split construction cannot mix stages")
    if any(record.class_id is None for record in records):
        raise SplitError("all training records need observed class IDs")
    if len(set(ratios)) == 0 or len(ratios) != 3 or any(r <= 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise SplitError("ratios must be three positive values summing to one")
    if seed < 0:
        raise SplitError("seed must be non-negative")

    parent_digest = sha256_json([record.to_dict() for record in records])
    groups = group_exact_duplicates(records)
    grouped = _group_features(records, groups)
    targets = _target_counts(records, ratios)
    current: dict[str, Counter[str]] = {partition: Counter() for partition in PARTITIONS}
    assigned: dict[str, str] = {}
    # Large groups first reduces the chance that a duplicate group forces a
    # visibly poor tail split.  Hash ties make the result seed-dependent but
    # independent of ZIP/filesystem discovery order.
    group_order = sorted(
        grouped,
        key=lambda group_id: (
            -len(grouped[group_id]),
            hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest(),
            group_id,
        ),
    )
    partition_order = list(PARTITIONS)
    for group_id in group_order:
        members = grouped[group_id]
        contribution = Counter(record.class_id for record in members)
        scored: list[tuple[float, str]] = []
        for partition_index, partition in enumerate(partition_order):
            score = 0.0
            for class_id, amount in contribution.items():
                target = targets[class_id][partition_index]
                before = current[partition][class_id]
                # Compare changes to the SAME global objective, not absolute
                # residuals of differently sized partitions. Squared deficits
                # favour training support when classes/groups are scarce.
                score += (before + amount - target) ** 2 - (before - target) ** 2
            total_target = len(records) * ratios[partition_index]
            before_total = sum(current[partition].values())
            score += ((before_total + len(members) - total_target) ** 2
                      - (before_total - total_target) ** 2) * 0.05 / len(records)
            tie = int(hashlib.sha256(f"{seed}:{group_id}:{partition}".encode()).hexdigest()[:8], 16) / 2**32
            scored.append((score + tie * 1e-6, partition))
        assigned[group_id] = min(scored)[1]
        current[assigned[group_id]].update(contribution)

    split_records = [
        SplitRecord(
            sample_id=record.sample_id,
            group_id=groups[record.sample_id],
            partition=assigned[groups[record.sample_id]],
            seed=seed,
            algorithm_version=ALGORITHM_VERSION,
            parent_manifest_digest=parent_digest,
            stage=record.stage,
            role=record.role,
            class_id=record.class_id,
        )
        for record in records
    ]
    return SplitManifest(split_records, parent_digest)


def write_split_manifest(path: Path | str, manifest: SplitManifest) -> None:
    write_json(path, manifest.to_dict())


def load_split_manifest(path: Path | str, *, require_parent_digest: str | None = None) -> SplitManifest:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    records = []
    for item in value.get("records", []):
        item = dict(item)
        item.pop("schema_version", None)
        records.append(SplitRecord(**item))
    manifest = SplitManifest(
        records=records,
        parent_manifest_digest=value["parent_manifest_digest"],
        algorithm_version=value.get("algorithm_version", ALGORITHM_VERSION),
    )
    if value.get("split_digest") != manifest.digest:
        raise SplitError("split digest does not match its records")
    if require_parent_digest is not None and manifest.parent_manifest_digest != require_parent_digest:
        raise SplitError("split does not belong to the requested parent manifest")
    return manifest


def assert_no_group_crossing(manifest: SplitManifest) -> None:
    partitions: dict[str, set[str]] = defaultdict(set)
    for record in manifest.records:
        partitions[record.group_id].add(record.partition)
    crossed = {group_id: values for group_id, values in partitions.items() if len(values) != 1}
    if crossed:
        raise SplitError(f"duplicate groups cross partitions: {sorted(crossed)[:3]}")
