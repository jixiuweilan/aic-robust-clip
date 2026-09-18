"""Explicit-manifest datasets with stage/role/partition access checks."""

from __future__ import annotations

import io
import hashlib
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..contracts import PARTITIONS, ContractError, SampleRecord, SplitRecord, sha256_json
from .relocation import resolve_archive


class DatasetError(ValueError):
    """Raised when a loader would violate stage or data-role boundaries."""


def identity_transform(value):
    """Pickle-safe default for spawn workers."""
    return value


@dataclass(frozen=True)
class SampleItem:
    image: Any
    sample_id: str
    class_id: str | None
    label_index: int | None


def _read_record_bytes(record: SampleRecord) -> bytes:
    archive_path = resolve_archive(Path(record.archive_path), archive_identity=record.archive_identity)
    if archive_path.is_dir():
        source = archive_path / record.member_path
        if not source.is_file():
            raise DatasetError(f"extracted image does not exist: {source}")
        return source.read_bytes()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            try:
                info = archive.getinfo(record.member_path)
            except KeyError as exc:
                raise DatasetError(f"manifest member is absent from archive: {record.member_path}") from exc
            with archive.open(info, "r") as handle:
                return handle.read()
    except (OSError, zipfile.BadZipFile) as exc:
        raise DatasetError(f"cannot read manifest archive {archive_path}: {exc}") from exc


class ManifestDataset:
    """A map-style dataset that never discovers images outside its records.

    ``purpose`` is intentionally explicit.  Routine training/scoring/history
    consumers can only see the training partition; dev, confirm and test are
    available only through their dedicated evaluation/inference purposes.
    """

    def __init__(
        self,
        records: Sequence[SampleRecord],
        *,
        stage: str,
        role: str,
        partition: str,
        purpose: str,
        class_to_index: dict[str, int] | None = None,
        transform: Callable[[bytes], Any] | None = None,
    ) -> None:
        if not records:
            raise DatasetError("dataset cannot be empty")
        if partition not in PARTITIONS:
            raise DatasetError(f"unsupported partition: {partition!r}")
        if purpose not in {"train", "scoring", "history", "dev", "confirm", "inference"}:
            raise DatasetError(f"unsupported loader purpose: {purpose!r}")
        if any(record.stage != stage for record in records):
            raise DatasetError("loader records mix competition stages")
        if any(record.role != role for record in records):
            raise DatasetError("loader records mix train and test roles")
        if role == "test" and class_to_index is not None:
            raise DatasetError("test inference records cannot be assigned fitted labels")
        if role == "train" and any(record.class_id is None for record in records):
            raise DatasetError("training loader records require observed labels")
        if purpose in {"train", "scoring", "history"} and (role != "train" or partition != "train"):
            raise DatasetError(f"{purpose} loader is restricted to the training partition")
        if purpose == "dev" and (role != "train" or partition != "dev"):
            raise DatasetError("dev loader requires the training role and dev partition")
        if purpose == "confirm" and (role != "train" or partition != "confirm"):
            raise DatasetError("confirm loader requires the training role and confirm partition")
        if purpose == "inference" and role != "test":
            raise DatasetError("inference loader requires test records")
        self.records = tuple(sorted(records, key=lambda record: record.sample_id))
        self.stage = stage
        self.role = role
        self.partition = partition
        self.purpose = purpose
        self.class_to_index = dict(class_to_index or {})
        self.transform = transform or identity_transform
        self._zip_handles: dict[str, zipfile.ZipFile] = {}
        self._pid = os.getpid()
        self.epoch = 0

    @classmethod
    def from_split(
        cls,
        records: Sequence[SampleRecord],
        split_records: Sequence[SplitRecord],
        *,
        stage: str,
        partition: str,
        purpose: str,
        class_to_index: dict[str, int],
        transform: Callable[[bytes], Any] | None = None,
    ) -> "ManifestDataset":
        if any(record.stage != stage or record.role != "train" for record in records):
            raise DatasetError("parent manifest mixes stages or roles")
        if len({record.sample_id for record in records}) != len(records):
            raise DatasetError("parent manifest contains duplicate sample IDs")
        parent_digest = sha256_json([record.to_dict() for record in sorted(records, key=lambda item: item.sample_id)])
        if any(item.parent_manifest_digest != parent_digest for item in split_records):
            raise DatasetError("split parent manifest digest mismatch")
        if any(item.stage != stage or item.role != "train" for item in split_records):
            raise DatasetError("split records mix stages or roles")
        if len({item.sample_id for item in split_records}) != len(split_records):
            raise DatasetError("split contains duplicate sample IDs across partitions")
        if {item.sample_id for item in split_records} != {record.sample_id for record in records}:
            raise DatasetError("split must cover the parent manifest exactly")
        group_partitions: dict[str, str] = {}
        for item in split_records:
            if group_partitions.setdefault(item.group_id, item.partition) != item.partition:
                raise DatasetError("duplicate group crosses split partitions")
        by_id = {record.sample_id: record for record in records}
        selected_items = [item for item in split_records if item.partition == partition and item.stage == stage]
        selected_ids = [item.sample_id for item in selected_items]
        if len(selected_ids) != len(set(selected_ids)):
            raise DatasetError("split contains duplicate sample IDs")
        missing = [sample_id for sample_id in selected_ids if sample_id not in by_id]
        if missing:
            raise DatasetError(f"split contains a sample absent from the parent manifest: {missing[0]}")
        chosen = [by_id[item.sample_id] for item in selected_items]
        if any(item.class_id != by_id[item.sample_id].class_id for item in selected_items):
            raise DatasetError("split label does not match the parent manifest")
        if not chosen:
            raise DatasetError(f"split partition is empty: {partition}")
        return cls(
            chosen,
            stage=stage,
            role="train",
            partition=partition,
            purpose=purpose,
            class_to_index=class_to_index,
            transform=transform,
        )

    def __len__(self) -> int:
        return len(self.records)

    def _ensure_process(self) -> None:
        current_pid = os.getpid()
        if current_pid != self._pid:
            self.close()
            self._pid = current_pid

    def _bytes_for(self, record: SampleRecord) -> bytes:
        # Opening one handle per archive per worker avoids sharing a ZipFile
        # object across DataLoader worker processes.
        self._ensure_process()
        archive_path = resolve_archive(Path(record.archive_path), archive_identity=record.archive_identity)
        if archive_path.is_dir():
            return _read_record_bytes(record)
        key = str(archive_path)
        archive = self._zip_handles.get(key)
        if archive is None:
            try:
                archive = zipfile.ZipFile(archive_path)
            except (OSError, zipfile.BadZipFile) as exc:
                raise DatasetError(f"cannot open archive {archive_path}: {exc}") from exc
            self._zip_handles[key] = archive
        try:
            with archive.open(record.member_path, "r") as handle:
                return handle.read()
        except (KeyError, OSError, zipfile.BadZipFile) as exc:
            raise DatasetError(f"cannot read {record.sample_id}: {exc}") from exc

    def __getitem__(self, index: int) -> SampleItem:
        record = self.records[index]
        raw = self._bytes_for(record)
        if record.byte_sha256 and hashlib.sha256(raw).hexdigest() != record.byte_sha256:
            raise DatasetError(f"image content changed since audit: {record.sample_id}")
        image = (self.transform.apply(raw, sample_id=record.sample_id, epoch=self.epoch)
                 if hasattr(self.transform, "apply") else self.transform(raw))
        label_index = None
        if record.class_id is not None:
            if record.class_id not in self.class_to_index:
                raise DatasetError(f"label {record.class_id!r} is absent from the class map")
            label_index = self.class_to_index[record.class_id]
        return SampleItem(image=image, sample_id=record.sample_id, class_id=record.class_id, label_index=label_index)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def close(self) -> None:
        for archive in self._zip_handles.values():
            archive.close()
        self._zip_handles.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_zip_handles"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._pid = os.getpid()
        self._zip_handles = {}

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown path
        try:
            self.close()
        except Exception:
            pass


def default_image_transform(raw: bytes, *, size: int = 224) -> Any:
    """Decode an image for tests or simple callers without fixing a model API."""

    try:
        from PIL import Image, ImageOps  # type: ignore
    except ImportError as exc:
        raise DatasetError("Pillow is required for image transforms") from exc
    with Image.open(io.BytesIO(raw)) as image:
        return ImageOps.exif_transpose(image).convert("RGB").resize((size, size))
