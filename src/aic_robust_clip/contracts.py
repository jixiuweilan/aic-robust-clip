"""Versioned, JSON-serialisable contracts shared by the pipeline.

The contracts intentionally contain provenance and stage information in every
derived record.  This makes it difficult for a later-stage run to accidentally
consume an earlier-stage manifest or a test record.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, ClassVar, Literal, Mapping


SCHEMA_VERSION = "1.0"
STAGES = ("preliminary", "second_round", "semifinal")
ROLES = ("train", "test")
PARTITIONS = ("train", "dev", "confirm")
EXECUTION_MODES = ("smoke", "formal")


class ContractError(ValueError):
    """Raised when a persisted pipeline contract is invalid."""


def _require_choice(name: str, value: str, choices: tuple[str, ...]) -> str:
    if value not in choices:
        choices_text = ", ".join(choices)
        raise ContractError(f"{name} must be one of {choices_text}; got {value!r}")
    return value


def _require_digest(name: str, value: str | None) -> str | None:
    if value is not None and (not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
        raise ContractError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def canonical_json(value: Any) -> str:
    """Return stable JSON used for contract digests and reproducible output."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DatasetRegistration:
    stage: str
    train_archive: str
    test_archive: str | None = None
    declared_provenance: str | None = None
    source_url: str | None = None
    retrieval_date: str | None = None
    expected_train_images: int | None = None
    expected_test_images: int | None = None
    expected_classes: int | None = None
    completeness: Literal["unknown", "partial", "complete"] = "unknown"
    official_name_mapping: str | None = None
    schema_version: ClassVar[str] = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_choice("stage", self.stage, STAGES)
        if not self.train_archive:
            raise ContractError("train_archive is required")
        if self.completeness not in ("unknown", "partial", "complete"):
            raise ContractError("completeness must be unknown, partial, or complete")
        for name in ("expected_train_images", "expected_test_images", "expected_classes"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ContractError(f"{name} cannot be negative")
        if self.retrieval_date is not None:
            try:
                date.fromisoformat(self.retrieval_date)
            except ValueError as exc:
                raise ContractError("retrieval_date must be YYYY-MM-DD") from exc

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, **asdict(self)}


@dataclass(frozen=True)
class SampleRecord:
    stage: str
    role: str
    archive_identity: str
    archive_path: str
    member_path: str
    sample_id: str
    byte_size: int
    crc32: int
    byte_sha256: str | None = None
    pixel_sha256: str | None = None
    width: int | None = None
    height: int | None = None
    channels: int | None = None
    decode_status: str = "unchecked"
    class_id: str | None = None
    schema_version: ClassVar[str] = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_choice("stage", self.stage, STAGES)
        _require_choice("role", self.role, ROLES)
        if self.role == "test" and self.class_id is not None:
            raise ContractError("test records cannot carry class_id")
        if not self.sample_id or not self.member_path or not self.archive_identity:
            raise ContractError("sample_id, member_path, and archive_identity are required")
        if self.member_path.startswith("/") or "\\" in self.member_path:
            raise ContractError(f"member_path must be a normalized relative path: {self.member_path!r}")
        if any(part in ("", ".", "..") for part in self.member_path.split("/")):
            raise ContractError(f"member_path contains unsafe components: {self.member_path!r}")
        if self.byte_size < 0 or not 0 <= self.crc32 <= 0xFFFFFFFF:
            raise ContractError("invalid archive member size or CRC32")
        _require_digest("byte_sha256", self.byte_sha256)
        _require_digest("pixel_sha256", self.pixel_sha256)
        if self.width is not None and self.width <= 0:
            raise ContractError("width must be positive")
        if self.height is not None and self.height <= 0:
            raise ContractError("height must be positive")
        if self.class_id is not None and not self.class_id.strip() == self.class_id:
            raise ContractError("class_id must not have surrounding whitespace")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, **asdict(self)}


@dataclass(frozen=True)
class ClassMap:
    stage: str
    id_to_index: Mapping[str, int]
    schema_version: ClassVar[str] = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_choice("stage", self.stage, STAGES)
        if not isinstance(self.id_to_index, Mapping):
            raise ContractError("id_to_index must be a mapping")
        values = list(self.id_to_index.values())
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ContractError("class map indices must be integers")
        if not self.id_to_index or sorted(values) != list(range(len(values))):
            raise ContractError("class map indices must be contiguous and start at zero")
        if any(not isinstance(key, str) or not key or key.strip() != key for key in self.id_to_index):
            raise ContractError("class IDs must be non-empty and whitespace-free")

    @classmethod
    def from_ids(cls, stage: str, class_ids: list[str] | set[str] | tuple[str, ...]) -> "ClassMap":
        ids = list(class_ids)
        if len(ids) != len(set(ids)):
            raise ContractError("class IDs must be unique")
        # Numeric folder IDs sort numerically while retaining their exact text.
        try:
            ordered = sorted(ids, key=lambda value: (int(value), value))
        except ValueError:
            ordered = sorted(ids)
        return cls(stage=stage, id_to_index={value: index for index, value in enumerate(ordered)})

    @property
    def index_to_id(self) -> dict[int, str]:
        return {index: class_id for class_id, index in self.id_to_index.items()}

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def index_for(self, class_id: str) -> int:
        try:
            return self.id_to_index[class_id]
        except KeyError as exc:
            raise ContractError(f"class ID is not in the class map: {class_id!r}") from exc

    def id_for(self, index: int) -> str:
        try:
            return self.index_to_id[index]
        except KeyError as exc:
            raise ContractError(f"class index is not in the class map: {index}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "id_to_index": dict(sorted(self.id_to_index.items())),
        }


@dataclass(frozen=True)
class SplitRecord:
    sample_id: str
    group_id: str
    partition: str
    seed: int
    algorithm_version: str
    parent_manifest_digest: str
    stage: str
    role: str
    class_id: str | None
    label_quality: str = "noisy_proxy"
    schema_version: ClassVar[str] = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_choice("stage", self.stage, STAGES)
        _require_choice("role", self.role, ROLES)
        if self.partition not in PARTITIONS:
            raise ContractError(f"partition must be one of {PARTITIONS}; got {self.partition!r}")
        if self.role == "test" or self.partition == "test":
            raise ContractError("test records cannot be split records")
        if self.role == "train" and self.class_id is None:
            raise ContractError("training split records require class_id")
        _require_digest("parent_manifest_digest", self.parent_manifest_digest)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, **asdict(self)}


@dataclass(frozen=True)
class RunConfig:
    stage: str
    execution_mode: str = "smoke"
    manifest_digest: str = ""
    split_digest: str = ""
    class_map_digest: str = ""
    official_weight_id: str = "openai/clip-vit-base-patch32"
    official_weight_revision: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)
    seed: int = 17
    output_root: str = "outputs"
    batch_size: int = 1
    max_samples: int = 8
    max_updates: int = 2
    max_eval_batches: int = 2
    schema_version: ClassVar[str] = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_choice("stage", self.stage, STAGES)
        _require_choice("execution_mode", self.execution_mode, EXECUTION_MODES)
        if self.official_weight_id != "openai/clip-vit-base-patch32":
            raise ContractError("only OpenAI CLIP ViT-B/32 is permitted")
        for name in ("batch_size", "max_samples", "max_updates", "max_eval_batches"):
            if getattr(self, name) <= 0:
                raise ContractError(f"{name} must be positive")
        if self.execution_mode == "smoke" and self.max_updates > 3:
            raise ContractError("smoke max_updates cannot exceed 3")
        if self.seed < 0:
            raise ContractError("seed must be non-negative")

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, **asdict(self)}


@dataclass(frozen=True)
class CheckpointMetadata:
    model_family: str
    stage: str
    class_map_digest: str
    manifest_digest: str
    split_digest: str
    official_weight_id: str
    official_weight_revision: str
    configuration_digest: str
    code_revision: str
    progress: Mapping[str, Any]
    optimizer_state_present: bool
    scheduler_state_present: bool
    rng_state_present: bool
    sampler_state_present: bool
    module_state: Mapping[str, Any] = field(default_factory=dict)
    weight_files_digest: str = ""
    preprocessing_digest: str = ""
    initialization_digest: str = ""
    execution_mode: str = "smoke"
    schema_version: ClassVar[str] = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_choice("stage", self.stage, STAGES)
        if self.official_weight_id != "openai/clip-vit-base-patch32":
            raise ContractError("checkpoint uses a non-permitted backbone")
        if not self.official_weight_revision or self.official_weight_revision in {"main", "master", "latest"}:
            raise ContractError("checkpoint must record an immutable weight revision")
        for name in ("class_map_digest", "manifest_digest", "split_digest", "configuration_digest"):
            _require_digest(name, getattr(self, name))

    def assert_compatible(
        self,
        *,
        stage: str,
        class_map_digest: str,
        manifest_digest: str,
        split_digest: str,
        official_weight_id: str,
        official_weight_revision: str,
        weight_files_digest: str | None = None,
        preprocessing_digest: str | None = None,
        initialization_digest: str | None = None,
        execution_mode: str | None = None,
    ) -> None:
        expected = {
            "stage": stage,
            "class map": class_map_digest,
            "manifest": manifest_digest,
            "split": split_digest,
            "weight ID": official_weight_id,
            "weight revision": official_weight_revision,
        }
        actual = {
            "stage": self.stage,
            "class map": self.class_map_digest,
            "manifest": self.manifest_digest,
            "split": self.split_digest,
            "weight ID": self.official_weight_id,
            "weight revision": self.official_weight_revision,
        }
        mismatches = [f"{name}: checkpoint={actual[name]!r}, requested={value!r}" for name, value in expected.items() if actual[name] != value]
        for name, value in (("weight_files_digest", weight_files_digest), ("preprocessing_digest", preprocessing_digest),
                            ("initialization_digest", initialization_digest), ("execution_mode", execution_mode)):
            if value is not None and (not getattr(self, name) or getattr(self, name) != value):
                mismatches.append(f"{name}: missing or mismatched identity")
        if mismatches:
            raise ContractError("incompatible checkpoint: " + "; ".join(mismatches))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, **asdict(self)}


def write_json(path: Path | str, value: Any) -> None:
    """Write a contract or plain JSON value atomically enough for local runs."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(canonical_json(value) + "\n", encoding="utf-8")


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
