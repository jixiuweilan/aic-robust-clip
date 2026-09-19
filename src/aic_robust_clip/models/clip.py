"""The sole permitted CLIP backbone and its fixed preprocessing contract."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


OPENAI_CLIP_VIT_B32 = "openai/clip-vit-base-patch32"

try:  # Keep contract/audit tests usable in the standard-library environment.
    import torch  # type: ignore
    from torch import nn  # type: ignore
except ImportError:  # pragma: no cover - exercised by the dependency-free CI image
    torch = None  # type: ignore

    class _Module:
        pass

    nn = type("nn", (), {"Module": _Module})  # type: ignore


class ClipDependencyError(RuntimeError):
    """Raised when model functionality is requested without its dependencies."""


class WeightIdentityError(ValueError):
    """Raised for an unapproved or insufficiently identified CLIP weight."""


def validate_weight_identity(model_id: str, revision: str) -> None:
    if model_id != OPENAI_CLIP_VIT_B32:
        raise WeightIdentityError("only openai/clip-vit-base-patch32 is permitted")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise WeightIdentityError("pass an immutable inspected Hugging Face revision, not a moving branch")


def sha256_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        digest.update(str(path.name).encode("utf-8"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def weight_digest(weight_dir: Path | str) -> str:
    root = Path(weight_dir)
    if not root.is_dir():
        raise WeightIdentityError(f"weight directory does not exist: {root}")
    files = [path for path in root.rglob("*") if path.is_file() and not path.name.startswith(".")]
    if not files:
        raise WeightIdentityError(f"weight directory is empty: {root}")
    return sha256_files(files)


@dataclass(frozen=True)
class WeightIdentity:
    model_id: str
    revision: str
    digest: str

    def __post_init__(self) -> None:
        validate_weight_identity(self.model_id, self.revision)
        if len(self.digest) != 64 or any(char not in "0123456789abcdef" for char in self.digest):
            raise WeightIdentityError("weight digest must be a lowercase SHA-256 hex digest")

    def to_dict(self) -> dict[str, str]:
        return {"model_id": self.model_id, "revision": self.revision, "digest": self.digest}


class FrozenCLIPEncoder(nn.Module):  # type: ignore[misc]
    """CLIP image projection with frozen parameters and normalized output."""

    def __init__(self, clip_model: Any) -> None:
        if torch is None:
            raise ClipDependencyError("torch and transformers are required for CLIP loading")
        super().__init__()
        self.clip_model = clip_model
        for parameter in self.clip_model.parameters():
            parameter.requires_grad_(False)
        projection_dim = getattr(getattr(clip_model, "config", None), "projection_dim", None)
        if projection_dim is None:
            projection_dim = getattr(getattr(clip_model, "config", None), "vision_config", None)
            projection_dim = getattr(projection_dim, "projection_dim", None)
        if projection_dim is None:
            raise ClipDependencyError("loaded CLIP model has no projection dimension")
        self.output_dim = int(projection_dim)

    def forward(self, pixel_values: Any, *, no_grad: bool = True) -> Any:
        if no_grad:
            with torch.no_grad():
                features = self.clip_model.get_image_features(pixel_values=pixel_values)
        else:
            features = self.clip_model.get_image_features(pixel_values=pixel_values)
        # Keep feature normalization in FP32 under training autocast as well.
        # FP32 callers retain the exact original arithmetic.
        if features.dtype in (torch.float16, torch.bfloat16):
            features = features.float()
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    @property
    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)


@dataclass
class ClipBundle:
    encoder: FrozenCLIPEncoder
    processor: Any
    identity: WeightIdentity

    def preprocess(self, image: Any) -> Any:
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]


def load_openai_clip(
    *,
    revision: str,
    local_path: Path | str | None = None,
    local_files_only: bool = True,
    weight_digest_value: str | None = None,
    device: str | None = None,
) -> ClipBundle:
    """Load only the permitted HF CLIP package.

    ``local_files_only`` defaults to True so tests and smoke checks cannot
    silently download a many-hundred-megabyte model.  Provisioning the locked
    revision belongs to the separate training environment.
    """

    validate_weight_identity(OPENAI_CLIP_VIT_B32, revision)
    from .provision import inspect_weights
    if local_path is None or not local_files_only:
        raise WeightIdentityError("use the explicit provision command, then load its verified local directory offline")
    inspected = inspect_weights(local_path, revision)
    if weight_digest_value is not None and weight_digest_value != inspected["digest"]:
        raise WeightIdentityError("supplied digest differs from actual official snapshot files")
    if torch is None:
        raise ClipDependencyError("install the clip extra (torch, transformers, Pillow) before loading weights")
    try:
        from transformers import CLIPImageProcessor, CLIPModel  # type: ignore
    except ImportError as exc:
        raise ClipDependencyError("transformers is required for the Hugging Face CLIP route") from exc

    source = str(local_path) if local_path is not None else OPENAI_CLIP_VIT_B32
    kwargs = {"revision": revision, "local_files_only": local_files_only}
    try:
        model = CLIPModel.from_pretrained(source, **kwargs)
        processor = CLIPImageProcessor.from_pretrained(source, **kwargs)
    except OSError as exc:
        location = local_path or OPENAI_CLIP_VIT_B32
        raise ClipDependencyError(
            f"locked CLIP revision {revision!r} is unavailable at {location!r}; "
            "provision it on the training machine or pass a validated local cache"
        ) from exc
    weight_digest_value = inspected["digest"]
    bundle = ClipBundle(
        encoder=FrozenCLIPEncoder(model),
        processor=processor,
        identity=WeightIdentity(OPENAI_CLIP_VIT_B32, revision, weight_digest_value),
    )
    if device:
        bundle.encoder.to(device)
    bundle.encoder.eval()
    return bundle


@dataclass(frozen=True)
class FeatureCacheKey:
    stage: str
    source_manifest_digest: str
    split_digest: str
    weight_identity: WeightIdentity
    preprocessing_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "source_manifest_digest": self.source_manifest_digest,
            "split_digest": self.split_digest,
            "weight_identity": self.weight_identity.to_dict(),
            "preprocessing_id": self.preprocessing_id,
        }


class FeatureCache:
    """Streaming cache envelope; only fixed-view features may use this API."""

    def __init__(self, key: FeatureCacheKey, sample_ids: list[str], features: Any) -> None:
        if not sample_ids:
            raise ValueError("feature cache cannot be empty")
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("feature cache sample IDs must be unique")
        if getattr(features, "shape", [None])[0] != len(sample_ids):
            raise ValueError("feature cache row count does not match sample IDs")
        self.key = key
        self.sample_ids = list(sample_ids)
        self.features = features

    def save(self, path: Path | str) -> None:
        if torch is None:
            raise ClipDependencyError("torch is required to save feature caches")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"key": self.key.to_dict(), "sample_ids": self.sample_ids, "features": self.features}, destination)

    @classmethod
    def load(cls, path: Path | str, expected_key: FeatureCacheKey) -> "FeatureCache":
        if torch is None:
            raise ClipDependencyError("torch is required to load feature caches")
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("key") != expected_key.to_dict():
            raise ValueError("feature cache key mismatch; regenerate it for this stage/split/weight/preprocessing")
        return cls(expected_key, list(payload["sample_ids"]), payload["features"])
