"""Full physical GPU identities across CUDA and nvidia-smi representations."""
import re
from uuid import UUID


_FULL_UUID = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")


def normalize_gpu_uuid(value):
    """Return lower-case bare UUID; reject indices, abbreviations and MIG IDs.

    This operates on individual identifiers, never rewrites sealed evidence or
    relaxes equality of runtime/source/dependency/asset dictionaries.
    """
    if isinstance(value, UUID):
        value = str(value)
    if not isinstance(value, str):
        raise ValueError("full physical GPU UUID required")
    bare = value[4:] if value.startswith("GPU-") else value
    if not _FULL_UUID.fullmatch(bare):
        raise ValueError("full physical GPU UUID required; device indices/abbreviations forbidden")
    return str(UUID(bare))


def cuda_gpu_uuids(values):
    """Validate CUDA_VISIBLE_DEVICES assignments and reject physical duplicates."""
    if not values or any(not isinstance(v, str) or not v.startswith("GPU-") for v in values):
        raise ValueError("CUDA assignments require full GPU-prefixed UUIDs")
    normalized = [normalize_gpu_uuid(v) for v in values]
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate physical GPU UUIDs")
    return ["GPU-" + v for v in normalized]
