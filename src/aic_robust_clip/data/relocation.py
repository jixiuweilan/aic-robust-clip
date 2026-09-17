"""Opt-in, content-verified archive locations; persisted identities never change.

Only image-read paths use this module. Audit/split/cache/head identities continue
to use the original records. Hash verification is process-local and invalidated
by changes to the target's inode, size or timestamps. Inputs must remain immutable
while a run is active; this is not a lock against concurrent filesystem writers.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat


class RelocationError(RuntimeError):
    """A configured relocation is invalid; never fall back to another archive."""


_mapping_cache: tuple | None = None
_verified: dict[tuple, tuple] = {}


def _signature(path: Path) -> tuple:
    try:
        value = path.stat()
    except (OSError, ValueError) as exc:
        raise RelocationError(f"archive relocation file is missing or unreadable: {path}") from exc
    if not stat.S_ISREG(value.st_mode):
        raise RelocationError(f"archive relocation requires a regular file: {path}")
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise RelocationError(f"duplicate archive-location JSON key: {key}")
        value[key] = item
    return value


def _mapping() -> dict:
    global _mapping_cache
    raw = os.environ.get("AIC_ARCHIVE_LOCATIONS")
    if not raw:
        return {}
    path = Path(raw).absolute()
    signature = _signature(path)
    key = (os.getpid(), str(path), signature)
    if _mapping_cache is not None and _mapping_cache[0] == key:
        return _mapping_cache[1]
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, ValueError) as exc:
        raise RelocationError(f"cannot parse archive-location mapping: {path}") from exc
    if _signature(path) != signature:
        raise RelocationError("archive-location mapping changed while being read")
    if (not isinstance(value, dict) or set(value) != {"schema_version", "reason", "archives"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1):
        raise RelocationError("archive-location mapping requires schema_version 1, reason and archives")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise RelocationError("archive-location mapping requires a non-empty reason")
    archives = value["archives"]
    if not isinstance(archives, dict) or not archives:
        raise RelocationError("archive-location mapping requires a non-empty archives object")
    for old, entry in archives.items():
        if ("\x00" in old or not Path(old).is_absolute() or not isinstance(entry, dict)
                or set(entry) != {"path", "sha256"}):
            raise RelocationError("archive-location entries require absolute source paths, path and sha256")
        target, expected = entry["path"], entry["sha256"]
        if not isinstance(target, str) or "\x00" in target or not Path(target).is_absolute():
            raise RelocationError("relocated archive path must be absolute")
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise RelocationError("relocated archive requires a lowercase SHA-256 digest")
    _mapping_cache = (key, archives)
    return archives


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RelocationError(f"cannot hash relocated archive: {path}") from exc
    return digest.hexdigest()


def resolve_archive(path: Path, *, archive_identity: str) -> Path:
    """Verify a mapped file against BOTH the mapping and the audited identity.

    Unmapped paths retain the original behavior. A malformed enabled mapping,
    missing target or mismatched digest fails closed before any member read.
    No alternate path is tried, even if the original archive is still present.
    """
    entry = _mapping().get(str(path))
    if entry is None:
        return path
    expected = entry["sha256"]
    if expected != archive_identity:
        raise RelocationError("relocated archive hash mismatch with audited archive_identity")
    target = Path(entry["path"])
    signature = _signature(target)
    key = (os.getpid(), str(target), expected)
    if _verified.get(key) != signature:
        actual = _sha256_file(target)
        if _signature(target) != signature:
            raise RelocationError(f"relocated archive changed during verification: {target}")
        if actual != expected:
            raise RelocationError(f"relocated archive hash mismatch: {target}; refusing to read")
        _verified[key] = signature
    return target
