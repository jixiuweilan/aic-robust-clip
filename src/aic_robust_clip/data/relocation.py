"""Opt-in, content-verified archive locations; persisted identities never change.

Only image-read paths use this module. Audit/split/cache/head identities continue
to use the original records. Hash verification is process-local, guarded by
Linux inotify plus stat. Only local Linux filesystems (including WSL2 Linux
volumes) are supported. Inputs must remain immutable
while a run is active; this is not a lock against concurrent filesystem writers.
"""
from __future__ import annotations

import atexit
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import sys
import weakref


class RelocationError(RuntimeError):
    """A configured relocation is invalid; never fall back to another archive."""


_mapping_cache: tuple | None = None
_verified: dict[tuple, _Watch] = {}
_readers = weakref.WeakSet()
_process = os.getpid()
_finalizer = None

# linux/inotify.h: watch writes/attributes and fail closed on identity/watch loss.
_CHANGED = 0x00000002 | 0x00000004 | 0x00000008
_LOST = 0x00000400 | 0x00000800 | 0x00002000 | 0x00004000 | 0x00008000
_HEADER = struct.Struct("iIII")


def register_reader(reader):
    _readers.add(reader)


def _invalidate(path):
    for reader in list(_readers):
        reader.invalidate_archive(str(path))


class _Watch:
    def __init__(self, path):
        self.path, self.fd, self.signature, self.error = path, -1, None, None
        try:
            if sys.platform != "linux":
                raise OSError("Linux inotify is required")
            libc = ctypes.CDLL(None, use_errno=True)
            init = libc.inotify_init1
            init.argtypes, init.restype = [ctypes.c_int], ctypes.c_int
            add = libc.inotify_add_watch
            add.argtypes, add.restype = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32], ctypes.c_int
            self.fd = init(os.O_NONBLOCK | os.O_CLOEXEC)
            if self.fd < 0:
                raise OSError(ctypes.get_errno(), "inotify_init1")
            self.wd = add(self.fd, os.fsencode(path), _CHANGED | _LOST)
            if self.wd < 0:
                raise OSError(ctypes.get_errno(), "inotify_add_watch")
        except (OSError, AttributeError) as exc:
            self.close()
            raise RelocationError(f"archive monitor unavailable: {path}: {exc}") from exc

    def close(self):
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def fail(self, message):
        self.signature = None
        self.error = message
        _invalidate(self.path)
        self.close()
        raise RelocationError(message)

    def poll(self):
        if self.error:
            raise RelocationError(self.error)
        changed = False
        try:
            while True:
                try:
                    events = os.read(self.fd, 65536)
                except BlockingIOError:
                    break
                except InterruptedError:
                    continue
                if not events:
                    self.fail(f"archive monitor lost: {self.path}")
                offset = 0
                while offset < len(events):
                    if len(events) - offset < _HEADER.size:
                        self.fail("truncated archive monitor event")
                    wd, mask, _, size = _HEADER.unpack_from(events, offset)
                    offset += _HEADER.size + size
                    if offset > len(events) or mask & _LOST or wd != self.wd:
                        self.fail(f"archive monitor lost, overflow, deletion or replacement: {self.path}")
                    changed |= bool(mask & _CHANGED)
        except OSError as exc:
            self.fail(f"archive monitor read failed: {self.path}: {exc}")
        if changed:
            self.signature = None
            _invalidate(self.path)
        return changed


def close_monitors():
    global _mapping_cache
    for watch in _verified.values():
        _invalidate(watch.path)
        watch.close()
    _verified.clear()
    _mapping_cache = None


def _after_fork():
    global _process, _finalizer
    close_monitors()  # close inherited descriptors, never remove parent's watches
    _process = os.getpid()
    # multiprocessing clears inherited finalizers AFTER os.register_at_fork.
    # Register our new finalizer lazily on the child's first resolver call.
    _finalizer = None


def _ensure_process():
    global _finalizer
    if _process != os.getpid():
        _after_fork()
    if _finalizer is None:
        # multiprocessing may exit with os._exit, bypassing atexit.
        from multiprocessing.util import Finalize
        _finalizer = Finalize(None, close_monitors, exitpriority=10)


atexit.register(close_monitors)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


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
    try:
        content = path.read_bytes()
        content_digest = hashlib.sha256(content).hexdigest()
        # Equal-length rewrites can share ALL stat fields on coarse clocks.
        # Recheck bytes as well as stat, including before returning a cache hit.
        if path.read_bytes() != content or _signature(path) != signature:
            raise RelocationError("archive-location mapping changed while being read")
    except OSError as exc:
        raise RelocationError(f"cannot read archive-location mapping: {path}") from exc
    key = (os.getpid(), str(path), content_digest)
    if _mapping_cache is not None and _mapping_cache[0] == key:
        return _mapping_cache[1]
    try:
        value = json.loads(content.decode("utf-8"), object_pairs_hook=_unique_object)
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
    _ensure_process()
    entry = _mapping().get(str(path))
    if entry is None:
        return path
    expected = entry["sha256"]
    if expected != archive_identity:
        raise RelocationError("relocated archive hash mismatch with audited archive_identity")
    target = Path(entry["path"])
    key = (os.getpid(), str(target), expected)
    watch = _verified.get(key)
    if watch is None:
        _invalidate(target)
        before = _signature(target)
        watch = _Watch(target)  # must precede the first complete hash
        _verified[key] = watch
        watch.identity = before[:2]
        if _signature(target) != before:
            watch.fail(f"relocated archive changed while establishing monitor: {target}")
    watch.poll()
    try:
        signature = _signature(target)
    except RelocationError:
        watch.fail(f"relocated archive deleted or unreadable: {target}")
    if signature[:2] != watch.identity:
        watch.fail(f"relocated archive replaced: {target}")
    if watch.signature != signature:
        _invalidate(target)
        watch.signature = None
        actual = _sha256_file(target)
        changed = watch.poll()
        if changed or _signature(target) != signature:
            watch.fail(f"relocated archive changed during verification: {target}")
        if actual != expected:
            raise RelocationError(f"relocated archive hash mismatch: {target}; refusing to read")
        watch.signature = signature
    return target
