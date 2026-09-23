"""Read-only ZIP inventory and image integrity auditing.

Auditing never extracts, rewrites, relabels, or removes an input member.  A
partial report is deliberately a different artifact from a complete manifest.
"""

from __future__ import annotations

import hashlib
import io
import json
import posixpath
import zlib
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Callable

from ..contracts import ClassMap, ContractError, SampleRecord, STAGES, sha256_json, write_json


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"})
PIXEL_HASH_SPEC = "pillow-exif-transpose-rgb-row-major-v1"


class AuditError(ValueError):
    """Raised for an archive that cannot produce a trustworthy report."""


@dataclass(frozen=True)
class AuditFailure:
    member_path: str
    sample_id: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "member_path": self.member_path,
            "sample_id": self.sample_id,
            "reason": self.reason,
        }


@dataclass
class AuditReport:
    stage: str
    role: str
    archive_path: str
    archive_identity: str
    total_members: int
    scanned_members: int
    complete: bool
    decode_policy: str
    records: list[SampleRecord] = field(default_factory=list)
    failures: list[AuditFailure] = field(default_factory=list)
    exclusions: list[dict[str, str]] = field(default_factory=list)
    nonimage_members: list[str] = field(default_factory=list)
    unsafe_members: list[str] = field(default_factory=list)
    schema_version: str = "1.0"

    @property
    def manifest_digest(self) -> str:
        return sha256_json([record.to_dict() for record in sorted(self.records, key=lambda item: item.sample_id)])

    @property
    def usable(self) -> bool:
        """Whether this report is complete and has no integrity failures."""

        return self.complete and not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "role": self.role,
            "archive_path": self.archive_path,
            "archive_identity": self.archive_identity,
            "total_members": self.total_members,
            "scanned_members": self.scanned_members,
            "complete": self.complete,
            "decode_policy": self.decode_policy,
            "manifest_digest": self.manifest_digest,
            "records": [record.to_dict() for record in sorted(self.records, key=lambda item: item.sample_id)],
            "failures": [failure.to_dict() for failure in sorted(self.failures, key=lambda item: (item.member_path, item.reason))],
            "exclusions": self.exclusions,
            "nonimage_members": sorted(self.nonimage_members),
            "unsafe_members": sorted(self.unsafe_members),
        }


def _validate_stage_role(stage: str, role: str) -> None:
    if stage not in STAGES:
        raise AuditError(f"stage must be one of {STAGES}; got {stage!r}")
    if role not in ("train", "test"):
        raise AuditError("role must be train or test")


def _archive_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _partial_identity(path: Path, total_members: int) -> str:
    stat = path.stat()
    value = f"partial:{path.name}:{stat.st_size}:{stat.st_mtime_ns}:{total_members}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _member_path(raw_name: str) -> str | None:
    """Return a safe normalized POSIX member path, or None if unsafe."""

    if not raw_name or "\\" in raw_name or raw_name.startswith("/"):
        return None
    normalized = posixpath.normpath(raw_name)
    if normalized != raw_name or normalized in ("", "."):
        return None
    if any(part in ("", ".", "..") for part in normalized.split("/")):
        return None
    return normalized


def _class_id_for(member_path: str, role: str) -> str | None:
    if role == "test":
        if "/" in member_path:
            raise AuditError(f"test image must be flat in the archive: {member_path!r}")
        return None
    parts = member_path.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise AuditError(f"training image must be in one class directory: {member_path!r}")
    return parts[0]


def _sample_id(archive_identity: str, role: str, member_path: str) -> str:
    return hashlib.sha256(f"{archive_identity}\0{role}\0{member_path}".encode("utf-8")).hexdigest()


def _decode_pixels(raw: bytes, *, allow_truncated: bool = False) -> tuple[str, int, int, int]:
    try:
        from PIL import Image, ImageFile, ImageOps  # type: ignore
    except ImportError as exc:
        raise AuditError("Pillow is required for image decoding; install the audit extra") from exc

    previous_truncated_policy = ImageFile.LOAD_TRUNCATED_IMAGES
    if allow_truncated:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            oriented = ImageOps.exif_transpose(image)
            rgb = oriented.convert("RGB")
            width, height = rgb.size
            pixels = rgb.tobytes("raw", "RGB")
            decoder = getattr(__import__("PIL"), "__version__", "unknown")
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous_truncated_policy
    frame = f"{PIXEL_HASH_SPEC}\0decoder={decoder}\0width={width}\0height={height}\0channels=3\0".encode("ascii")
    return hashlib.sha256(frame + pixels).hexdigest(), width, height, 3


def _read_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    max_member_bytes: int | None = 64 * 1024 * 1024,
) -> tuple[bytes, int]:
    if max_member_bytes is not None and (max_member_bytes <= 0 or info.file_size > max_member_bytes):
        raise AuditError(
            f"member exceeds bounded audit memory: {info.file_size} > {max_member_bytes} bytes"
        )
    digest = zlib.crc32(b"")
    chunks: list[bytes] = []
    with archive.open(info, "r") as handle:
        while chunk := handle.read(1024 * 1024):
            digest = zlib.crc32(chunk, digest)
            chunks.append(chunk)
    return b"".join(chunks), digest & 0xFFFFFFFF


def audit_archive(
    path: Path | str,
    *,
    stage: str,
    role: str,
    decode: bool = True,
    allow_truncated: bool = False,
    max_members: int | None = None,
    compute_archive_hash: bool = True,
    max_member_bytes: int | None = 64 * 1024 * 1024,
    member_prefix: str | None = None,
    progress: Callable[[dict], None] | None = None,
    expected_sha256: str | None = None,
    exclude_member: str | None = None,
    exclude_byte_sha256: str | None = None,
) -> AuditReport:
    """Inspect one archive and return a deterministic report.

    ``max_members`` is only for a visibly partial bounded check.  Formal
    manifests must use the defaults so the report is complete and has the
    actual archive SHA-256 identity.
    """

    _validate_stage_role(stage, role)
    if member_prefix is not None and (role != "train" or not isinstance(member_prefix, str)
                                      or not member_prefix or _member_path(member_prefix) != member_prefix):
        raise AuditError("member_prefix requires an explicit safe training directory")
    if (exclude_member is None) != (exclude_byte_sha256 is None):
        raise AuditError("excluded training member requires its byte SHA256")
    if exclude_member is not None and (role != "train" or _member_path(exclude_member) != exclude_member
                                       or len(exclude_byte_sha256) != 64):
        raise AuditError("invalid excluded training member")
    archive_path = Path(path)
    if not archive_path.is_file():
        raise AuditError(f"archive does not exist: {archive_path}")
    try:
        archive = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise AuditError(f"cannot open ZIP archive {archive_path}: {exc}") from exc

    with archive:
        infos = archive.infolist()
        if max_members is not None and max_members <= 0:
            raise AuditError("max_members must be positive when a bounded audit is requested")
        if progress:
            progress({"phase": "hashing", "total_members": len(infos), "scanned_members": 0})
        archive_identity = _archive_sha256(archive_path) if compute_archive_hash else _partial_identity(archive_path, len(infos))
        if expected_sha256 is not None and archive_identity != expected_sha256:
            raise AuditError("archive differs from registered SHA256; decoding not started")
        selected = infos if max_members is None else infos[:max_members]
        complete = max_members is None or len(selected) == len(infos)
        report = AuditReport(
            stage=stage,
            role=role,
            archive_path=str(archive_path),
            archive_identity=archive_identity,
            total_members=len(infos),
            scanned_members=len(selected),
            complete=complete,
            decode_policy="strict" if not allow_truncated else "truncated-recovery-enabled",
        )
        seen: set[str] = set()
        for index, info in enumerate(selected):
            if progress and index % 100 == 0:
                progress({"phase": "decoding", "total_members": len(infos), "scanned_members": index,
                          "recorded_images": len(report.records), "failures": len(report.failures)})
            if info.is_dir():
                continue
            normalized = _member_path(info.filename)
            if normalized is None:
                sample_id = _sample_id(archive_identity, role, info.filename)
                report.unsafe_members.append(info.filename)
                report.failures.append(AuditFailure(info.filename, sample_id, "unsafe_member_path"))
                continue
            if normalized in seen:
                sample_id = _sample_id(archive_identity, role, normalized)
                report.failures.append(AuditFailure(normalized, sample_id, "duplicate_member_path"))
                continue
            seen.add(normalized)
            suffix = Path(normalized).suffix.lower()
            sample_id = _sample_id(archive_identity, role, normalized)
            if suffix not in IMAGE_SUFFIXES:
                report.nonimage_members.append(normalized)
                report.failures.append(AuditFailure(normalized, sample_id, "nonimage_member"))
                continue
            try:
                class_path = normalized
                if member_prefix is not None:
                    if not normalized.startswith(member_prefix + "/"):
                        raise AuditError("training image outside declared member_prefix")
                    class_path = normalized[len(member_prefix) + 1:]
                class_id = _class_id_for(class_path, role)
            except AuditError as exc:
                report.failures.append(AuditFailure(normalized, sample_id, "invalid_layout:" + str(exc)))
                continue
            if info.flag_bits & 0x1:
                report.failures.append(AuditFailure(normalized, sample_id, "encrypted_member"))
                report.records.append(
                    SampleRecord(
                        stage=stage,
                        role=role,
                        archive_identity=archive_identity,
                        archive_path=str(archive_path),
                        member_path=normalized,
                        sample_id=sample_id,
                        byte_size=info.file_size,
                        crc32=info.CRC,
                        decode_status="encrypted",
                        class_id=class_id if role == "train" else None,
                    )
                )
                continue
            raw: bytes | None = None
            byte_digest: str | None = None
            decode_status = "unchecked"
            pixel_digest = None
            width = height = channels = None
            try:
                raw, crc = _read_member(archive, info, max_member_bytes=max_member_bytes)
                if crc != info.CRC:
                    raise AuditError(f"bad_crc:expected={info.CRC:08x},actual={crc:08x}")
                byte_digest = hashlib.sha256(raw).hexdigest()
                if decode:
                    try:
                        pixel_digest, width, height, channels = _decode_pixels(raw)
                        decode_status = "decoded"
                    except Exception:
                        if not allow_truncated:
                            raise
                        pixel_digest, width, height, channels = _decode_pixels(raw, allow_truncated=True)
                        decode_status = "decoded_truncated_recovery"
                else:
                    decode_status = "crc_verified"
            except (AuditError, OSError, RuntimeError, SyntaxError, zipfile.BadZipFile) as exc:
                reason = str(exc)
                if (normalized == exclude_member and byte_digest == exclude_byte_sha256
                        and isinstance(exc, SyntaxError)):
                    report.exclusions.append({"member_path": normalized, "sample_id": sample_id,
                                              "byte_sha256": byte_digest, "reason": reason})
                    continue
                if decode_status == "unchecked":
                    decode_status = "failed"
                report.failures.append(AuditFailure(normalized, sample_id, reason))
            try:
                report.records.append(
                    SampleRecord(
                        stage=stage,
                        role=role,
                        archive_identity=archive_identity,
                        archive_path=str(archive_path),
                        member_path=normalized,
                        sample_id=sample_id,
                        byte_size=info.file_size,
                        crc32=info.CRC,
                        byte_sha256=byte_digest,
                        pixel_sha256=pixel_digest,
                        width=width,
                        height=height,
                        channels=channels,
                        decode_status=decode_status,
                        class_id=class_id,
                    )
                )
            except ContractError as exc:
                report.failures.append(AuditFailure(normalized, sample_id, "invalid_record:" + str(exc)))
    if exclude_member is not None and len(report.exclusions) != 1:
        raise AuditError("authorized training exclusion was not verified")
    if progress:
        progress({"phase": "decoded", "total_members": report.total_members,
                  "scanned_members": report.scanned_members, "recorded_images": len(report.records),
                  "failures": len(report.failures)})
    return report


def audit_stage(
    train_archive: Path | str,
    test_archive: Path | str,
    *,
    stage: str,
    decode_test: bool = False,
    **kwargs: Any,
) -> tuple[AuditReport, AuditReport]:
    """Audit train and test archives as separate, non-interchangeable roles."""

    train = audit_archive(train_archive, stage=stage, role="train", **kwargs)
    test = audit_archive(test_archive, stage=stage, role="test", decode=decode_test, **kwargs)
    return train, test


def write_audit_report(path: Path | str, report: AuditReport) -> None:
    write_json(path, report.to_dict())


def write_manifest(path: Path | str, records: Iterable[SampleRecord], *, complete: bool) -> str:
    """Persist a manifest with an explicit completeness flag and digest."""

    ordered = [record.to_dict() for record in sorted(records, key=lambda item: item.sample_id)]
    value = {"schema_version": "1.0", "complete": complete, "records": ordered}
    value["manifest_digest"] = sha256_json(ordered)
    write_json(path, value)
    return value["manifest_digest"]


def load_manifest(path: Path | str, *, require_complete: bool = False) -> list[SampleRecord]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if require_complete and not value.get("complete", False):
        raise AuditError("partial manifest cannot be used for a formal run")
    if require_complete and value.get("failures"):
        raise AuditError("manifest is complete in coverage but contains audit failures")
    records: list[SampleRecord] = []
    for item in value.get("records", []):
        item = dict(item)
        item.pop("schema_version", None)
        records.append(SampleRecord(**item))
    expected = sha256_json([record.to_dict() for record in sorted(records, key=lambda item: item.sample_id)])
    if value.get("manifest_digest") != expected:
        raise AuditError("manifest digest does not match its records")
    return records


def class_map_from_records(records: Iterable[SampleRecord]) -> ClassMap:
    """Build a numeric-label class map without renaming or padding IDs."""

    records = list(records)
    if not records or any(record.role != "train" or record.class_id is None for record in records):
        raise AuditError("class maps require non-empty training records with observed IDs")
    stages = {record.stage for record in records}
    if len(stages) != 1:
        raise AuditError("class map construction cannot mix stages")
    return ClassMap.from_ids(next(iter(stages)), {record.class_id for record in records if record.class_id is not None})
