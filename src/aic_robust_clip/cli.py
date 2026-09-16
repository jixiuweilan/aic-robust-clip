"""Small explicit command-line entry points for preparation and packaging."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .contracts import ClassMap, ContractError, read_json, write_json
from .data.audit import AuditError, audit_archive, class_map_from_records, load_manifest, write_audit_report
from .data.splits import SplitError, make_grouped_split, write_split_manifest
from .inference import package_submission
from .submission import SubmissionError


def _stage_role_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage", required=True, choices=("preliminary", "second_round", "semifinal"))


def _audit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit one official ZIP without extraction")
    parser.add_argument("archive", type=Path)
    _stage_role_parser(parser)
    parser.add_argument("--role", required=True, choices=("train", "test"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-decode", action="store_true", help="check bytes/CRC only; not a formal image audit")
    parser.add_argument("--allow-truncated", action="store_true")
    return parser


def audit_main(argv: list[str] | None = None) -> int:
    args = _audit_parser().parse_args(argv)
    try:
        report = audit_archive(
            args.archive,
            stage=args.stage,
            role=args.role,
            decode=not args.no_decode,
            allow_truncated=args.allow_truncated,
        )
        write_audit_report(args.output, report)
    except (AuditError, OSError, ContractError) as exc:
        print(f"audit failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"complete": report.complete, "records": len(report.records), "failures": len(report.failures), "manifest_digest": report.manifest_digest}, sort_keys=True))
    return 0 if report.usable else 1


def _split_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a deterministic grouped train/dev/confirm split")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def split_main(argv: list[str] | None = None) -> int:
    args = _split_parser().parse_args(argv)
    try:
        records = load_manifest(args.manifest, require_complete=True)
        if any(record.decode_status != "decoded" for record in records):
            raise AuditError("formal split requires a complete decoded image audit; CRC-only checks cannot establish pixel groups")
        manifest = make_grouped_split(records, seed=args.seed)
        write_split_manifest(args.output, manifest)
        if args.report:
            write_json(args.report, manifest.report())
    except (AuditError, SplitError, OSError, ContractError) as exc:
        print(f"split failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest.report(), sort_keys=True))
    return 0


def class_map_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a fixed exact-ID class map from a complete train manifest")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        class_map = class_map_from_records(load_manifest(args.manifest, require_complete=True))
        write_json(args.output, class_map.to_dict())
    except (AuditError, ContractError, OSError) as exc:
        print(f"class-map failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(class_map.to_dict(), sort_keys=True))
    return 0


def _package_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and package one prediction CSV")
    parser.add_argument("csv", type=Path)
    parser.add_argument("zip", type=Path)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--expected-files", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def package_main(argv: list[str] | None = None) -> int:
    args = _package_parser().parse_args(argv)
    try:
        value = read_json(args.class_map)
        mapping = value.get("id_to_index", value)
        if not isinstance(mapping, dict):
            raise SubmissionError("class map JSON must contain id_to_index")
        class_map = ClassMap(stage=value.get("stage", "preliminary"), id_to_index=mapping)
        count = package_submission(
            args.csv,
            args.zip,
            class_map=class_map,
            expected_files=args.expected_files,
            overwrite=args.overwrite,
        )
    except (ContractError, SubmissionError, OSError, ValueError) as exc:
        print(f"package failed: {exc}", file=sys.stderr)
        return 1
    print(f"valid package: {count} prediction row(s)")
    return 0


def startup_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one finite synthetic baseline startup check")
    args = parser.parse_args(argv)
    del args
    try:
        from .training.startup import run_fixture_startup_check

        result = run_fixture_startup_check()
    except (RuntimeError, ValueError) as exc:
        print(f"startup check unavailable or failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0
