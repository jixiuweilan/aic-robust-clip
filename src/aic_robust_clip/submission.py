"""Validate the organizer's prediction CSV and ZIP submission formats."""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Mapping, TextIO

LABEL_PATTERN = re.compile(r"^[0-9]{4}$")
REQUIRED_CSV_NAME = "pred_results.csv"


class SubmissionError(ValueError):
    """Raised when a prediction file violates the organizer's format."""


def _is_filename(value: str) -> bool:
    """Reject both POSIX and Windows path syntax, even on Linux."""

    return bool(value) and value not in {".", ".."} and "/" not in value and "\\" not in value


def _validate_rows(handle: TextIO, allowed_class_ids: set[str] | None = None) -> list[str]:
    filenames: list[str] = []
    seen: set[str] = set()

    for line_number, row in enumerate(csv.reader(handle), start=1):
        if len(row) != 2:
            raise SubmissionError(
                f"line {line_number}: expected 2 fields, found {len(row)}"
            )

        filename, label = row
        if filename != filename.strip() or label != label.strip():
            raise SubmissionError(
                f"line {line_number}: filename and class ID may not have surrounding whitespace"
            )
        if not filename:
            raise SubmissionError(f"line {line_number}: filename is empty")
        if not _is_filename(filename):
            raise SubmissionError(
                f"line {line_number}: expected a filename, not a path: {filename!r}"
            )
        if filename in seen:
            raise SubmissionError(
                f"line {line_number}: duplicate filename: {filename!r}"
            )
        if not LABEL_PATTERN.fullmatch(label):
            raise SubmissionError(
                f"line {line_number}: class ID must be exactly four digits: {label!r}"
            )
        if allowed_class_ids is not None and label not in allowed_class_ids:
            raise SubmissionError(
                f"line {line_number}: class ID is absent from the supplied class map: {label!r}"
            )

        seen.add(filename)
        filenames.append(filename)

    if not filenames:
        raise SubmissionError("prediction CSV is empty")
    return filenames


def _read_expected_files(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig") as handle:
        filenames = []
        for line_number, line in enumerate(handle, start=1):
            value = line.rstrip("\n\r")
            if value != value.strip():
                raise SubmissionError(f"expected-file line {line_number}: surrounding whitespace is invalid")
            if value:
                if not _is_filename(value):
                    raise SubmissionError(f"expected-file line {line_number}: not a filename: {value!r}")
                filenames.append(value)
    if not filenames:
        raise SubmissionError("expected-file list is empty")
    if len(filenames) != len(set(filenames)):
        raise SubmissionError("expected-file list contains duplicates")
    return filenames


def _compare_expected(actual: Iterable[str], expected: Iterable[str]) -> None:
    actual_set = set(actual)
    expected_set = set(expected)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing {len(missing)} file(s), first: {missing[0]!r}")
        if unexpected:
            details.append(
                f"unexpected {len(unexpected)} file(s), first: {unexpected[0]!r}"
            )
        raise SubmissionError("; ".join(details))


def validate_submission(
    path: Path | str,
    expected_files: Path | str | None = None,
    allowed_class_ids: Iterable[str] | Mapping[str, int] | None = None,
) -> int:
    """Validate a CSV or ZIP and return the number of prediction rows."""

    submission_path = Path(path)
    allowed = None if allowed_class_ids is None else set(allowed_class_ids)
    if allowed is not None:
        if not allowed or any(not isinstance(value, str) or not LABEL_PATTERN.fullmatch(value) for value in allowed):
            raise SubmissionError("allowed class IDs must be a non-empty set of four-digit strings")
    if not submission_path.is_file():
        raise SubmissionError(f"submission does not exist: {submission_path}")

    if submission_path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(submission_path) as archive:
                members = [item for item in archive.infolist() if not item.is_dir()]
                if [item.filename for item in members] != [REQUIRED_CSV_NAME]:
                    raise SubmissionError(
                        "ZIP must contain only pred_results.csv at its root"
                    )
                with archive.open(members[0]) as raw_handle:
                    with io.TextIOWrapper(raw_handle, encoding="utf-8-sig", newline="") as handle:
                        filenames = _validate_rows(handle, allowed)
        except zipfile.BadZipFile as exc:
            raise SubmissionError(f"invalid ZIP archive: {exc}") from exc
    elif submission_path.suffix.lower() == ".csv":
        if submission_path.name != REQUIRED_CSV_NAME:
            raise SubmissionError("CSV must be named pred_results.csv")
        with submission_path.open("r", encoding="utf-8-sig", newline="") as handle:
            filenames = _validate_rows(handle, allowed)
    else:
        raise SubmissionError("submission must be a .csv or .zip file")

    if expected_files is not None:
        expected = _read_expected_files(Path(expected_files))
        _compare_expected(filenames, expected)

    return len(filenames)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an AIC pred_results.csv file or submission ZIP."
    )
    parser.add_argument("submission", type=Path, help="CSV or ZIP to validate")
    parser.add_argument(
        "--expected-files",
        type=Path,
        help="optional UTF-8 file containing one expected image filename per line",
    )
    parser.add_argument(
        "--class-map",
        type=Path,
        help="optional JSON class map with an id_to_index object",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        allowed = None
        if args.class_map is not None:
            import json

            value = json.loads(args.class_map.read_text(encoding="utf-8"))
            allowed = value.get("id_to_index", value)
            if not isinstance(allowed, dict):
                raise SubmissionError("class map JSON must contain an id_to_index object")
        row_count = validate_submission(args.submission, args.expected_files, allowed)
    except (OSError, UnicodeError, SubmissionError) as exc:
        print(f"invalid submission: {exc}", file=sys.stderr)
        return 1

    print(f"valid submission: {row_count} prediction row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
