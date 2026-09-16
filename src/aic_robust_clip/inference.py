"""Single-model prediction and prediction-package generation."""

from __future__ import annotations

import csv
import os
import tempfile
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .contracts import ClassMap, ContractError
from .data.dataset import ManifestDataset, SampleItem
from .models.clip import ClipDependencyError, torch
from .submission import REQUIRED_CSV_NAME, SubmissionError, validate_submission


class InferenceError(RuntimeError):
    """Raised when prediction cannot be produced without fitting/adaptation."""


def _tensor_batch(images: Sequence[Any]) -> Any:
    if torch is None:
        raise ClipDependencyError("torch is required for prediction")
    if not images or not all(isinstance(image, torch.Tensor) for image in images):
        raise InferenceError("inference transform must return torch tensors")
    return torch.stack(list(images), dim=0)


def predict_dataset(
    model: Any,
    dataset: ManifestDataset,
    class_map: ClassMap,
    *,
    batch_size: int = 1,
    max_images: int | None = None,
    device: str | None = None,
) -> list[tuple[str, str]]:
    """Predict with exactly one model and one forward flow.

    The dataset must be an explicit test/inference dataset.  There is no
    optimizer, adaptation, label fitting, or test-time augmentation here.
    """

    if torch is None:
        raise ClipDependencyError("torch is required for prediction")
    if dataset.role != "test" or dataset.purpose != "inference":
        raise InferenceError("formal prediction requires a test inference dataset")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_images is not None and max_images <= 0:
        raise ValueError("max_images must be positive")
    if device is None:
        device = str(next(model.parameters()).device)
    model.eval()
    rows: list[tuple[str, str]] = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            if max_images is not None and len(rows) >= max_images:
                break
            stop = min(start + batch_size, len(dataset))
            if max_images is not None:
                stop = min(stop, start + max_images - len(rows))
            items = [dataset[index] for index in range(start, stop)]
            logits = model(_tensor_batch([item.image for item in items]).to(device))
            indices = logits.argmax(dim=-1).cpu().tolist()
            for item, index in zip(items, indices):
                try:
                    rows.append((item.sample_id, class_map.id_for(int(index))))
                except ContractError as exc:
                    raise InferenceError(str(exc)) from exc
    return rows


def records_to_filename_rows(dataset: ManifestDataset, predictions: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    by_id = {record.sample_id: record.member_path.rsplit("/", 1)[-1] for record in dataset.records}
    result: list[tuple[str, str]] = []
    for sample_id, class_id in predictions:
        if sample_id not in by_id:
            raise InferenceError(f"prediction sample ID is absent from the inference manifest: {sample_id}")
        result.append((by_id[sample_id], class_id))
    return result


def write_prediction_csv(
    path: Path | str,
    rows: Iterable[tuple[str, str]],
    *,
    class_map: ClassMap,
    expected_files: Path | str | None = None,
    overwrite: bool = False,
) -> int:
    destination = Path(path)
    if destination.name != REQUIRED_CSV_NAME:
        raise SubmissionError(f"prediction CSV must be named {REQUIRED_CSV_NAME}")
    if destination.exists() and not overwrite:
        raise SubmissionError(f"refusing to overwrite existing prediction: {destination}")
    materialized = list(rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x" if not overwrite else "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(materialized)
    try:
        return validate_submission(destination, expected_files, class_map.id_to_index)
    except Exception:
        # Leave the failed artifact for diagnosis; callers can choose a new
        # output namespace and no prediction is silently dropped.
        raise


def package_submission(
    csv_path: Path | str,
    zip_path: Path | str,
    *,
    class_map: ClassMap,
    expected_files: Path | str | None = None,
    overwrite: bool = False,
) -> int:
    csv_path = Path(csv_path)
    zip_path = Path(zip_path)
    if zip_path.exists() and not overwrite:
        raise SubmissionError(f"refusing to overwrite existing package: {zip_path}")
    count = validate_submission(csv_path, expected_files, class_map.id_to_index)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname=REQUIRED_CSV_NAME)
    if validate_submission(zip_path, expected_files, class_map.id_to_index) != count:
        raise SubmissionError("packaged prediction count changed")
    return count
