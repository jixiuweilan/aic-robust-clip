"""Lineage-complete checkpoint persistence and compatibility checks."""

from __future__ import annotations

import hashlib
import random
import os
import tempfile
import shutil
from pathlib import Path
from typing import Any

from ..contracts import CheckpointMetadata, ContractError
from ..models.clip import ClipDependencyError, torch


def publish_best(directory):
    """Atomically alias the committed last snapshot, without serializing twice.

    Replacing last.pt on the next epoch creates a new inode; the best snapshot
    stays immutable. Filesystems without hard links use an atomic copy.
    """
    root = Path(directory)
    descriptor, name = tempfile.mkstemp(dir=root, prefix="best.pt.")
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.unlink()
        try:
            os.link(root / "last.pt", temporary)
        except OSError:
            shutil.copyfile(root / "last.pt", temporary)
        os.replace(temporary, root / "best.pt")
    finally:
        temporary.unlink(missing_ok=True)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate()}
    if torch is not None:
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np  # type: ignore

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if torch is not None and "torch" in state:
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])
    if "numpy" in state:
        try:
            import numpy as np  # type: ignore

            np.random.set_state(state["numpy"])
        except ImportError:
            pass


def save_checkpoint(
    path: Path | str,
    *,
    model: Any,
    metadata: CheckpointMetadata,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    sampler_state: Any | None = None,
    module_state: dict[str, Any] | None = None,
) -> str:
    if torch is None:
        raise ClipDependencyError("torch is required for checkpoint persistence")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": metadata.schema_version,
        "metadata": metadata.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "sampler_state": sampler_state,
        "module_state": module_state if module_state is not None else dict(metadata.module_state),
        "rng_state": _rng_state(),
    }
    # Replace only after serialization succeeds: interruption must not destroy
    # the previous resumable checkpoint.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    with destination.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_checkpoint(
    path: Path | str,
    *,
    model: Any,
    requested: dict[str, str],
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    restore_rng: bool = True,
    expected_configuration_digest: str | None = None,
) -> dict[str, Any]:
    """Load only a checkpoint matching every stage/lineage identity."""

    if torch is None:
        raise ClipDependencyError("torch is required for checkpoint persistence")
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:  # compatibility with older torch releases
        payload = torch.load(Path(path), map_location="cpu")
    metadata_value = dict(payload.get("metadata", {}))
    metadata_value.pop("schema_version", None)
    metadata = CheckpointMetadata(**metadata_value)
    if metadata.model_family == "BENCHMARK":
        raise ContractError("benchmark checkpoints cannot be loaded for training or inference")
    if expected_configuration_digest is not None and metadata.configuration_digest != expected_configuration_digest:
        raise ContractError("resume configuration digest mismatch")
    try:
        metadata.assert_compatible(**requested)
    except TypeError as exc:
        raise ContractError("requested checkpoint compatibility fields are incomplete") from exc
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        if payload.get("optimizer_state") is None:
            raise ContractError("checkpoint has no optimizer state for resume")
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None:
        if payload.get("scheduler_state") is None:
            raise ContractError("checkpoint has no scheduler state for resume")
        scheduler.load_state_dict(payload["scheduler_state"])
    if restore_rng and payload.get("rng_state") is not None:
        _restore_rng_state(payload["rng_state"])
    return {
        "metadata": metadata,
        "module_state": payload.get("module_state", {}),
        "sampler_state": payload.get("sampler_state"),
    }
