"""Execution guards and deterministic runtime helpers.

The current development machine is intentionally a smoke-test machine.  A
formal profile must be explicitly run on the separate training machine and
cannot be enabled by merely changing a batch size here.
"""

from __future__ import annotations

import hashlib
import os
import random
from itertools import islice
from dataclasses import dataclass
from typing import Iterable, Sequence, TypeVar

from .contracts import ContractError, RunConfig


class RuntimeLimitError(RuntimeError):
    """Raised when an execution request exceeds the local bounded policy."""


@dataclass(frozen=True)
class RuntimePolicy:
    machine: str = "local-development"
    allow_formal: bool = False
    default_mode: str = "smoke"
    max_smoke_updates: int = 3
    max_smoke_samples: int = 8
    max_smoke_eval_batches: int = 2

    def validate(self, config: RunConfig) -> None:
        if config.execution_mode == "formal":
            from .environment import DEVELOPMENT_HOST, machine_fingerprint
            if machine_fingerprint() == DEVELOPMENT_HOST:
                raise RuntimeLimitError("formal execution is forbidden on this development host")
        if config.execution_mode == "formal" and not self.allow_formal:
            raise RuntimeLimitError(
                "formal execution is disabled on this machine; use a separate "
                "training machine with allow_formal=True"
            )
        if config.execution_mode == "smoke":
            if config.max_updates > self.max_smoke_updates:
                raise RuntimeLimitError("smoke update limit exceeds local policy")
            if config.max_samples > self.max_smoke_samples:
                raise RuntimeLimitError("smoke sample limit exceeds local policy")
            if config.max_eval_batches > self.max_smoke_eval_batches:
                raise RuntimeLimitError("smoke evaluation limit exceeds local policy")


LOCAL_POLICY = RuntimePolicy()


def resolve_run_config(config: RunConfig, policy: RuntimePolicy = LOCAL_POLICY) -> RunConfig:
    """Validate a resolved configuration before any data/model loading."""

    if config.execution_mode not in ("smoke", "formal"):
        raise ContractError(f"unsupported execution mode: {config.execution_mode!r}")
    policy.validate(config)
    return config


T = TypeVar("T")


def bounded_items(items: Iterable[T], *, limit: int, name: str = "items") -> list[T]:
    """Materialise at most ``limit`` items and reject an invalid bound."""

    if limit <= 0:
        raise RuntimeLimitError(f"{name} limit must be positive")
    return list(islice(items, limit))


def seed_everything(seed: int) -> None:
    """Seed Python and optional numerical/torch libraries without requiring them."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def stable_seeded_order(sample_ids: Sequence[str], seed: int) -> list[str]:
    """Order IDs without depending on filesystem/archive discovery order."""

    return sorted(sample_ids, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())


def assert_bounded_startup(*, updates: int, samples: int, config: RunConfig) -> None:
    """Assert a startup check actually stopped within its declared bounds."""

    if config.execution_mode != "smoke":
        raise RuntimeLimitError("startup assertions only apply to smoke mode")
    if updates <= 0:
        raise RuntimeLimitError("startup check did not perform an optimizer update")
    if updates > config.max_updates or samples > config.max_samples:
        raise RuntimeLimitError("startup check exceeded its resolved limits")


def current_code_revision() -> str:
    """Return a caller-supplied revision, or an explicit local placeholder."""

    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    # A source digest covers uncommitted/untracked implementation files too.
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return f"{revision or 'unversioned'}+src.{digest.hexdigest()}"
