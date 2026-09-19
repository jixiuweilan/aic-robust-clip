"""Training-only reliability history and observed-prior helpers."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from ..models.clip import ClipDependencyError, torch


class ReliabilityError(ValueError):
    pass


def _ids_digest(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256("\0".join(sample_ids).encode("utf-8")).hexdigest()


@dataclass
class ReliabilityState:
    """Detached per-sample EMA losses for W, keyed by stable train IDs."""

    sample_ids: tuple[str, ...]
    observed_class_ids: Mapping[str, str]
    beta: float = 0.7
    w_min: float = 0.2
    warmup_epochs: int = 2
    ema_losses: dict[str, float] | None = None
    weights: dict[str, float] | None = None
    completed_epochs: int = 0

    def __post_init__(self) -> None:
        if not self.sample_ids or len(self.sample_ids) != len(set(self.sample_ids)):
            raise ReliabilityError("sample IDs must be unique and non-empty")
        if not 0 <= self.beta < 1 or not 0 <= self.w_min <= 1 or self.warmup_epochs < 0:
            raise ReliabilityError("invalid reliability hyperparameters")
        if set(self.observed_class_ids) != set(self.sample_ids):
            raise ReliabilityError("reliability class IDs must cover exactly the training IDs")
        if self.ema_losses is None:
            self.ema_losses = {}
        if self.weights is None:
            self.weights = {sample_id: 1.0 for sample_id in self.sample_ids}
        self._assert_train_ids(self.ema_losses)
        self._assert_train_ids(self.weights)

    def _assert_train_ids(self, values: Mapping[str, float]) -> None:
        unknown = set(values) - set(self.sample_ids)
        if unknown:
            raise ReliabilityError(f"reliability history contains non-training IDs: {sorted(unknown)[:3]}")

    def update_losses(self, losses: Mapping[str, float]) -> None:
        """EMA-update only training IDs after a complete scoring pass."""

        self._assert_train_ids(losses)
        if set(losses) != set(self.sample_ids):
            raise ReliabilityError("reliability scoring must cover every training sample")
        assert self.ema_losses is not None
        for sample_id in self.sample_ids:
            loss = float(losses[sample_id])
            if not math.isfinite(loss) or loss < 0:
                raise ReliabilityError(f"invalid loss for {sample_id}: {loss}")
            if sample_id not in self.ema_losses:
                self.ema_losses[sample_id] = loss
            else:
                self.ema_losses[sample_id] = self.beta * self.ema_losses[sample_id] + (1 - self.beta) * loss

    def _classwise_weights(self) -> dict[str, float]:
        assert self.ema_losses is not None
        by_class: dict[str, list[str]] = {}
        for sample_id in self.sample_ids:
            by_class.setdefault(self.observed_class_ids[sample_id], []).append(sample_id)
        result = {sample_id: 1.0 for sample_id in self.sample_ids}
        for members in by_class.values():
            if len(members) < 5:
                continue
            values = sorted((self.ema_losses[sample_id], sample_id) for sample_id in members)
            if values[0][0] == values[-1][0]:
                continue
            positions: dict[str, float] = {}
            cursor = 0
            while cursor < len(values):
                end = cursor + 1
                while end < len(values) and values[end][0] == values[cursor][0]:
                    end += 1
                # Midrank on the zero-based [0, n-1] scale.
                rank = (cursor + end - 1) / 2
                for _, sample_id in values[cursor:end]:
                    positions[sample_id] = rank / (len(values) - 1)
                cursor = end
            for sample_id, rank in positions.items():
                result[sample_id] = self.w_min + (1 - self.w_min) * (1 - rank)
        return result

    def next_epoch_weights(self, *, completed_epochs: int | None = None) -> dict[str, float]:
        """Return detached weights for the next epoch after scoring."""

        if completed_epochs is None:
            completed_epochs = self.completed_epochs
        if completed_epochs < self.warmup_epochs or self.ema_losses is None or len(self.ema_losses) != len(self.sample_ids):
            self.weights = {sample_id: 1.0 for sample_id in self.sample_ids}
        else:
            self.weights = self._classwise_weights()
        return dict(self.weights)

    def finish_epoch(self, losses: Mapping[str, float]) -> dict[str, float]:
        self.update_losses(losses)
        self.completed_epochs += 1
        return self.next_epoch_weights()

    def effective_sample_size(self, class_id: str) -> float:
        weights = [weight for sample_id, weight in (self.weights or {}).items() if self.observed_class_ids[sample_id] == class_id]
        if not weights or sum(value * value for value in weights) == 0:
            return 0.0
        return sum(weights) ** 2 / sum(value * value for value in weights)

    def diagnostics(self) -> dict[str, dict[str, float]]:
        """One linear pass, not one scan of all training IDs for every class."""
        result = {}
        for sample_id, weight in (self.weights or {}).items():
            row = result.setdefault(self.observed_class_ids[sample_id],
                                    {"count": 0, "weight_sum": 0., "weight_square_sum": 0.})
            row["count"] += 1
            row["weight_sum"] += weight
            row["weight_square_sum"] += weight * weight
        for row in result.values():
            row["ess"] = row["weight_sum"] ** 2 / max(row["weight_square_sum"], 1e-12)
        return result

    def state_dict(self) -> dict[str, object]:
        return {
            "sample_ids": list(self.sample_ids),
            "sample_ids_digest": _ids_digest(self.sample_ids),
            "observed_class_ids": dict(self.observed_class_ids),
            "beta": self.beta,
            "w_min": self.w_min,
            "warmup_epochs": self.warmup_epochs,
            "ema_losses": dict(self.ema_losses or {}),
            "weights": dict(self.weights or {}),
            "completed_epochs": self.completed_epochs,
        }

    @classmethod
    def from_state_dict(cls, value: Mapping[str, object]) -> "ReliabilityState":
        sample_ids = tuple(str(item) for item in value["sample_ids"])
        if value.get("sample_ids_digest") != _ids_digest(sample_ids):
            raise ReliabilityError("reliability sample-ID mapping digest mismatch")
        return cls(
            sample_ids=sample_ids,
            observed_class_ids={str(key): str(item) for key, item in dict(value["observed_class_ids"]).items()},
            beta=float(value["beta"]),
            w_min=float(value["w_min"]),
            warmup_epochs=int(value["warmup_epochs"]),
            ema_losses={str(key): float(item) for key, item in dict(value.get("ema_losses", {})).items()},
            weights={str(key): float(item) for key, item in dict(value.get("weights", {})).items()},
            completed_epochs=int(value.get("completed_epochs", 0)),
        )


def observed_prior(labels: Iterable[int], class_count: int, *, laplace: float = 1.0) -> list[float]:
    """Return the frozen observed-label prior from training labels only."""

    if class_count <= 0 or laplace <= 0:
        raise ReliabilityError("class_count and Laplace smoothing must be positive")
    counts = [0] * class_count
    for label in labels:
        if not 0 <= int(label) < class_count:
            raise ReliabilityError(f"label is outside the class map: {label}")
        counts[int(label)] += 1
    total = sum(counts)
    denominator = total + laplace * class_count
    return [(count + laplace) / denominator for count in counts]


def observed_prior_tensor(labels: Iterable[int], class_count: int, *, laplace: float = 1.0) -> object:
    if torch is None:
        raise ClipDependencyError("torch is required for tensor priors")
    return torch.tensor(observed_prior(labels, class_count, laplace=laplace), dtype=torch.float32)
