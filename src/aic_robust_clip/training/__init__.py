"""Bounded baseline training and checkpoint utilities."""

from .baseline import (
    BaselineResult,
    FrozenFeatureBaseline,
    OnlineFrozenBaseline,
    TrainConfig,
    train_baseline,
)
from .objectives import combined_wpi_loss, gce_loss, preservation_loss, sce_loss, supervised_loss
from .reliability import ReliabilityState, observed_prior, observed_prior_tensor
from .startup import run_fixture_startup_check

__all__ = [
    "BaselineResult",
    "FrozenFeatureBaseline",
    "OnlineFrozenBaseline",
    "TrainConfig",
    "train_baseline",
    "combined_wpi_loss",
    "gce_loss",
    "preservation_loss",
    "sce_loss",
    "supervised_loss",
    "ReliabilityState",
    "observed_prior",
    "observed_prior_tensor",
    "run_fixture_startup_check",
]
