"""One finite synthetic startup check for the local development machine."""

from __future__ import annotations

from typing import Any

from ..contracts import RunConfig
from ..models.clip import ClipDependencyError, torch
from ..runtime import resolve_run_config
from .baseline import FrozenFeatureBaseline, TrainConfig, train_baseline


def run_fixture_startup_check(*, run: RunConfig | None = None) -> dict[str, Any]:
    """Verify data iteration, forward/backward, and an optimizer update.

    This intentionally uses generated tensors and never touches competition
    archives.  The resolved limits are hard-coded by the RunConfig contract;
    the function has no path into formal training.
    """

    if torch is None:
        raise ClipDependencyError("install the training extra before the startup check")
    run = run or RunConfig(stage="preliminary", execution_mode="smoke")
    if run.execution_mode != "smoke":
        raise ValueError("fixture startup check requires smoke mode")
    resolve_run_config(run)
    torch.manual_seed(run.seed)
    features = torch.randn(run.max_samples, 8)
    labels = torch.arange(run.max_samples, dtype=torch.long) % 3
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(features, labels), batch_size=run.batch_size)
    model = FrozenFeatureBaseline(feature_dim=8, class_count=3)
    result = train_baseline(
        model,
        loader,
        config=TrainConfig(run=run, epochs=1),
        class_count=3,
    )
    if not result.optimizer_updated:
        raise RuntimeError("startup check did not perform an optimizer update")
    return result.to_dict()
