"""Standalone robust-loss and W/P/I objective primitives from the design."""

from __future__ import annotations

from typing import Any

from ..models.clip import ClipDependencyError, torch


class ObjectiveError(ValueError):
    pass


def _require_torch() -> Any:
    if torch is None:
        raise ClipDependencyError("torch is required for training objectives")
    return torch


def gce_loss(logits: Any, labels: Any, *, q: float = 0.7, reduction: str = "mean") -> Any:
    """Generalized cross entropy, with the q->0 cross-entropy limit."""

    torch_module = _require_torch()
    if q < 0:
        raise ObjectiveError("GCE q must be non-negative")
    log_probability = torch_module.log_softmax(logits, dim=-1)
    target_log_probability = log_probability.gather(1, labels.reshape(-1, 1)).squeeze(1)
    if q == 0:
        losses = -target_log_probability
    else:
        losses = -torch_module.expm1(q * target_log_probability) / q
    if reduction == "none":
        return losses
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    raise ObjectiveError(f"unsupported reduction: {reduction!r}")


def sce_loss(
    logits: Any,
    labels: Any,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    clip_probability: float = 1e-4,
    reduction: str = "mean",
) -> Any:
    """Symmetric CE with stable forward CE and clipped reverse CE."""

    torch_module = _require_torch()
    if alpha < 0 or beta < 0 or not 0 < clip_probability < 1:
        raise ObjectiveError("invalid SCE coefficients or probability clip")
    labels = labels.reshape(-1).long()
    probabilities = torch_module.softmax(logits, dim=-1)
    one_hot = torch_module.nn.functional.one_hot(labels, num_classes=logits.shape[-1]).to(logits.dtype)
    forward = torch_module.nn.functional.cross_entropy(logits, labels, reduction="none")
    reverse = -(probabilities * torch_module.log(one_hot.clamp_min(clip_probability))).sum(dim=-1)
    losses = alpha * forward + beta * reverse
    if reduction == "none":
        return losses
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    raise ObjectiveError(f"unsupported reduction: {reduction!r}")


def preservation_loss(adapted_features: Any, frozen_features: Any) -> Any:
    """Mean 1-cosine distance to a detached original-encoder feature."""

    torch_module = _require_torch()
    adapted = adapted_features / adapted_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    frozen = frozen_features.detach()
    frozen = frozen / frozen.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (1.0 - (adapted * frozen).sum(dim=-1)).mean()


def supervised_loss(
    logits: Any,
    labels: Any,
    *,
    sample_weights: Any | None = None,
    observed_prior: Any | None = None,
    tau: float = 0.0,
) -> Any:
    """W/P/I supervised term; weights and prior are detached inputs."""

    torch_module = _require_torch()
    adjusted = logits
    if observed_prior is not None and tau != 0:
        prior = observed_prior if hasattr(observed_prior, "to") else torch_module.as_tensor(observed_prior)
        prior = prior.to(device=logits.device, dtype=logits.dtype).clamp_min(1e-12)
        adjusted = logits + tau * torch_module.log(prior).reshape(1, -1)
    losses = torch_module.nn.functional.cross_entropy(adjusted, labels.reshape(-1).long(), reduction="none")
    if sample_weights is None:
        return losses.mean()
    weights = sample_weights if hasattr(sample_weights, "detach") else torch_module.as_tensor(sample_weights)
    weights = weights.detach().to(device=losses.device, dtype=losses.dtype).reshape(-1)
    if weights.shape != losses.shape or torch_module.any(weights < 0):
        raise ObjectiveError("sample weights must be non-negative and match the batch")
    denominator = weights.sum().clamp_min(1e-12)
    return (losses * weights).sum() / denominator


def elr_loss(logits: Any, labels: Any, target_history: Any, *, lambda_elr: float = 1.0) -> Any:
    """Basic single-network ELR term using a detached normalized target history."""

    torch_module = _require_torch()
    if lambda_elr < 0:
        raise ObjectiveError("lambda_elr must be non-negative")
    probabilities = torch_module.softmax(logits, dim=-1)
    target = target_history.detach().to(device=logits.device, dtype=logits.dtype)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    regularizer = torch_module.log((1.0 - (probabilities * target).sum(dim=-1)).clamp_min(1e-7))
    return torch_module.nn.functional.cross_entropy(logits, labels.reshape(-1).long()) + lambda_elr * regularizer.mean()


def combined_wpi_loss(
    logits: Any,
    labels: Any,
    *,
    sample_weights: Any | None = None,
    prior: Any | None = None,
    tau: float = 0.0,
    adapted_features: Any | None = None,
    frozen_features: Any | None = None,
    lambda_preserve: float = 0.0,
) -> Any:
    """Explicit composition; all modules are off when their inputs are absent."""

    if lambda_preserve < 0:
        raise ObjectiveError("lambda_preserve must be non-negative")
    loss = supervised_loss(logits, labels, sample_weights=sample_weights, observed_prior=prior, tau=tau)
    if adapted_features is not None or frozen_features is not None:
        if adapted_features is None or frozen_features is None:
            raise ObjectiveError("preservation requires both adapted and frozen features")
        loss = loss + lambda_preserve * preservation_loss(adapted_features, frozen_features)
    return loss
