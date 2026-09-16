"""Frozen-feature linear classifier used by the baseline recipes."""

from __future__ import annotations

from typing import Any

from .clip import ClipDependencyError, torch, nn


class ClassifierDependencyError(RuntimeError):
    pass


class LinearClassifier(nn.Module):  # type: ignore[misc]
    def __init__(self, feature_dim: int, class_count: int) -> None:
        if torch is None:
            raise ClassifierDependencyError("torch is required for the classifier")
        if feature_dim <= 0 or class_count <= 0:
            raise ValueError("feature_dim and class_count must be positive")
        super().__init__()
        self.linear = nn.Linear(feature_dim, class_count)

    def forward(self, features: Any) -> Any:
        return self.linear(features)

    @property
    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)


def normalized_features(encoder: Any, pixel_values: Any) -> Any:
    features = encoder(pixel_values)
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
