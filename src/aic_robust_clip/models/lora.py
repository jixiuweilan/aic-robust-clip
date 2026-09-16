"""Visual Q/V-only LoRA for the permitted CLIP image encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .clip import ClipDependencyError, FrozenCLIPEncoder, torch, nn
from .classifier import LinearClassifier


class LoRAError(ValueError):
    """Raised when a LoRA injection would alter an unintended parameter set."""


class LoRALinear(nn.Module):  # type: ignore[misc]
    """A frozen linear layer plus a zero-at-start low-rank residual."""

    def __init__(self, base: Any, *, rank: int, alpha: float, dropout: float = 0.0) -> None:
        if torch is None:
            raise ClipDependencyError("torch is required for LoRA")
        if not isinstance(base, nn.Linear):
            raise LoRAError("LoRA target must be nn.Linear")
        if rank <= 0 or alpha <= 0 or not 0 <= dropout < 1:
            raise LoRAError("rank and alpha must be positive; dropout must be in [0, 1)")
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Any) -> Any:
        residual = self.dropout(inputs) @ self.lora_A.transpose(0, 1)
        residual = residual @ self.lora_B.transpose(0, 1)
        return self.base(inputs) + residual * self.scaling


@dataclass(frozen=True)
class LoRAInjection:
    module_name: str
    rank: int
    alpha: float


def _vision_layers(encoder: FrozenCLIPEncoder) -> Any:
    model = encoder.clip_model
    vision_model = getattr(model, "vision_model", None)
    layers = getattr(getattr(vision_model, "encoder", None), "layers", None)
    if layers is None:
        raise LoRAError("loaded CLIP does not expose vision_model.encoder.layers")
    return layers


def inject_visual_qv_lora(
    encoder: FrozenCLIPEncoder,
    *,
    rank: int = 4,
    alpha: float = 4.0,
    dropout: float = 0.0,
) -> tuple[LoRAInjection, ...]:
    """Replace only visual attention q_proj and v_proj modules.

    Hugging Face CLIP exposes separate Q/K/V projections.  Refuse an unknown
    layout instead of guessing slices and potentially updating K.
    """

    if torch is None:
        raise ClipDependencyError("torch is required for LoRA")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    injections: list[LoRAInjection] = []
    for layer_index, layer in enumerate(_vision_layers(encoder)):
        attention = getattr(layer, "self_attn", None)
        if attention is None or not all(hasattr(attention, name) for name in ("q_proj", "k_proj", "v_proj")):
            raise LoRAError("CLIP attention must expose separate q_proj, k_proj, and v_proj modules")
        for projection_name in ("q_proj", "v_proj"):
            base = getattr(attention, projection_name)
            if isinstance(base, LoRALinear):
                raise LoRAError(f"LoRA already injected at layer {layer_index} {projection_name}")
            wrapped = LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout)
            setattr(attention, projection_name, wrapped)
            injections.append(LoRAInjection(f"vision_model.encoder.layers.{layer_index}.self_attn.{projection_name}", rank, alpha))
        for parameter in attention.k_proj.parameters():
            if parameter.requires_grad:
                raise LoRAError(f"K projection became trainable at layer {layer_index}")
    if not injections:
        raise LoRAError("CLIP vision encoder contains no attention layers")
    return tuple(injections)


class VisualLoRAModel(nn.Module):  # type: ignore[misc]
    """B03 model: visual Q/V LoRA plus the same linear classifier as B04."""

    def __init__(self, encoder: FrozenCLIPEncoder, class_count: int, *, inject: bool = True, rank: int = 4, alpha: float = 4.0) -> None:
        if torch is None:
            raise ClipDependencyError("torch is required for LoRA")
        super().__init__()
        self.encoder = encoder
        if inject:
            self.injections = inject_visual_qv_lora(encoder, rank=rank, alpha=alpha)
        else:
            self.injections = tuple()
        self.classifier = LinearClassifier(encoder.output_dim, class_count)

    def forward(self, pixel_values: Any) -> Any:
        return self.forward_with_features(pixel_values)[0]

    def forward_with_features(self, pixel_values: Any) -> tuple[Any, Any]:
        # no_grad=False is required for gradients to reach the LoRA residuals;
        # original CLIP weights remain frozen by requires_grad=False.
        features = self.encoder(pixel_values, no_grad=False)
        return self.classifier(features), features

    @property
    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)

    def assert_qv_only(self) -> None:
        names = self.trainable_parameter_names
        bad = [name for name in names if not ("lora_A" in name or "lora_B" in name or name.startswith("classifier."))]
        if bad:
            raise LoRAError(f"unexpected trainable visual parameters: {bad[:5]}")
