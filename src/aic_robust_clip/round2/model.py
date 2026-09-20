"""Explicit visual/head/adapter/auxiliary parameter ownership."""
import torch
from torch import nn
from ..models.classifier import LinearClassifier
from ..models.lora import inject_visual_qv_lora


class Student(nn.Module):
    def __init__(self, encoder, classes, adaptation):
        super().__init__()
        self.encoder = encoder
        self.adaptation = adaptation
        encoder.requires_grad_(False)
        if adaptation == "full_visual":
            encoder.clip_model.vision_model.requires_grad_(True)
            encoder.clip_model.visual_projection.requires_grad_(True)
        elif adaptation == "lora":
            inject_visual_qv_lora(encoder, rank=4, alpha=4)
        else:
            raise ValueError("unknown adaptation")
        self.classifier = LinearClassifier(encoder.output_dim, classes)

    def forward_with_features(self, images):
        features = self.encoder(images, no_grad=False)
        return self.classifier(features), features

    def forward(self, images):
        return self.forward_with_features(images)[0]


def optimizer_groups(student, auxiliary=None):
    groups, seen, expected = {}, set(), set()
    modules = [("student", student)] + ([("auxiliary", auxiliary)] if auxiliary is not None else [])
    for owner, model in modules:
        norms = {id(p) for module in model.modules() if isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.modules.batchnorm._BatchNorm))
                 for p in module.parameters(recurse=False)}
        for name, p in model.named_parameters(remove_duplicate=False):
            if not p.requires_grad:
                continue
            expected.add(id(p))
            if id(p) in seen:
                raise ValueError("duplicate optimizer parameter")
            seen.add(id(p))
            if owner == "auxiliary" and name.startswith(("stochastic.", "projector.")):
                family, lr = "auxiliary", 1e-3
            elif owner == "student" and name.startswith("classifier."):
                family, lr = "head", 1e-3
            elif owner == "student" and name.startswith("encoder.clip_model.vision_model.") and ("lora_A" in name or "lora_B" in name):
                family, lr = "lora", 1e-4
            elif owner == "student" and student.adaptation == "full_visual" and name.startswith(("encoder.clip_model.vision_model.", "encoder.clip_model.visual_projection.")):
                family, lr = "visual", 1e-5
            else:
                raise ValueError(f"unowned trainable parameter: {owner}.{name}")
            wd = 0. if name.endswith("bias") or id(p) in norms else 1e-4
            group = groups.setdefault((family, wd), {"params": [], "lr": lr, "initial_lr": lr,
                                                     "weight_decay": wd, "group_name": f"{family}:wd={wd}"})
            group["params"].append(p)
    if not expected or expected != seen:
        raise ValueError("optimizer coverage failure")
    return list(groups.values())
