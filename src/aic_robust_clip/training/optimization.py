"""Shared optimizer groups, update-based schedules and checkpoint ranking."""
from __future__ import annotations

import math

from ..models.clip import torch


def optimizer_groups(model, *, head_lr, lora_lr, weight_decay):
    norms = {id(parameter) for module in model.modules()
             if isinstance(module, (torch.nn.LayerNorm, torch.nn.modules.batchnorm._BatchNorm,
                                    torch.nn.GroupNorm)) for parameter in module.parameters(recurse=False)}
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        family = "lora" if "lora_A" in name or "lora_B" in name else "head"
        decay = 0.0 if name.endswith("bias") or id(parameter) in norms else weight_decay
        key = family, decay
        group = groups.setdefault(key, {"params": [], "lr": lora_lr if family == "lora" else head_lr,
                                       "weight_decay": decay, "group_name": f"{family}:wd={decay}"})
        group["params"].append(parameter)
    return list(groups.values())


def warmup_cosine_factor(update, *, warmup_updates, total_updates):
    """Scale for the NEXT update; initial update is index zero."""
    if update < warmup_updates:
        return (update + 1) / max(1, warmup_updates)
    progress = (update - warmup_updates) / max(1, total_updates - warmup_updates)
    return 0.5 * (1 + math.cos(math.pi * min(1., progress)))


def selection_key(metrics, epoch):
    return metrics.macro_recall, metrics.micro_top1, -epoch
