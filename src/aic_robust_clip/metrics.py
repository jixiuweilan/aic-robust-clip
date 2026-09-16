"""Metrics used for provisional noisy-label validation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ClassificationMetrics:
    micro_top1: float
    macro_recall: float
    class_recall: Mapping[int, float]
    class_support: Mapping[int, int]
    present_classes: int
    total_classes: int
    groups: Mapping[str, float]
    group_support: Mapping[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "micro_top1": self.micro_top1,
            "macro_recall": self.macro_recall,
            "class_recall": {str(key): value for key, value in sorted(self.class_recall.items())},
            "class_support": {str(key): value for key, value in sorted(self.class_support.items())},
            "present_classes": self.present_classes,
            "total_classes": self.total_classes,
            "groups": dict(self.groups),
            "group_support": dict(self.group_support),
        }


def class_frequency_groups(training_counts: Mapping[int, int], total_classes: int | None = None) -> dict[str, frozenset[int]]:
    """Freeze deterministic head/mid/tail groups from training counts."""

    if total_classes is None:
        total_classes = len(training_counts)
    classes = sorted(training_counts, key=lambda class_id: (-training_counts[class_id], class_id))
    # For incomplete class maps, still expose missing classes in the tail
    # diagnostics only when they are explicitly part of total_classes.
    if len(classes) != total_classes:
        classes = sorted(set(classes))
    base, remainder = divmod(len(classes), 3)
    sizes = [base + (index < remainder) for index in range(3)]
    groups: dict[str, frozenset[int]] = {}
    cursor = 0
    for name, size in zip(("head", "mid", "tail"), sizes):
        groups[name] = frozenset(classes[cursor : cursor + size])
        cursor += size
    return groups


def evaluate_classification(
    labels: Sequence[int] | Iterable[int],
    predictions: Sequence[int] | Iterable[int],
    *,
    total_classes: int,
    training_counts: Mapping[int, int] | None = None,
) -> ClassificationMetrics:
    labels = list(labels)
    predictions = list(predictions)
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have equal lengths")
    if not labels:
        raise ValueError("cannot evaluate an empty prediction set")
    if total_classes <= 0:
        raise ValueError("total_classes must be positive")
    support = Counter(labels)
    correct = Counter(label for label, prediction in zip(labels, predictions) if label == prediction)
    recalls = {class_id: correct[class_id] / count for class_id, count in sorted(support.items())}
    groups = class_frequency_groups(training_counts or dict(support), total_classes)
    group_values: dict[str, float] = {}
    group_support: dict[str, int] = {}
    for name, class_ids in groups.items():
        present = [recalls[class_id] for class_id in class_ids if class_id in recalls]
        group_values[name] = sum(present) / len(present) if present else float("nan")
        group_support[name] = sum(support[class_id] for class_id in class_ids if class_id in support)
    return ClassificationMetrics(
        micro_top1=sum(label == prediction for label, prediction in zip(labels, predictions)) / len(labels),
        macro_recall=sum(recalls.values()) / len(recalls),
        class_recall=recalls,
        class_support=dict(sorted(support.items())),
        present_classes=len(recalls),
        total_classes=total_classes,
        groups=group_values,
        group_support=group_support,
    )
