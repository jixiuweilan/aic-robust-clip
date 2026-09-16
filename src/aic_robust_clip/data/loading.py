"""Tensor collation and a resumable, zero-worker manifest batch stream.

Resume deliberately uses this stream, not DataLoader prefetch/replay: its
cursor advances only for delivered samples and iterator creation draws no RNG.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from ..contracts import sha256_json
from ..runtime import stable_seeded_order
from .dataset import DatasetError, SampleItem


def collate_samples(items: Sequence[SampleItem]) -> dict[str, Any]:
    import torch

    if not items or not all(isinstance(item, SampleItem) for item in items):
        raise DatasetError("collation requires non-empty SampleItem batches")
    if not all(isinstance(item.image, torch.Tensor) for item in items):
        raise DatasetError("image transform must return a torch tensor before collation")
    labels = [item.label_index for item in items]
    if any(label is None for label in labels) and not all(label is None for label in labels):
        raise DatasetError("cannot collate mixed labelled and unlabelled samples")
    return {
        "image": torch.stack([item.image for item in items]),
        "sample_id": [item.sample_id for item in items],
        "class_id": [item.class_id for item in items],
        "label_index": None if labels[0] is None else torch.tensor(labels, dtype=torch.long),
    }


class StatefulBatchLoader:
    """Map-style dataset batching with exact deterministic cursor restoration.

    Use ``collate_samples`` with a standard DataLoader for non-resumable use.
    This implementation intentionally has no workers/prefetch. In smoke mode
    callers MUST set max_samples before any dataset access.
    """

    def __init__(self, dataset: Any, *, batch_size: int = 1, seed: int = 17,
                 shuffle: bool = True, max_samples: int | None = None,
                 sample_ids: Sequence[str] | None = None) -> None:
        if batch_size <= 0 or (max_samples is not None and max_samples <= 0):
            raise DatasetError("batch size and sample bound must be positive")
        self.dataset = dataset
        records = getattr(dataset, "records", None)
        ids = list(sample_ids) if sample_ids is not None else [record.sample_id for record in records or ()]
        if len(ids) != len(dataset) or len(ids) != len(set(ids)) or not ids:
            raise DatasetError("loader needs one unique stable ID per dataset row")
        self.identity = sha256_json([record.to_dict() for record in records] if records is not None else ids)
        self.ids = ids
        self.batch_size, self.seed, self.shuffle = batch_size, seed, shuffle
        self.max_samples = max_samples
        self.epoch = 0
        self.position = 0
        self._set_order()

    def _set_order(self) -> None:
        ids = stable_seeded_order(self.ids, self.seed + self.epoch) if self.shuffle else self.ids
        ids = ids[:self.max_samples]
        by_id = {sample_id: index for index, sample_id in enumerate(self.ids)}
        self.order = [by_id[sample_id] for sample_id in ids]

    def reset(self, epoch: int) -> None:
        if epoch < 0:
            raise DatasetError("epoch must be non-negative")
        self.epoch, self.position = epoch, 0
        self._set_order()
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def __len__(self) -> int:
        return math.ceil(len(self.order) / self.batch_size)

    def __iter__(self) -> "StatefulBatchLoader":
        return self

    def __next__(self) -> dict[str, Any]:
        if self.exhausted:
            raise StopIteration
        stop = min(self.position + self.batch_size, len(self.order))
        batch = collate_samples([self.dataset[index] for index in self.order[self.position:stop]])
        self.position = stop
        return batch

    @property
    def exhausted(self) -> bool:
        return self.position == len(self.order)

    def state_dict(self) -> dict[str, Any]:
        return {"identity": self.identity, "batch_size": self.batch_size,
                "seed": self.seed, "shuffle": self.shuffle, "max_samples": self.max_samples,
                "epoch": self.epoch, "position": self.position}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key in ("identity", "batch_size", "seed", "shuffle", "max_samples"):
            if state.get(key) != getattr(self, key):
                raise DatasetError(f"resume loader mismatch: {key}")
        epoch, position = state["epoch"], state["position"]
        if not isinstance(epoch, int) or epoch < 0 or not isinstance(position, int):
            raise DatasetError("invalid loader resume cursor")
        self.reset(epoch)
        if not 0 <= position <= len(self.order):
            raise DatasetError("loader resume cursor is out of bounds")
        self.position = position
