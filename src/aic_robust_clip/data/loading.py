"""Ordered batches with a delivered-only cursor and optional spawn prefetch."""

from __future__ import annotations

import math
from typing import Any, Sequence

from ..contracts import sha256_json
from ..runtime import stable_seeded_order
from .dataset import DatasetError, SampleItem


class _AddressedDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self._failed = False

    def __getitem__(self, address):
        if self._failed:
            raise DatasetError("worker stopped after an earlier read failure; no retry")
        epoch, index = address
        try:
            if hasattr(self.dataset, "set_epoch"):
                self.dataset.set_epoch(epoch)
            return self.dataset[index]
        except Exception:
            self._failed = True
            if hasattr(self.dataset, "close"):
                self.dataset.close()
            raise

    def __len__(self):
        return len(self.dataset)


class _RemainingSampler:
    def __init__(self, stream):
        self.stream = stream

    def __iter__(self):
        # Snapshot BEFORE prefetch starts; delivered position may advance later.
        stream = self.stream
        addresses = [(stream.epoch, index) for index in stream.order[stream.position:]]
        def remaining():
            for address in addresses:
                if stream._prefetch_stopping:
                    return
                yield address
        return remaining()

    def __len__(self):
        return len(self.stream.order) - self.stream.position


def _worker_init(_worker_id):
    import torch
    torch.set_num_threads(1)


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

    The optional DataLoader never owns the checkpoint cursor. In smoke mode
    callers MUST set max_samples and use zero workers before dataset access.
    """

    def __init__(self, dataset: Any, *, batch_size: int = 1, seed: int = 17,
                 shuffle: bool = True, max_samples: int | None = None,
                 sample_ids: Sequence[str] | None = None, num_workers: int = 0,
                 prefetch_factor: int = 2, pin_memory: bool = False) -> None:
        if batch_size <= 0 or (max_samples is not None and max_samples <= 0):
            raise DatasetError("batch size and sample bound must be positive")
        if (type(num_workers) is not int or not 0 <= num_workers <= 16
                or type(prefetch_factor) is not int or not 1 <= prefetch_factor <= 4
                or type(pin_memory) is not bool):
            raise DatasetError("invalid loader performance settings")
        if max_samples is not None and (num_workers or pin_memory):
            raise DatasetError("bounded smoke streams cannot prefetch or pin memory")
        self.num_workers, self.prefetch_factor, self.pin_memory = num_workers, prefetch_factor, pin_memory
        self._parallel = self._iterator = None
        self._prefetch_stopping = self._worker_failed = False
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
        if self._iterator is not None and not self.exhausted:
            self.close()  # discard outstanding prefetch on partial reset/restore
        self._iterator = None
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
        try:
            if self.num_workers:
                if self._parallel is None:
                    import torch
                    self._prefetch_stopping = False
                    self._parallel = torch.utils.data.DataLoader(
                        _AddressedDataset(self.dataset), batch_size=self.batch_size,
                        sampler=_RemainingSampler(self), collate_fn=collate_samples,
                        num_workers=self.num_workers, prefetch_factor=self.prefetch_factor,
                        pin_memory=self.pin_memory, persistent_workers=True,
                        multiprocessing_context="spawn", worker_init_fn=_worker_init,
                        generator=torch.Generator().manual_seed(self.seed), timeout=300)
                if self._iterator is None:
                    self._iterator = iter(self._parallel)
                batch = next(self._iterator)
            else:
                batch = collate_samples([self.dataset[index] for index in self.order[self.position:stop]])
                if self.pin_memory:
                    batch = {key: value.pin_memory() if hasattr(value, "pin_memory") else value
                             for key, value in batch.items()}
            if batch["sample_id"] != [self.ids[index] for index in self.order[self.position:stop]]:
                raise DatasetError("prefetch returned an unexpected sample order")
        except BaseException as exc:
            self._worker_failed = True
            try:
                self.close()
            except Exception as cleanup_error:
                if hasattr(exc, "add_note"):
                    exc.add_note(f"prefetch shutdown also failed: {cleanup_error}")
            raise
        self.position = stop
        return batch

    def close(self) -> None:
        iterator = self._iterator or getattr(self._parallel, "_iterator", None)
        workers = list(getattr(iterator, "_workers", ()))
        self._prefetch_stopping = True
        try:
            if iterator is not None:
                try:
                    if not self._worker_failed:
                        # Stop the sampler before draining. next() now consumes
                        # only already-dispatched work; it cannot refill queues.
                        # Keep consumers (including the pin thread) alive until
                        # worker tensor transfers finish. Do NOT advance the
                        # delivered/checkpoint cursor or run model computation.
                        for _ in range(self.num_workers * self.prefetch_factor + 1):
                            try:
                                next(iterator)
                            except StopIteration:
                                break
                        else:
                            raise DatasetError("prefetch shutdown exceeded its pending-batch bound")
                except BaseException as exc:
                    self._worker_failed = True
                    try:
                        iterator._shutdown_workers()
                    except Exception as cleanup_error:
                        if hasattr(exc, "add_note"):
                            exc.add_note(f"prefetch shutdown also failed: {cleanup_error}")
                    raise
                else:
                    # PyTorch 2.x has no public iterator close; isolate its use.
                    iterator._shutdown_workers()
        finally:
            self._iterator = self._parallel = None
            # Also reap workers terminated by PyTorch's exceptional shutdown.
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                worker.join(timeout=5)
                if worker.is_alive():
                    worker.kill()
                    worker.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, exc, _traceback):
        try:
            self.close()
        except Exception as cleanup_error:
            if exc is None:
                raise
            if hasattr(exc, "add_note"):
                exc.add_note(f"prefetch shutdown also failed: {cleanup_error}")

    def __del__(self):  # pragma: no cover - explicit close owns normal cleanup
        try:
            self.close()
        except Exception:
            pass

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
