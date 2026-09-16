"""Partition-bound, sharded fixed-view feature caches with lazy verification."""
from __future__ import annotations

from pathlib import Path

from ..contracts import read_json, sha256_json, write_json
from ..models.clip import torch
from ..models.provision import file_sha256
from .dataset import DatasetError, SampleItem
from .loading import StatefulBatchLoader


def cache_key(run, weight_digest, preprocessing_digest, partition):
    return {"schema_version": 2, "stage": run.stage, "execution_mode": run.execution_mode,
            "manifest_digest": run.manifest_digest, "split_digest": run.split_digest,
            "class_map_digest": run.class_map_digest, "weight_digest": weight_digest,
            "preprocessing_digest": preprocessing_digest, "partition": partition}


def generate_cache(dataset, encoder, directory, *, key, device, batch_size=1, shard_rows=1024):
    if dataset.role != "train" or dataset.partition not in {"train", "dev"} or key["partition"] != dataset.partition:
        raise DatasetError("feature caches accept train/dev only, never confirm/test")
    if getattr(dataset.transform, "online", False):
        raise DatasetError("random online views cannot be cached as fixed features")
    if key["execution_mode"] == "smoke" and len(dataset) > 8:
        raise DatasetError("smoke cache exceeds eight samples")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    encoder.to(device).eval()
    loader = StatefulBatchLoader(dataset, batch_size=batch_size, shuffle=False,
                                 max_samples=8 if key["execution_mode"] == "smoke" else None)
    shards, rows, buffer = [], [], []
    dimension = None

    def flush():
        if not buffer:
            return
        import numpy as np
        name = f"shard-{len(shards):05d}.npy"
        np.save(root / name, torch.stack(buffer).numpy(), allow_pickle=False)
        shards.append({"file": name, "sha256": file_sha256(root / name), "rows": len(buffer)})
        buffer.clear()

    with torch.no_grad():
        for batch in loader:
            features = encoder(batch["image"].to(device)).detach().cpu().float()
            if features.ndim != 2 or not torch.isfinite(features).all():
                raise DatasetError("invalid encoder features")
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            dimension = features.shape[1]
            for sample_id, feature in zip(batch["sample_id"], features):
                rows.append({"sample_id": sample_id, "shard": len(shards), "offset": len(buffer)})
                buffer.append(feature.clone())
                if len(buffer) == shard_rows:
                    flush()
    flush()
    value = {"key": key, "rows": rows, "shards": shards, "feature_dim": dimension,
             "complete": True, "records_digest": sha256_json([record.to_dict() for record in dataset.records])}
    value["digest"] = sha256_json(value)
    write_json(root / "index.json", value)
    return value


class CachedDataset:
    def __init__(self, dataset, directory, *, expected_key):
        self.root = Path(directory)
        value = read_json(self.root / "index.json")
        digest = value.pop("digest")
        if sha256_json(value) != digest or not value["complete"] or value["key"] != expected_key:
            raise DatasetError("cache identity/completeness mismatch")
        if value["records_digest"] != sha256_json([record.to_dict() for record in dataset.records]):
            raise DatasetError("cache record identity mismatch")
        self.rows = value["rows"]
        if [row["sample_id"] for row in self.rows] != [record.sample_id for record in dataset.records]:
            raise DatasetError("cache sample order/coverage mismatch")
        self.shards, self.feature_dim, self.digest = value["shards"], value["feature_dim"], digest
        for name in ("records", "stage", "role", "partition", "purpose", "class_to_index"):
            setattr(self, name, getattr(dataset, name))
        self._mapped = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        shard_index = row["shard"]
        if shard_index not in self._mapped:
            import numpy as np
            shard = self.shards[shard_index]
            path = self.root / shard["file"]
            if path.parent != self.root or file_sha256(path) != shard["sha256"]:
                raise DatasetError("cache shard path/hash mismatch")
            features = np.load(path, mmap_mode="r", allow_pickle=False)
            if tuple(features.shape) != (shard["rows"], self.feature_dim) or features.dtype != np.float32:
                raise DatasetError("cache shard shape mismatch")
            self._mapped[shard_index] = features
        record = self.records[index]
        return SampleItem(torch.from_numpy(self._mapped[shard_index][row["offset"]].copy()), record.sample_id, record.class_id,
                          self.class_to_index[record.class_id])
