"""Versioned CLI configuration and metadata-only preflight."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .contracts import ClassMap, RunConfig, read_json, sha256_json
from .data.audit import load_manifest
from .data.splits import ALGORITHM_VERSION, assert_no_group_crossing, load_split_manifest
from .environment import machine_policy
from .models.provision import inspect_weights
from .runtime import resolve_run_config
from .training.baseline import TrainConfig

RECIPES = {"B01": {}, "B04": {}, "B03": {}, "R01": {"objective": "gce", "gce_q": .7},
           "F100": {"weighting": True}, "F010": {"lambda_preserve": .1}, "F001": {"prior_tau": 1.}}
PATHS = {"manifest", "split", "class_map", "weights", "train_cache", "dev_cache", "head", "output", "machine_config"}
FIELDS = PATHS | {"schema_version", "recipe", "stage", "execution_mode", "seed", "device", "weight_revision",
                  "batch_size", "effective_batch_size", "limits", "parameters"}


@dataclass
class PreparedRun:
    config: dict
    run: RunConfig
    train: TrainConfig
    policy: object
    records: list
    split: object
    class_map: ClassMap
    weights: dict
    preprocessing_digest: str

    def summary(self):
        return {"config": self.config, "run": self.run.to_dict(), "train_digest": self.train.digest,
                "weight_identity": self.weights, "preprocessing_digest": self.preprocessing_digest,
                "class_count": len(self.class_map.id_to_index), "split_counts": self.split.counts()}


def load_config(path):
    path = Path(path).resolve()
    value = read_json(path)
    unknown = set(value) - FIELDS
    if unknown or value.get("schema_version") != 2 or value.get("recipe") not in RECIPES:
        raise ValueError(f"invalid pipeline configuration/schema/recipe; unknown fields: {sorted(unknown)}")
    for key in PATHS:
        if value.get(key):
            value[key] = str((path.parent / value[key]).resolve())
    return value


def prepare(path):
    """Permission first; never generates artifacts, reads pixels or loads a model."""
    from .data.transforms import ClipTransform
    config = load_config(path)
    limits = config.get("limits", {})
    base = RunConfig(stage=config["stage"], execution_mode=config.get("execution_mode", "smoke"),
        seed=config.get("seed", 17), batch_size=config.get("batch_size", 1),
        max_samples=limits.get("max_samples", 8), max_updates=limits.get("max_updates", 2),
        max_eval_batches=limits.get("max_eval_batches", 2))
    policy = machine_policy(config.get("machine_config"))
    resolve_run_config(base, policy)  # before even inspecting input artifacts
    if base.execution_mode == "smoke" and base.batch_size != 1:
        raise ValueError("pipeline smoke uses microbatch=1")
    if config.get("device", "cpu") not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if base.execution_mode == "formal" and config.get("device") != "cuda":
        raise ValueError("formal workflow targets the enrolled CUDA experiment machine")
    for name in ("manifest", "split", "class_map", "weights", "weight_revision", "output"):
        if not config.get(name):
            raise ValueError(f"missing required input: {name}")
    records = load_manifest(config["manifest"], require_complete=True)
    if not records or any(record.stage != base.stage or record.role != "train" or record.decode_status != "decoded" for record in records):
        raise ValueError("current-stage complete strict-decoded training audit required")
    manifest_digest = sha256_json([record.to_dict() for record in sorted(records, key=lambda item: item.sample_id)])
    split = load_split_manifest(config["split"], require_parent_digest=manifest_digest)
    if split.algorithm_version != ALGORITHM_VERSION:
        raise ValueError("legacy split: regenerate with grouped split v2")
    assert_no_group_crossing(split)
    value = read_json(config["class_map"])
    class_map = ClassMap(stage=value["stage"], id_to_index=value["id_to_index"])
    if class_map.stage != base.stage or set(class_map.id_to_index) != {record.class_id for record in records}:
        raise ValueError("class map differs from current-stage training manifest")
    # from_split enforces complete parent/coverage before selecting anything.
    from .data.dataset import ManifestDataset
    for partition in ("train", "dev", "confirm"):
        if any(item.partition == partition for item in split.records):
            ManifestDataset.from_split(records, split.records, stage=base.stage, partition=partition,
                purpose=partition, class_to_index=dict(class_map.id_to_index))
    weights = inspect_weights(config["weights"], config["weight_revision"])
    preprocessing = ClipTransform.identity(weights["preprocessing_digest"])
    effective = config.get("effective_batch_size", 128)
    if effective <= 0 or effective % base.batch_size:
        raise ValueError("effective batch size must be a positive multiple of microbatch")
    defaults = {"formal_epochs": 20 if config["recipe"] == "B01" else 10,
                "learning_rate": 1e-3, "lora_learning_rate": 1e-4, "weight_decay": 1e-4,
                "scheduler": "warmup_cosine", "warmup_epochs": 1,
                "profile": "CACHE20" if config["recipe"] == "B01" else "ONLINE10",
                "accumulation_steps": effective // base.batch_size if base.execution_mode == "formal" else 1,
                **RECIPES[config["recipe"]]}
    defaults.update(config.get("parameters", {}))
    if base.execution_mode == "smoke":
        defaults["accumulation_steps"] = 1
        defaults["epochs"] = 1
    run = RunConfig(stage=base.stage, execution_mode=base.execution_mode, seed=base.seed,
        batch_size=base.batch_size, max_samples=base.max_samples, max_updates=base.max_updates,
        max_eval_batches=base.max_eval_batches, manifest_digest=manifest_digest, split_digest=split.digest,
        class_map_digest=class_map.digest, official_weight_revision=weights["revision"],
        output_root=config["output"], parameters=defaults)
    return PreparedRun(config, run, TrainConfig.from_run(run), policy, records, split, class_map, weights, preprocessing)
