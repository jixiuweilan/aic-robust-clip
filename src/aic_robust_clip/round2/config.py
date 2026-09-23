"""Metadata-only second-round configuration and immutable asset identities."""
from pathlib import Path
import os
from dataclasses import dataclass
from ..contracts import ClassMap, read_json, sha256_json, write_json
from ..data.audit import load_manifest
from ..data.splits import make_grouped_split, load_split_manifest
from ..data.dataset import ManifestDataset
from ..models.provision import file_sha256, inspect_weights
from ..data.transforms import ClipTransform

VERSION = "round2-v1"
GMM_MAX_ITERATIONS = 1000
RUNS = {"T4-0-CE": ("t4", "full_visual", "ce"), "T4-1-TURN": ("t4", "full_visual", "turn"),
        "T4-2-FINE": ("t4", "full_visual", "fine"), "T4-3-SNSCL": ("t4", "full_visual", "snscl"),
        "4060-A-CE": ("4060-a", "lora", "ce"), "4060-A-TURN": ("4060-a", "lora", "turn"),
        "4060-B-CE": ("4060-b", "lora", "ce"), "4060-B-FINE": ("4060-b", "lora", "fine"),
        "C4090-LORA-CE": ("cloud4090-lora", "lora", "ce"),
        "C4090-LORA-TURN": ("cloud4090-lora", "lora", "turn"),
        "C4090-FULL-CE": ("cloud4090-full", "full_visual", "ce"),
        "C4090-FULL-TURN": ("cloud4090-full", "full_visual", "turn")}
CLOUD_GROUPS = {"cloud4090-lora", "cloud4090-full"}
GROUPS = {group for group, _, _ in RUNS.values()}


def adaptation_for(group):
    if group not in GROUPS:
        raise ValueError("unknown round2 admission group")
    return "full_visual" if group in {"t4", "cloud4090-full"} else "lora"
RECIPE = {"stage": "second_round", "seed": 17, "epochs": 30, "pause_after_epoch": 10,
          "initializer": "HEAD20-GCE", "input_size": 224, "precision": "fp16", "effective_batch": 128,
          "head_lr": 1e-3, "visual_lr": 1e-5, "lora_lr": 1e-4, "auxiliary_lr": 1e-3,
          "weight_decay": 1e-4, "warmup_epochs": 1, "scheduler": "epoch_fraction_cosine",
          "lora_rank": 4, "lora_alpha": 4, "eval_batch": 64, "scoring_batch": 64,
          "gmm": {"iterations": GMM_MAX_ITERATIONS, "tolerance": 1e-6, "variance_floor": 1e-6, "threshold": .6},
          "snscl": {"warmup_epochs": 5, "threshold": .5, "ema": .99, "momentum": .999,
                    "temperature": .07, "queue_capacity": 32, "projection_dim": 128,
                    "contrastive_weight": .1, "kl_weight": 1e-4}}


def stage_path(path, *, stage="second_round"):
    path = Path(path).resolve()
    if stage not in {"second_round", "preliminary"} or stage not in path.parts or any(
            other in path.parts for other in {"preliminary", "second_round", "semifinal"} - {stage}):
        raise ValueError(f"isolated {stage} path required: {path}")
    return path


def sealed(value):
    return {**value, "digest": sha256_json(value)}


def verify_seal(value):
    body = {k: v for k, v in value.items() if k != "digest"}
    if value.get("digest") != sha256_json(body):
        raise ValueError("artifact descriptor digest mismatch")
    return body


@dataclass
class Assets:
    path: Path
    descriptor: dict
    records: list
    split: object
    class_map: ClassMap
    weights: dict

    @property
    def root(self):
        return self.path.parent

    @property
    def weights_path(self):
        return os.environ.get("AIC_ROUND2_WEIGHTS", self.descriptor["weights_path"])

    def dataset(self, partition, processor=None, online=False, purpose=None):
        if partition not in {"train", "dev"}:
            raise ValueError("round2 work permits train/dev only")
        return ManifestDataset.from_split(self.records, self.split.records, stage="second_round", partition=partition,
            purpose=purpose or partition, class_to_index=dict(self.class_map.id_to_index),
            transform=ClipTransform(processor, seed=17, online=online) if processor else None)

    def head_identity(self):
        return {"version": VERSION, "asset_digest": self.descriptor["digest"], "stage": "second_round",
                "profile": "HEAD20-GCE", "epoch": 20, "seed": 17, "objective": "gce", "q": .7,
                "optimizer": "AdamW", "lr": .01, "weight_decay": 1e-4, "effective_batch": 128,
                "warmup_epochs": 1, "scheduler": "epoch_fraction_cosine"}


def load_assets(path, *, verify_archives=False):
    path = stage_path(path)
    audit_status = read_json(path.parent / "status.json")
    if audit_status.get("status") != "succeeded" or audit_status.get("exit_code") != 0:
        raise ValueError("complete successful audit required; interrupted tasks are not assets")
    value = read_json(path)
    verify_seal(value)
    if value.get("version") != VERSION or value.get("stage") != "second_round" or value.get("protocol") != "train-only-grouped-80-10-10-seed17":
        raise ValueError("invalid round2 asset protocol")
    if not value.get("source_url") or not value.get("retrieved_at") or not value.get("organizer_version"):
        raise ValueError("organizer provenance required")
    for name, digest in value["files"].items():
        target = stage_path(path.parent / name)
        if target.parent != path.parent or file_sha256(target) != digest:
            raise ValueError("asset metadata file hash mismatch")
    records = load_manifest(path.parent / "manifest.json", require_complete=True)
    if "exclusions" in value:
        exclusions = read_json(path.parent / "exclusions.json")
        excluded_paths = {e["member_path"] for e in exclusions}
        if exclusions != value["exclusions"] or any(r.member_path in excluded_paths for r in records):
            raise ValueError("training exclusion does not match asset records")
    if not records or any(r.stage != "second_round" or r.role != "train" or r.decode_status != "decoded" for r in records):
        raise ValueError("only fully decoded second-round training records permitted")
    for archive, digest in value["archives"].items():
        stage_path(archive)
        if verify_archives:
            from ..data.relocation import resolve_archive
            relocated = resolve_archive(Path(archive), archive_identity=digest)
            stage_path(relocated)
            if file_sha256(relocated) != digest:
                raise ValueError("round2 archive hash mismatch")
    if any(str(Path(r.archive_path).resolve()) not in value["archives"] for r in records):
        raise ValueError("manifest references an unregistered archive")
    if any(r.archive_identity != value["archives"][str(Path(r.archive_path).resolve())] for r in records):
        raise ValueError("manifest archive identity does not match registered bytes")
    expected = make_grouped_split(records, seed=17)
    split = load_split_manifest(path.parent / "split.json", require_parent_digest=expected.parent_manifest_digest)
    if split.digest != expected.digest or split.algorithm_version != expected.algorithm_version:
        raise ValueError("split differs from fixed seed17 duplicate-group protocol")
    cmap = read_json(path.parent / "class-map.json")
    class_map = ClassMap(stage=cmap["stage"], id_to_index=cmap["id_to_index"])
    if class_map.stage != "second_round" or set(class_map.id_to_index) != {r.class_id for r in records}:
        raise ValueError("class map differs from round2 records")
    weights = inspect_weights(os.environ.get("AIC_ROUND2_WEIGHTS", value["weights_path"]), value["weight_revision"])
    if weights != value["weights"]:
        raise ValueError("official weights changed")
    assets = Assets(path, value, records, split, class_map, weights)
    for partition in ("train", "dev"):
        dataset = assets.dataset(partition)
        if not len(dataset):
            raise ValueError("empty train/dev split")
        dataset.close()
    return assets


def head_descriptor(assets):
    root = assets.root / "HEAD20-GCE"
    value = read_json(root / "head.json")
    if value["identity"] != assets.head_identity() or value["sha256"] != file_sha256(root / "head.pt"):
        raise ValueError("HEAD20-GCE identity/hash mismatch; HEAD3 is forbidden")
    return value


def prepare_configs(output, assets_path=None, receipt_path=None):
    """No pixels, model construction, cache building, training or downloads."""
    root = stage_path(output)
    assets = load_assets(assets_path) if assets_path else None
    head = head_descriptor(assets) if assets else None
    receipt = read_json(receipt_path) if receipt_path else None
    if receipt:
        from .admission import validate_receipt
        validate_receipt(receipt, assets, live=False)
    root.mkdir(parents=True, exist_ok=False)
    configs = []
    for name, (group, adaptation, method) in RUNS.items():
        applies = receipt and receipt["group"] == group and method in receipt["assignments"]
        status = "blocked_on_round2_assets" if assets is None else "ready" if applies else "blocked_on_machine_admission"
        config = {"version": VERSION, "run_id": name, "group": group, "adaptation": adaptation, "method": method,
                  "recipe": RECIPE, "status": status, "assets": str(assets.path) if assets else None,
                  "asset_digest": assets.descriptor["digest"] if assets else None,
                  "head_sha256": head["sha256"] if head else None,
                  "receipt": str(Path(receipt_path).resolve()) if applies else None,
                  "receipt_digest": receipt["digest"] if applies else None,
                  "engineering": receipt["engineering"] if applies else None,
                  "output": str(root.parent / "runs" / name)}
        write_json(root / f"{name}.json", sealed(config))
        configs.append({"run_id": name, "status": status})
    write_json(root / "index.json", configs)
    return configs


def check_config(path, *, ready=True):
    value = read_json(path)
    verify_seal(value)
    if value.get("version") != VERSION or value.get("recipe") != RECIPE or value.get("run_id") not in RUNS:
        raise ValueError("unknown round2 configuration/recipe")
    if (value["group"], value["adaptation"], value["method"]) != RUNS[value["run_id"]]:
        raise ValueError("assignment mismatch")
    stage_path(value["output"])
    if value["status"] != "ready":
        if ready:
            raise ValueError(value["status"])
        return value, None
    assets = load_assets(value["assets"])
    if assets.descriptor["digest"] != value["asset_digest"] or head_descriptor(assets)["sha256"] != value["head_sha256"]:
        raise ValueError("configuration assets/head changed")
    from .admission import validate_receipt
    receipt = read_json(value["receipt"])
    validate_receipt(receipt, assets, live=ready)
    if receipt["digest"] != value["receipt_digest"] or receipt["group"] != value["group"] or receipt["engineering"] != value["engineering"]:
        raise ValueError("configuration admission mismatch")
    if ready:
        from .admission import runtime_identity
        from ..gpu_identity import normalize_gpu_uuid
        if normalize_gpu_uuid(receipt["assignments"].get(value["method"])) != normalize_gpu_uuid(runtime_identity()["gpu"]["uuid"]):
            raise ValueError("run must use its assigned GPU UUID")
    return value, assets
