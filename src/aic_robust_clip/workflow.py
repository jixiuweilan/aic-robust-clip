"""One orchestration path for preparation, training, evaluation and prediction.

No command implicitly audits, downloads weights, generates caches or fits an
initializer. Each prerequisite must already exist and match the resolved run.
"""
from __future__ import annotations

import copy
from contextlib import ExitStack
from collections import Counter
from dataclasses import replace
from pathlib import Path
import time

from .configuration import prepare
from .contracts import CheckpointMetadata, RunConfig, read_json, sha256_json, write_json
from .data.audit import load_manifest
from .data.cache import CachedDataset, cache_key, generate_cache
from .data.dataset import ManifestDataset
from .data.loading import StatefulBatchLoader
from .data.transforms import ClipTransform
from .inference import package_submission, predict_dataset, records_to_filename_rows, write_prediction_csv
from .models.clip import load_openai_clip, torch
from .models.lora import VisualLoRAModel
from .models.provision import file_sha256
from .runtime import current_code_revision, seed_everything, stable_seeded_order
from .training.baseline import FrozenFeatureBaseline, OnlineFrozenBaseline, TrainConfig, evaluate_loader, train_baseline
from .training.checkpoint import load_checkpoint


def dataset_for(ctx, partition, processor=None, *, online=False, purpose=None):
    dataset = ManifestDataset.from_split(ctx.records, ctx.split.records, stage=ctx.run.stage,
        partition=partition, purpose=purpose or partition, class_to_index=dict(ctx.class_map.id_to_index),
        transform=ClipTransform(processor, seed=ctx.run.seed, online=online) if processor is not None else None)
    if ctx.run.execution_mode == "smoke":
        limit = ctx.run.max_samples if partition == "train" else ctx.run.max_eval_batches * ctx.run.batch_size
        ids = set(stable_seeded_order([record.sample_id for record in dataset.records], ctx.run.seed)[:limit])
        dataset.records = tuple(record for record in dataset.records if record.sample_id in ids)
    return dataset


def stream(ctx, dataset, *, shuffle=True, batch_size=None):
    perf = ctx.performance
    if batch_size is None:
        batch_size = ctx.run.batch_size if dataset.purpose == "train" else perf.eval_batch_size
    return StatefulBatchLoader(dataset, batch_size=batch_size, seed=ctx.run.seed, shuffle=shuffle,
        max_samples=ctx.run.max_samples if ctx.run.execution_mode == "smoke" else None,
        num_workers=0 if isinstance(dataset, CachedDataset) else (
            perf.num_workers if dataset.purpose == "train" else perf.eval_num_workers),
        prefetch_factor=perf.prefetch_factor, pin_memory=perf.pin_memory)


def cache_for(ctx, partition, dataset):
    name = f"{partition}_cache"
    if not ctx.config.get(name):
        raise ValueError(f"missing prerequisite: {name}; run aic-cache-features explicitly")
    return CachedDataset(dataset, ctx.config[name], expected_key=cache_key(ctx.run, ctx.weights["digest"],
        ctx.preprocessing_digest, partition))


def load_bundle(ctx):
    return load_openai_clip(revision=ctx.weights["revision"], local_path=ctx.config["weights"],
        weight_digest_value=ctx.weights["digest"], local_files_only=True)


def reserve_output(path, *, resume=False):
    root = Path(path)
    if resume:
        if not root.is_dir() or not (root / "resolved.json").is_file():
            raise ValueError("resume requires an existing identified run directory")
    else:
        root.mkdir(parents=True, exist_ok=False)
    return root


def metadata_for(ctx, config, *, initialization, family=None):
    return CheckpointMetadata(model_family=family or ctx.config["recipe"], stage=ctx.run.stage,
        class_map_digest=ctx.run.class_map_digest, manifest_digest=ctx.run.manifest_digest,
        split_digest=ctx.run.split_digest, official_weight_id=ctx.run.official_weight_id,
        official_weight_revision=ctx.run.official_weight_revision, configuration_digest=config.digest,
        code_revision=current_code_revision(), progress={}, optimizer_state_present=True,
        scheduler_state_present=True, rng_state_present=True, sampler_state_present=True,
        weight_files_digest=ctx.weights["digest"], preprocessing_digest=ctx.preprocessing_digest,
        initialization_digest=initialization, execution_mode=ctx.run.execution_mode)


def compatibility(metadata):
    return {name: getattr(metadata, name) for name in ("stage", "class_map_digest", "manifest_digest", "split_digest",
        "official_weight_id", "official_weight_revision", "weight_files_digest", "preprocessing_digest",
        "initialization_digest", "execution_mode")}


def head_identity(ctx):
    return {"schema_version": 2, "stage": ctx.run.stage, "mode": ctx.run.execution_mode, "seed": ctx.run.seed,
        "manifest": ctx.run.manifest_digest, "split": ctx.run.split_digest, "class_map": ctx.run.class_map_digest,
        "weights": ctx.weights["digest"], "preprocessing": ctx.preprocessing_digest,
        "profile": "HEAD3" if ctx.run.execution_mode == "formal" else "HEAD-SMOKE"}


def load_head(ctx):
    if not ctx.config.get("head"):
        raise ValueError("online methods require the shared head initializer")
    root = Path(ctx.config["head"])
    descriptor = read_json(root / "head.json")
    if descriptor["identity"] != head_identity(ctx) or file_sha256(root / "head.pt") != descriptor["sha256"]:
        raise ValueError("shared head initializer identity/hash mismatch")
    return torch.load(root / "head.pt", map_location="cpu", weights_only=True), descriptor["sha256"]


def cache_command(config_path, partition):
    ctx = prepare(config_path)
    if partition not in {"train", "dev"}:
        raise ValueError("cache accepts train/dev only")
    directory = ctx.config.get(f"{partition}_cache")
    if not directory or Path(directory).exists():
        raise ValueError("cache output must be configured and must not already exist")
    bundle = load_bundle(ctx)
    dataset = dataset_for(ctx, partition, bundle.processor)
    try:
        return generate_cache(dataset, bundle.encoder, directory,
            key=cache_key(ctx.run, ctx.weights["digest"], ctx.preprocessing_digest, partition),
            device=ctx.config.get("device", "cpu"), batch_size=ctx.performance.cache_batch_size,
            num_workers=ctx.performance.num_workers, prefetch_factor=ctx.performance.prefetch_factor,
            pin_memory=ctx.performance.pin_memory)
    finally:
        dataset.close()


def init_head_command(config_path):
    ctx = prepare(config_path)
    if not ctx.config.get("head"):
        raise ValueError("configure the shared head output directory")
    dataset = cache_for(ctx, "train", dataset_for(ctx, "train"))
    root = reserve_output(ctx.config["head"])
    seed_everything(ctx.run.seed)
    head_batch = ctx.performance.head_batch_size
    accumulation = (ctx.config.get("effective_batch_size", 128) // head_batch
                    if ctx.run.execution_mode == "formal" else 1)
    run = replace(ctx.run, batch_size=head_batch,
        parameters={"formal_epochs": 3, "profile": "HEAD3", "scheduler": "warmup_cosine",
        "accumulation_steps": accumulation}, output_root=str(root))
    config = TrainConfig.from_run(run)
    model = FrozenFeatureBaseline(dataset.feature_dim, len(ctx.class_map.id_to_index))
    meta = metadata_for(ctx, config, initialization=sha256_json({"random_head_seed": ctx.run.seed}), family="HEAD3")
    resolved = {"run": run.to_dict(), "identity": head_identity(ctx)}
    if "performance" in ctx.config:
        resolved["performance"] = ctx.config["performance"]
    write_json(root / "resolved.json", resolved)
    try:
        with stream(ctx, dataset, batch_size=head_batch) as loader:
            result = train_baseline(model, loader, config=config, class_count=len(ctx.class_map.id_to_index),
                device=ctx.config.get("device", "cpu"), policy=ctx.policy, checkpoint_dir=root, checkpoint_metadata=meta)
        torch.save({name: tensor.cpu() for name, tensor in model.classifier.state_dict().items()}, root / "head.pt")
        descriptor = {"identity": head_identity(ctx), "sha256": file_sha256(root / "head.pt"), "result": result.to_dict()}
        write_json(root / "head.json", descriptor)
        return descriptor
    except Exception as exc:
        write_json(root / "failure.json", {"error": str(exc), "status": "failed", "auto_retry": False})
        raise


def training_components(ctx):
    seed_everything(ctx.run.seed)
    if ctx.config["recipe"] == "B01":
        train = cache_for(ctx, "train", dataset_for(ctx, "train"))
        dev = cache_for(ctx, "dev", dataset_for(ctx, "dev"))
        model = FrozenFeatureBaseline(train.feature_dim, len(ctx.class_map.id_to_index))
        return model, train, dev, None, None, sha256_json({"random_head_seed": ctx.run.seed})
    # Resolve prerequisite before allocating a backbone.
    head, initialization = load_head(ctx)
    bundle = load_bundle(ctx)
    reference = copy.deepcopy(bundle.encoder) if ctx.train.lambda_preserve else None
    if ctx.config["recipe"] == "B04":
        model = OnlineFrozenBaseline(bundle.encoder, bundle.encoder.output_dim, len(ctx.class_map.id_to_index))
    else:
        model = VisualLoRAModel(bundle.encoder, len(ctx.class_map.id_to_index))
        model.assert_qv_only()
    model.classifier.load_state_dict(head)
    train = dataset_for(ctx, "train", bundle.processor, online=True)
    dev = dataset_for(ctx, "dev", bundle.processor)
    scoring = dataset_for(ctx, "train", bundle.processor, purpose="scoring") if ctx.train.weighting else None
    return model, train, dev, scoring, reference, initialization


def train_command(config_path, *, resume=None, stop_after_updates=None):
    ctx = prepare(config_path)
    output = Path(ctx.config["output"])
    if resume is not None:
        resume = Path(resume).resolve()
        if resume.parent != output:
            raise ValueError("resume checkpoint must belong to the configured run directory")
        prior = read_json(output / "resolved.json")
        if prior != ctx.summary():
            raise ValueError("resolved configuration differs from the existing run")
    elif output.exists():
        raise FileExistsError(output)
    model, train, dev, scoring, reference, initialization = training_components(ctx)
    root = reserve_output(output, resume=resume is not None)
    write_json(root / "resolved.json", ctx.summary())
    meta = metadata_for(ctx, ctx.train, initialization=initialization)
    device = ctx.config.get("device", "cpu")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        labels = {record.sample_id: train.class_to_index[record.class_id] for record in train.records}
        with ExitStack() as streams:
            train_stream = streams.enter_context(stream(ctx, train))
            dev_stream = streams.enter_context(stream(ctx, dev, shuffle=False))
            scoring_stream = streams.enter_context(stream(ctx, scoring, shuffle=False)) if scoring else None
            result = train_baseline(model, train_stream, config=ctx.train,
                class_count=len(ctx.class_map.id_to_index), dev_loader=dev_stream,
                training_counts=dict(Counter(labels.values())), device=device, policy=ctx.policy,
                training_labels=labels, scoring_loader=scoring_stream,
                reference_encoder=reference, checkpoint_dir=root, checkpoint_metadata=meta,
                resume_from=resume, stop_after_updates=stop_after_updates)
        report = {**result.to_dict(), "status": "paused" if stop_after_updates is not None else "complete",
            "evidence": "startup_only" if ctx.run.execution_mode == "smoke" else "formal",
            "label_quality": "noisy_proxy", "elapsed_seconds": time.monotonic() - started,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "code_revision": meta.code_revision}
        write_json(root / "result.json", report)
        if result.dev_predictions:
            write_json(root / "dev-predictions.json", [
                {"sample_id": record.sample_id, "label": label, "prediction": prediction}
                for record, label, prediction in zip(dev.records, result.dev_labels, result.dev_predictions)])
        return report
    except Exception as exc:
        write_json(root / "failure.json", {"status": "failed", "error": str(exc), "auto_retry": False})
        raise
    finally:
        for dataset in (train, dev, scoring):
            if hasattr(dataset, "close"):
                dataset.close()


def inference_components(ctx, checkpoint):
    seed_everything(ctx.run.seed)
    bundle = load_bundle(ctx)
    recipe = ctx.config["recipe"]
    initialization = sha256_json({"random_head_seed": ctx.run.seed}) if recipe == "B01" else load_head(ctx)[1]
    if recipe in {"B01", "B04"}:
        model = OnlineFrozenBaseline(bundle.encoder, bundle.encoder.output_dim, len(ctx.class_map.id_to_index))
    else:
        model = VisualLoRAModel(bundle.encoder, len(ctx.class_map.id_to_index))
    metadata = metadata_for(ctx, ctx.train, initialization=initialization)
    # B01 checkpoints contain just the classifier; attach the SAME frozen
    # official encoder for pixel inference, without fitting a second model.
    target = FrozenFeatureBaseline(bundle.encoder.output_dim, len(ctx.class_map.id_to_index)) if recipe == "B01" else model
    load_checkpoint(checkpoint, model=target, requested=compatibility(metadata), restore_rng=False,
                    expected_configuration_digest=ctx.train.digest)
    if recipe == "B01":
        model.classifier.load_state_dict(target.classifier.state_dict())
    model.to(ctx.config.get("device", "cpu")).eval()
    return model, bundle.processor


def check_selection(ctx, checkpoint, selection):
    value = read_json(selection)
    expected = {"locked": True, "recipe": ctx.config["recipe"], "stage": ctx.run.stage,
                "split_digest": ctx.run.split_digest, "configuration_digest": ctx.train.digest,
                "checkpoint_sha256": file_sha256(checkpoint)}
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError("selection record is absent, unlocked or incompatible")


def lock_selection_command(config_path, checkpoint, output):
    ctx = prepare(config_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["metadata"].get("model_family") == "BENCHMARK":
        raise ValueError("benchmark checkpoints cannot be selected")
    initial = sha256_json({"random_head_seed": ctx.run.seed}) if ctx.config["recipe"] == "B01" else load_head(ctx)[1]
    metadata_value = dict(payload["metadata"])
    metadata_value.pop("schema_version", None)
    meta = CheckpointMetadata(**metadata_value)
    meta.assert_compatible(**compatibility(metadata_for(ctx, ctx.train, initialization=initial)))
    if meta.configuration_digest != ctx.train.digest:
        raise ValueError("cannot lock a checkpoint from another configuration")
    if Path(output).exists():
        raise FileExistsError(output)
    value = {"locked": True, "recipe": ctx.config["recipe"], "stage": ctx.run.stage,
             "split_digest": ctx.run.split_digest, "configuration_digest": ctx.train.digest,
             "checkpoint_sha256": file_sha256(checkpoint)}
    write_json(output, value)
    return value


def evaluate_command(config_path, checkpoint, partition, output, *, selection=None):
    ctx = prepare(config_path)
    if partition not in {"dev", "confirm"}:
        raise ValueError("evaluation accepts dev/confirm only")
    if partition == "confirm":
        if selection is None:
            raise ValueError("confirm requires an explicit locked selection record")
        check_selection(ctx, checkpoint, selection)
    if Path(output).exists():
        raise FileExistsError(output)
    model, processor = inference_components(ctx, checkpoint)
    dataset = dataset_for(ctx, partition, processor)
    train_records = dataset_for(ctx, "train").records
    counts = Counter(ctx.class_map.index_for(record.class_id) for record in train_records)
    try:
        with stream(ctx, dataset, shuffle=False) as loader:
            metrics, predictions, labels = evaluate_loader(model, loader,
                total_classes=len(ctx.class_map.id_to_index), training_counts=counts,
                max_batches=ctx.run.max_eval_batches if ctx.run.execution_mode == "smoke" else None,
                device=ctx.config.get("device", "cpu"))
        result = {"partition": partition, "label_quality": "noisy_proxy", "metrics": metrics.to_dict(),
            "checkpoint_sha256": file_sha256(checkpoint), "predictions": [
                {"sample_id": record.sample_id, "label": label, "prediction": prediction}
                for record, label, prediction in zip(dataset.records, labels, predictions)]}
        write_json(output, result)
        return result
    finally:
        dataset.close()


def predict_command(config_path, checkpoint, manifest, output, *, selection=None):
    ctx = prepare(config_path)
    if ctx.run.execution_mode == "formal":
        if selection is None:
            raise ValueError("formal prediction requires a locked selection")
        check_selection(ctx, checkpoint, selection)
    elif not read_json(manifest).get("synthetic_fixture", False):
        raise ValueError("local prediction checks accept synthetic fixture manifests only")
    records = load_manifest(manifest, require_complete=True)
    if not records or any(record.stage != ctx.run.stage or record.role != "test" for record in records):
        raise ValueError("prediction requires a current-stage test manifest")
    if ctx.run.execution_mode == "smoke" and len(records) > ctx.run.max_samples:
        raise ValueError("synthetic prediction fixture exceeds smoke sample cap")
    if Path(output).exists():
        raise FileExistsError(output)
    model, processor = inference_components(ctx, checkpoint)
    dataset = ManifestDataset(records, stage=ctx.run.stage, role="test", partition="train", purpose="inference",
                              transform=ClipTransform(processor))
    root = reserve_output(output)
    try:
        predictions = predict_dataset(model, dataset, ctx.class_map, batch_size=ctx.run.batch_size,
                                      device=ctx.config.get("device", "cpu"))
        rows = records_to_filename_rows(dataset, predictions)
        expected = root / "expected-files.txt"
        expected.write_text("\n".join(record.member_path.rsplit("/", 1)[-1] for record in dataset.records) + "\n", encoding="utf-8")
        write_prediction_csv(root / "pred_results.csv", rows, class_map=ctx.class_map, expected_files=expected)
        count = package_submission(root / "pred_results.csv", root / "submission.zip", class_map=ctx.class_map, expected_files=expected)
        report = {"rows": count, "checkpoint_sha256": file_sha256(checkpoint), "package_sha256": file_sha256(root / "submission.zip"),
                  "stage": ctx.run.stage, "class_map_digest": ctx.class_map.digest}
        write_json(root / "prediction.json", report)
        return report
    finally:
        dataset.close()
