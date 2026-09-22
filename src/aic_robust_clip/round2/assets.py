"""Explicit audit/cache/HEAD20 operations; never called by config preparation."""
from pathlib import Path
from collections import Counter
import torch
from ..contracts import RunConfig, write_json
from ..data.audit import audit_archive, write_audit_report, class_map_from_records
from ..data.splits import make_grouped_split, write_split_manifest
from ..data.cache import CachedDataset, generate_cache
from ..models.provision import inspect_weights, file_sha256
from ..models.classifier import LinearClassifier
from ..data.transforms import ClipTransform
from ..runtime import current_code_revision, seed_everything
from ..training.objectives import gce_loss
from .config import VERSION, stage_path, sealed, load_assets
from .engine import require_machine, bundle_for, loader_for, schedule, atomic_save


def audit(archive, output, *, source_url, retrieved_at, organizer_version, weights, weight_revision,
          member_prefix=None, expected_sha256=None):
    from .journal import task_journal, atomic_json
    archive, root = stage_path(archive), stage_path(output)
    if archive.name.lower() == "test.zip":
        raise ValueError("test archive forbidden")
    if not source_url.startswith("https://") or not retrieved_at or not organizer_version:
        raise ValueError("provide organizer source URL, retrieval date and received version")
    root.mkdir(parents=True, exist_ok=False)
    with task_journal(root) as journal:
        weight_identity = inspect_weights(weights, weight_revision)
        report = audit_archive(archive, stage="second_round", role="train", decode=True,
                               member_prefix=member_prefix, expected_sha256=expected_sha256,
                               progress=lambda event: journal.update(**event))
        write_audit_report(root / "manifest.json", report)
        write_json(root / "decode-failures.json", [f.to_dict() for f in report.failures])
        # Failed or partial reports never become trainable descriptors.
        from ..data.audit import load_manifest
        records = load_manifest(root / "manifest.json", require_complete=True)
        journal.update(phase="grouping_and_splitting")
        split = make_grouped_split(records, seed=17)
        class_map = class_map_from_records(records)
        write_split_manifest(root / "split.json", split)
        write_json(root / "class-map.json", class_map.to_dict())
        write_json(root / "coverage.json", split.report())
        groups = {}
        for row in split.records:
            groups.setdefault(row.group_id, []).append(row.sample_id)
        write_json(root / "duplicates.json", {k: v for k, v in groups.items() if len(v) > 1})
        journal.update(phase="verifying_archive_unchanged")
        if file_sha256(archive) != report.archive_identity:
            raise ValueError("archive changed during audit")
        value = {"version": VERSION, "stage": "second_round", "protocol": "train-only-grouped-80-10-10-seed17",
                 "source_url": source_url, "retrieved_at": retrieved_at, "organizer_version": organizer_version,
                 "archives": {str(archive): report.archive_identity}, "weights_path": str(Path(weights).resolve()),
                 "weight_revision": weight_revision, "weights": weight_identity,
                 "files": {name: file_sha256(root / name) for name in
                           ("manifest.json", "split.json", "class-map.json", "coverage.json",
                            "duplicates.json", "decode-failures.json")}}
        if member_prefix is not None:
            value["train_member_prefix"] = member_prefix
        value = sealed(value)
        atomic_json(root / "assets.json", value)
    return value


def cache_identity(assets):
    return {"schema_version": VERSION, "stage": "second_round", "execution_mode": "formal",
            "asset_digest": assets.descriptor["digest"], "partition": "train",
            "preprocessing": ClipTransform.identity(assets.weights["preprocessing_digest"])}


def cache(assets_path, *, machine):
    require_machine(machine)
    assets = load_assets(assets_path, verify_archives=True)
    bundle = bundle_for(assets)
    dataset = assets.dataset("train", bundle.processor)
    try:
        return generate_cache(dataset, bundle.encoder, assets.root / "train-cache", key=cache_identity(assets),
                              device="cuda", batch_size=64, num_workers=2, pin_memory=True)
    finally:
        dataset.close()


def init_head(assets_path, *, machine):
    require_machine(machine)
    assets = load_assets(assets_path)
    seed_everything(17)
    dataset = CachedDataset(assets.dataset("train"), assets.root / "train-cache", expected_key=cache_identity(assets))
    root = assets.root / "HEAD20-GCE"
    root.mkdir(parents=True, exist_ok=False)
    model = LinearClassifier(dataset.feature_dim, len(assets.class_map.id_to_index)).to("cuda")
    optimizer = torch.optim.AdamW([{"params": [model.linear.weight], "weight_decay": 1e-4},
                                  {"params": [model.linear.bias], "weight_decay": 0.}], lr=.01)
    history = []
    try:
        with loader_for(dataset, batch=128, shuffle=True) as loader:
            for epoch in range(20):
                loader.reset(epoch)
                loss_sum, count = 0., 0
                for batch in loader:
                    for group in optimizer.param_groups:
                        group["lr"] = .01 * schedule(epoch + loader.position / len(loader.order), total=20)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(batch["image"].to("cuda"))
                    loss = gce_loss(logits, batch["label_index"].to("cuda"), q=.7).mean()
                    if not torch.isfinite(loss):
                        raise ValueError("HEAD20-GCE nonfinite loss")
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.detach()) * len(logits)
                    count += len(logits)
                history.append({"epoch": epoch + 1, "loss": loss_sum / count, "samples": count})
        digest = atomic_save({k: v.cpu() for k, v in model.state_dict().items()}, root / "head.pt")
        from .admission import runtime_identity
        value = {"identity": assets.head_identity(), "sha256": digest, "history": history,
                 "train_cache_digest": dataset.digest,
                 "source": current_code_revision(), "runtime": runtime_identity(), "selection": "fixed_epoch20_no_dev"}
        write_json(root / "head.json", value)
        return value
    except BaseException as exc:
        write_json(root / "failure.json", {"error": repr(exc), "auto_retry": False})
        raise
