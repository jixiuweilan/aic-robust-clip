"""Round2 training, complete epoch-boundary resume and student-only replay."""
from __future__ import annotations
import copy
import math
import os
import tempfile
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
import torch
from torch.nn import functional as F
from ..contracts import write_json, sha256_json
from ..data.loading import StatefulBatchLoader
from ..environment import machine_policy
from ..contracts import RunConfig
from ..runtime import current_code_revision, seed_everything
from ..models.clip import load_openai_clip
from ..models.provision import file_sha256
from ..metrics import evaluate_classification
from ..training.checkpoint import _rng_state, _restore_rng_state, publish_best
from .config import VERSION, head_descriptor, check_config
from .methods import MethodState, SNSCL, MethodError
from .model import Student, optimizer_groups


def clock(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    return time.monotonic()


def schedule(epoch_fraction, total=30):
    if not 0 <= epoch_fraction <= total:
        raise ValueError("schedule position outside full plan")
    return epoch_fraction if epoch_fraction < 1 else .5 * (1 + math.cos(math.pi * (epoch_fraction - 1) / (total - 1)))


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    return file_sha256(path)


def record_failure(root, exc):
    root = Path(root)
    index = 1
    while (root / f"failure-{index:03d}.json").exists():
        index += 1
    value = {"status": "failed", "error": repr(exc), "auto_retry": False}
    if isinstance(exc, MethodError):
        value["method_diagnostics"] = exc.diagnostics
    write_json(root / f"failure-{index:03d}.json", value)
    if not (root / "failure.json").exists():
        write_json(root / "failure.json", value)


def bundle_for(assets):
    return load_openai_clip(revision=assets.weights["revision"], local_path=assets.weights_path,
                            weight_digest_value=assets.weights["digest"], local_files_only=True)


def student_for(assets, adaptation):
    descriptor = head_descriptor(assets)
    bundle = bundle_for(assets)
    model = Student(bundle.encoder, len(assets.class_map.id_to_index), adaptation)
    model.classifier.load_state_dict(torch.load(assets.root / "HEAD20-GCE/head.pt", weights_only=True, map_location="cpu"))
    return model, bundle.processor, descriptor


def loader_for(dataset, *, batch, workers=0, shuffle=False):
    return StatefulBatchLoader(dataset, batch_size=batch, num_workers=workers, shuffle=shuffle,
                               seed=17, pin_memory=workers > 0)


def scoring(model, loader, *, device, features_required, max_batches=None, expected_stage="second_round"):
    dataset = loader.dataset
    if (dataset.stage, dataset.role, dataset.partition, dataset.purpose) != (expected_stage, "train", "train", "scoring"):
        raise MethodError(f"scoring accepts {expected_stage} train only")
    model.eval()
    ids, losses, features, probabilities = [], [], [], []
    loader.reset(0)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            logits, feature = model.forward_with_features(batch["image"].to(device))
            logits = logits.float()
            ids.extend(batch["sample_id"])
            losses.extend(F.cross_entropy(logits, batch["label_index"].to(device), reduction="none").cpu().tolist())
            probabilities.append(logits.softmax(1).cpu())
            if features_required:
                features.append(feature.float().cpu())
            if max_batches is not None and i + 1 >= max_batches:
                break
    if not ids or len(ids) != len(set(ids)):
        raise MethodError("empty or duplicate scoring pass")
    return ids, losses, torch.cat(features).numpy() if features else None, torch.cat(probabilities)


def evaluate(model, loader, *, device, classes, training_counts, max_batches=None, expected_stage="second_round"):
    if (loader.dataset.stage, loader.dataset.role, loader.dataset.partition) != (expected_stage, "train", "dev"):
        raise MethodError("explicit dev replay only; no confirm/test")
    model.eval()
    loader.reset(0)
    predictions, labels, ids, logits_list = [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            logits = model(batch["image"].to(device)).float().cpu()
            if not torch.isfinite(logits).all():
                raise MethodError("nonfinite dev logits")
            logits_list.append(logits)
            predictions.extend(logits.argmax(1).tolist())
            labels.extend(batch["label_index"].tolist())
            ids.extend(batch["sample_id"])
            if max_batches is not None and i + 1 >= max_batches:
                break
    metrics = evaluate_classification(labels, predictions, total_classes=classes, training_counts=training_counts)
    return {"metrics": metrics.to_dict(), "predictions": [dict(sample_id=s, label=y, prediction=p)
            for s, y, p in zip(ids, labels, predictions)], "logits": torch.cat(logits_list)}


class Trainer:
    """Only constructed by guarded commands; tests supply bounded synthetic data.

    Checkpoints are committed after scoring AND dev validation. A crash within
    an epoch restarts that epoch from the prior complete snapshot and RNG.
    """
    def __init__(self, student, state, *, identity, device="cpu", precision="fp32", effective_batch=128):
        self.student, self.state = student.to(device), state
        self.identity = copy.deepcopy(identity)
        self.device, self.precision, self.effective_batch = device, precision, effective_batch
        if precision == "fp16" and not str(device).startswith("cuda"):
            raise ValueError("FP16 requires CUDA")
        self.auxiliary = SNSCL(student, state.observed.shape[1]).to(device) if state.method == "snscl" else None
        self.optimizer = torch.optim.AdamW(optimizer_groups(student, self.auxiliary))
        self.scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
        self.updates, self.samples, self.skipped = 0, 0, 0
        self.history, self.best_key = [], None

    def update_window(self, batches, *, epoch, fraction):
        if not batches:
            raise ValueError("empty optimizer update")
        total = sum(len(b["sample_id"]) for b in batches)
        self.optimizer.zero_grad(set_to_none=True)
        self.student.train()
        for group in self.optimizer.param_groups:
            group["lr"] = group["initial_lr"] * schedule(epoch + fraction)
        active = self.auxiliary is not None and epoch >= 5
        aux_rng = self.auxiliary.generator.get_state() if active else None
        pending_images, pending_labels, pending_weights = [], [], []
        loss_sum = 0.
        for batch in batches:
            indices = self.state.indices(batch["sample_id"])
            observed = batch["label_index"].cpu()
            if not torch.equal(observed, self.state.labels[indices]):
                raise MethodError("observed labels changed")
            images = batch["image"].to(self.device)
            with torch.autocast(device_type="cuda" if str(self.device).startswith("cuda") else "cpu",
                                dtype=torch.float16, enabled=self.precision == "fp16"):
                logits, features = self.student.forward_with_features(images)
                if active:
                    soft = self.state.soft[indices].to(self.device)
                    hard = soft.argmax(1)
                    losses = -(soft * F.log_softmax(logits.float(), dim=1)).sum(1)
                    losses = losses + self.auxiliary.loss(features, hard)
                    pending_images.append(batch["image"].cpu())
                    pending_labels.append(hard.detach())
                    pending_weights.append(self.state.weights[indices].to(self.device))
                else:
                    losses = F.cross_entropy(logits.float(), observed.to(self.device), reduction="none")
                loss = losses.sum() / total
            if not torch.isfinite(loss):
                raise MethodError("nonfinite training objective")
            self.scaler.scale(loss).backward()
            loss_sum += float(loss.detach())
        old_scale = self.scaler.get_scale()
        if self.precision == "fp32" and any(p.grad is not None and not torch.isfinite(p.grad).all()
                                           for g in self.optimizer.param_groups for p in g["params"]):
            raise MethodError("nonfinite FP32 gradient")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        successful = self.scaler.get_scale() >= old_scale
        if successful:
            self.updates += 1
            if active:
                self.auxiliary.successful_update(self.student, pending_images, pending_labels, pending_weights)
        else:
            self.skipped += 1
            if active:
                self.auxiliary.generator.set_state(aux_rng)
        self.samples += total
        return loss_sum, successful

    def train_epoch(self, loader, *, epoch, max_updates=None):
        loader.reset(epoch)
        batches, count, losses, windows = [], 0, [], 0
        for batch in loader:
            batches.append(batch)
            count += len(batch["sample_id"])
            if count >= self.effective_batch or loader.exhausted:
                loss, _ = self.update_window(batches, epoch=epoch, fraction=loader.position / len(loader.order))
                losses.append(loss)
                batches, count = [], 0
                windows += 1
                if max_updates is not None and windows >= max_updates:
                    break
        return losses

    def checkpoint(self):
        return {"version": VERSION, "identity": self.identity, "student": self.student.state_dict(),
                "optimizer": self.optimizer.state_dict(), "scaler": self.scaler.state_dict(),
                "method": self.state.state_dict(), "auxiliary": self.auxiliary.state_dict() if self.auxiliary else None,
                "method_rng": self.auxiliary.generator.get_state() if self.auxiliary else None,
                "rng": _rng_state(), "updates": self.updates, "samples": self.samples, "skipped": self.skipped,
                "history": self.history, "best_key": self.best_key,
                "scheduler": {"total_epochs": 30, "completed_epochs": self.state.completed_epochs},
                "sampler": {"seed": 17, "next_epoch": self.state.completed_epochs,
                            "selected_ids": [s for s, keep in zip(self.state.ids, self.state.selected) if keep]}}

    def restore(self, payload):
        if set(payload) != set(self.checkpoint()) or payload["version"] != VERSION or payload["identity"] != self.identity:
            raise ValueError("round2 checkpoint identity/state mismatch")
        self.state.load_state_dict(payload["method"])
        epoch = self.state.completed_epochs
        if payload["scheduler"] != {"total_epochs": 30, "completed_epochs": epoch} or payload["sampler"] != self.checkpoint()["sampler"]:
            raise ValueError("scheduler/sampler phase mismatch")
        if len(payload["history"]) != epoch or any(payload[k] < 0 for k in ("updates", "samples", "skipped")):
            raise ValueError("invalid progress history")
        required_rng = {"python", "numpy", "torch"} | ({"cuda"} if str(self.device).startswith("cuda") else set())
        if not required_rng <= set(payload["rng"]):
            raise ValueError("missing RNG state")
        if self.auxiliary is not None:
            if payload["auxiliary"] is None or payload["method_rng"] is None:
                raise ValueError("SNSCL momentum/queue/RNG state missing")
            self.auxiliary.load_state_dict(payload["auxiliary"], strict=True)
            queue = self.auxiliary.queue
            if (queue.count < 0).any() or (queue.count > 32).any() or (queue.pointer < 0).any() or (queue.pointer >= 32).any() or not torch.isfinite(queue.features).all():
                raise ValueError("invalid SNSCL queue state")
            self.auxiliary.generator.set_state(payload["method_rng"])
        elif payload["auxiliary"] is not None or payload["method_rng"] is not None:
            raise ValueError("unexpected auxiliary state")
        self.student.load_state_dict(payload["student"], strict=True)
        expected_groups = self.optimizer.state_dict()["param_groups"]
        loaded_groups = payload["optimizer"]["param_groups"]
        if len(expected_groups) != len(loaded_groups) or any(
                any(expected[k] != loaded.get(k) for k in ("initial_lr", "weight_decay", "group_name", "params"))
                for expected, loaded in zip(expected_groups, loaded_groups)):
            raise ValueError("optimizer parameter ownership or fixed recipe changed")
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scaler.load_state_dict(payload["scaler"])
        for name in ("updates", "samples", "skipped", "history", "best_key"):
            setattr(self, name, copy.deepcopy(payload[name]))
        _restore_rng_state(payload["rng"])


def require_machine(machine, *, stage="second_round"):
    policy = machine_policy(machine)
    policy.validate(RunConfig(stage=stage, execution_mode="formal"))
    if not torch.cuda.is_available():
        raise ValueError("CUDA training machine required")


def require_matching_control(config):
    if config.get("group") not in {"4060-a", "4060-b", "cloud4090-lora", "cloud4090-full"} or config["method"] == "ce":
        return
    from ..contracts import read_json
    name = {"4060-a": "4060-A-CE", "4060-b": "4060-B-CE",
            "cloud4090-lora": "C4090-LORA-CE", "cloud4090-full": "C4090-FULL-CE"}[config["group"]]
    root = Path(config["output"]).parent / name
    control = read_json(root / "resolved.json")
    for field in ("asset_digest", "head_sha256", "receipt_digest", "engineering", "recipe", "adaptation"):
        if control.get(field) != config.get(field):
            raise ValueError("matching local CE control identity differs")
    result = read_json(root / "result.json")
    if result.get("status") != "paused_at_epoch10" or result.get("completed_epochs") != 10 or result.get("checkpoint_sha256") != file_sha256(root / "last.pt"):
        raise ValueError("matching local CE control must finish epoch10 before candidate")


def run(config_path, *, machine, resume=False, startup=False, stop_after_epoch=10):
    require_machine(machine) if not startup else None
    config, assets = check_config(config_path, ready=not startup)
    if assets is None:
        raise ValueError(config["status"])
    if not startup:
        require_matching_control(config)
    return _run_prepared(config, assets, resume=resume, startup=startup, stop_after_epoch=stop_after_epoch)


def _run_prepared(config, assets, *, resume=False, startup=False, student_factory=None,
                  expected_stage="second_round", purpose="formal", stop_after_epoch=10):
    """Shared numerical loop; public callers own stage and machine admission."""
    zero_policy = config["recipe"].get("zero_selection_policy", "error")
    if stop_after_epoch not in {1, 5, 6, 10}:
        raise ValueError("only epoch 1/5/6/10 observation boundaries are supported")
    if zero_policy != "error" and expected_stage != "preliminary":
        raise ValueError("abstention recipe is restricted to the preliminary pilot")
    seed_everything(17)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor, _ = (student_factory or student_for)(assets, config["adaptation"])
    with ExitStack() as stack:
        train = assets.dataset("train", processor, online=True)
        score = assets.dataset("train", processor, purpose="scoring")
        dev = assets.dataset("dev", processor)
        for dataset in (train, score, dev):
            stack.callback(dataset.close)
        if startup:
            train.records, score.records, dev.records = train.records[:4], score.records[:4], dev.records[:2]
        state = MethodState(config["method"], [r.sample_id for r in train.records],
                            [train.class_to_index[r.class_id] for r in train.records], len(assets.class_map.id_to_index),
                            zero_selection_policy=zero_policy)
        identity = {"configuration": config["digest"], "source": current_code_revision(), "purpose": "startup" if startup else purpose}
        trainer = Trainer(model, state, identity=identity, device=device, precision="fp32" if startup else "fp16",
                          effective_batch=2 if startup else 128)
        engineering = {"microbatch": 1, "workers": 0} if startup else config["engineering"]
        root = Path(config["output"] + "-startup" if startup else config["output"])
        if resume:
            if startup:
                raise ValueError("startup never resumes into training")
            trainer.restore(torch.load(root / "last.pt", weights_only=False, map_location="cpu"))
        else:
            root.mkdir(parents=True, exist_ok=False)
            write_json(root / "resolved.json", config)
        try:
            if startup:
                with loader_for(train, batch=1, shuffle=True) as loader:
                    trainer.train_epoch(loader, epoch=0, max_updates=2)
                result = {"status": "startup_only", "updates": trainer.updates, "samples": trainer.samples}
                if trainer.updates != 2 or trainer.samples > 4:
                    raise ValueError("startup did not complete exactly two updates within four samples")
                write_json(root / "result.json", result)
                return result
            score_loader = stack.enter_context(loader_for(score, batch=64, workers=engineering["workers"]))
            dev_loader = stack.enter_context(loader_for(dev, batch=64, workers=engineering["workers"]))
            counts = Counter(state.labels.tolist())
            if not resume and state.method in {"turn", "fine"}:
                state.rescore(*scoring(model, score_loader, device=device, features_required=state.method == "fine",
                                      expected_stage=expected_stage), completed_epochs=0)
            if state.completed_epochs >= 10:
                raise ValueError("epoch10 pause reached; no continuation authorized in this release")
            if state.completed_epochs >= stop_after_epoch:
                raise ValueError("observation boundary already reached")
            for epoch in range(state.completed_epochs, stop_after_epoch):
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                prior_samples, prior_updates = trainer.samples, trainer.updates
                began = clock(device)
                selected = copy.copy(train)
                selected.records = tuple(r for r, keep in zip(train.records, state.selected) if keep)
                with loader_for(selected, batch=engineering["microbatch"], workers=engineering["workers"], shuffle=True) as loader:
                    losses = trainer.train_epoch(loader, epoch=epoch)
                trained = clock(device)
                scored_this_epoch = state.method in {"turn", "fine"} or state.method == "snscl" and epoch >= 4
                if scored_this_epoch:
                    state.rescore(*scoring(model, score_loader, device=device, features_required=state.method == "fine",
                                          expected_stage=expected_stage), completed_epochs=epoch + 1)
                else:
                    state.completed_epochs = epoch + 1
                scored = clock(device)
                evaluation = evaluate(model, dev_loader, device=device, classes=len(assets.class_map.id_to_index),
                                      training_counts=counts, expected_stage=expected_stage)
                evaluated = clock(device)
                metrics = evaluation["metrics"]
                key = (metrics["macro_recall"], metrics["micro_top1"], -(epoch + 1))
                best = trainer.best_key is None or key > trainer.best_key
                if best:
                    trainer.best_key = key
                log = {"epoch": epoch + 1, "metrics": metrics, "mean_update_loss": sum(losses) / len(losses),
                       "label_quality": "noisy_proxy", "train_samples": trainer.samples - prior_samples,
                       "optimizer_updates": trainer.updates - prior_updates,
                       "updates": trainer.updates, "samples": trainer.samples, "amp_skips": trainer.skipped,
                       "scoring_samples": len(score) if scored_this_epoch else 0,
                       "dev_samples": len(dev), "selection": state.report, "selected": int(state.selected.sum()),
                       "train_seconds": trained - began, "scoring_seconds": scored - trained,
                       "eval_seconds": evaluated - scored, "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                       "queue_counts": trainer.auxiliary.queue.count.tolist() if trainer.auxiliary else None,
                       "queue_pointers": trainer.auxiliary.queue.pointer.tolist() if trainer.auxiliary else None,
                       "soft_labels_changed": int((state.soft.argmax(1) != state.labels).sum()),
                       "reliability_weight_mean": float(state.weights.mean()),
                       "soft_label_entropy": float(-(state.soft * state.soft.clamp_min(1e-12).log()).sum(1).mean())}
                trainer.history.append(log)
                if state.method != "ce":
                    write_json(root / f"selection-epoch-{epoch + 1:02d}.json", {
                        "sample_ids": state.ids, "selected": state.selected.tolist(),
                        "weights": state.weights.tolist() if state.method == "snscl" else None,
                        "report": state.report, "labels_unchanged": True})
                atomic_save(evaluation, root / f"dev-epoch-{epoch + 1:02d}.pt")
                write_json(root / f"dev-epoch-{epoch + 1:02d}.json", {k: v for k, v in evaluation.items() if k != "logits"})
                digest = atomic_save(trainer.checkpoint(), root / "last.pt")
                if best:
                    publish_best(root)
                saved = clock(device)
                write_json(root / f"epoch-{epoch + 1:02d}.json", {**log, "save_seconds": saved - evaluated,
                    "epoch_seconds": saved - began, "checkpoint_sha256": digest})
            # Workers must terminate cleanly before reporting a successful pause.
            stack.close()
            result = {"status": "paused_at_epoch10" if state.completed_epochs == 10 else "paused_at_observation",
                      "full_epochs": 30, "completed_epochs": state.completed_epochs,
                      "best": trainer.best_key, "last": trainer.history[-1]["metrics"],
                      "best_epoch": -trainer.best_key[2], "best_metrics": trainer.history[-trainer.best_key[2] - 1]["metrics"],
                      "checkpoint_sha256": file_sha256(root / "last.pt"), "best_sha256": file_sha256(root / "best.pt")}
            write_json(root / "result.json", result)
            return result
        except BaseException as exc:
            record_failure(root, exc)
            raise


def replay(config_path, *, machine, output):
    require_machine(machine)
    config, assets = check_config(config_path)
    return _replay_prepared(config, assets, output=output)


def _replay_prepared(config, assets, *, output, student_factory=None, expected_stage="second_round", purpose="formal", device="cuda"):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(Path(config["output"]) / "last.pt", weights_only=False, map_location="cpu")
    if checkpoint["identity"] != {"configuration": config["digest"], "source": current_code_revision(), "purpose": purpose}:
        raise ValueError("student export checkpoint identity mismatch")
    epoch = checkpoint["method"]["completed_epochs"]
    reference = torch.load(Path(config["output"]) / f"dev-epoch-{epoch:02d}.pt", weights_only=False, map_location="cpu")
    # Export explicitly excludes text parameters and every auxiliary module.
    exported = {k: v for k, v in checkpoint["student"].items()
                if k.startswith(("encoder.clip_model.vision_model.", "encoder.clip_model.visual_projection.", "classifier."))}
    export_identity = {"version": VERSION, "stage": expected_stage, "purpose": purpose,
                       "asset_digest": assets.descriptor["digest"], "adaptation": config["adaptation"],
                       "checkpoint_sha256": file_sha256(Path(config["output"]) / "last.pt")}
    atomic_save({"identity": export_identity, "student": exported}, root / "student.pt")
    student, processor, _ = (student_factory or student_for)(assets, config["adaptation"])
    payload = torch.load(root / "student.pt", weights_only=True, map_location="cpu")
    expected = {k for k in student.state_dict() if k.startswith(("encoder.clip_model.vision_model.", "encoder.clip_model.visual_projection.", "classifier."))}
    if set(payload["student"]) != expected or payload["identity"] != export_identity:
        raise ValueError("exported student identity/parameter coverage mismatch")
    student.load_state_dict({**student.state_dict(), **payload["student"]})
    student.to(device).eval()
    dataset = assets.dataset("dev", processor)
    training = assets.dataset("train")
    try:
        with loader_for(dataset, batch=64, workers=config["engineering"]["workers"]) as loader:
            result = evaluate(student, loader, device=device, classes=len(assets.class_map.id_to_index), expected_stage=expected_stage,
                              training_counts=Counter(assets.class_map.index_for(r.class_id) for r in training.records))
        passed = result["predictions"] == reference["predictions"] and torch.allclose(result["logits"], reference["logits"], atol=1e-6, rtol=1e-5)
        report = {"partition": "dev", "passed": passed, "epoch": epoch, "identity": export_identity,
                  "max_abs_logit_difference": float((result["logits"] - reference["logits"]).abs().max()),
                  "student_sha256": file_sha256(root / "student.pt")}
        write_json(root / "replay.json", report)
        if not passed:
            raise ValueError("student dev replay mismatch")
        return report
    finally:
        dataset.close()
        training.close()
