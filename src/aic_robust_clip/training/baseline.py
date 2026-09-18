"""Common B01/B04 training loop with an explicit bounded smoke mode."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from itertools import count
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import math
import time
from typing import Any, Iterable, Mapping

from ..contracts import CheckpointMetadata, RunConfig, sha256_json, write_json
from ..data.loading import StatefulBatchLoader
from ..metrics import ClassificationMetrics, evaluate_classification
from ..models.clip import ClipDependencyError, torch, nn
from ..models.classifier import LinearClassifier
from ..runtime import LOCAL_POLICY, RuntimePolicy, assert_bounded_startup, resolve_run_config, seed_everything
from ..performance import transfer_tensor
from .checkpoint import load_checkpoint, save_checkpoint
from .objectives import combined_wpi_loss, gce_loss, sce_loss, preservation_loss
from .optimization import optimizer_groups, warmup_cosine_factor, selection_key
from .reliability import ReliabilityState, observed_prior


class TrainingError(RuntimeError):
    """Raised for an invalid trainer/model/data combination."""


if torch is not None:
    _ModuleBase = nn.Module
else:  # pragma: no cover
    class _ModuleBase:
        pass


class FrozenFeatureBaseline(_ModuleBase):
    """B01: a linear classifier over precomputed normalized CLIP features."""

    def __init__(self, feature_dim: int, class_count: int) -> None:
        if torch is None:
            raise ClipDependencyError("torch is required for training")
        super().__init__()
        self.classifier = LinearClassifier(feature_dim, class_count)

    def forward(self, features: Any) -> Any:
        return self.classifier(features)


class OnlineFrozenBaseline(_ModuleBase):
    """B04: online preprocessing with a frozen CLIP image encoder."""

    def __init__(self, encoder: Any, feature_dim: int, class_count: int) -> None:
        if torch is None:
            raise ClipDependencyError("torch is required for training")
        super().__init__()
        self.encoder = encoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.classifier = LinearClassifier(feature_dim, class_count)

    def train(self, mode: bool = True) -> "OnlineFrozenBaseline":
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, pixel_values: Any) -> Any:
        features = self.encoder(pixel_values, no_grad=True)
        return self.classifier(features)

    @property
    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)


@dataclass(frozen=True)
class TrainConfig:
    run: RunConfig
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    accumulation_steps: int = 1
    profile: str = "startup-check"
    objective: str = "ce"
    gce_q: float = 0.7
    weighting: bool = False
    lambda_preserve: float = 0.0
    prior_tau: float = 0.0
    scheduler: str = "constant"
    scheduler_step_size: int = 1
    scheduler_gamma: float = 0.9
    lora_learning_rate: float = 1e-4
    warmup_epochs: int = 1

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.learning_rate <= 0 or self.weight_decay < 0 or self.accumulation_steps <= 0:
            raise ValueError("invalid training configuration")
        if self.run.execution_mode == "smoke" and self.epochs > 1:
            raise ValueError("smoke training must be a bounded startup check, not full epochs")
        if self.objective not in {"ce", "gce", "sce"} or self.gce_q < 0 or self.lambda_preserve < 0:
            raise ValueError("invalid research objective")
        if self.objective != "ce" and (self.weighting or self.lambda_preserve or self.prior_tau):
            raise ValueError("GCE/SCE controls must have W/P/I disabled")
        if self.scheduler not in {"constant", "step", "warmup_cosine"} or self.scheduler_step_size <= 0 or not 0 < self.scheduler_gamma <= 1:
            raise ValueError("invalid scheduler configuration")
        if self.lora_learning_rate <= 0 or self.warmup_epochs < 0:
            raise ValueError("invalid optimizer/scheduler parameters")

    @classmethod
    def from_run(cls, run: RunConfig) -> "TrainConfig":
        parameters = dict(run.parameters)
        formal_epochs = parameters.pop("formal_epochs", 1)
        parameters.setdefault("epochs", formal_epochs if run.execution_mode == "formal" else 1)
        return cls(run=run, **parameters)

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass
class BaselineResult:
    updates: int
    samples: int
    losses: list[float] = field(default_factory=list)
    dev_metrics: ClassificationMetrics | None = None
    dev_predictions: list[int] = field(default_factory=list)
    dev_labels: list[int] = field(default_factory=list)
    optimizer_updated: bool = False
    stopped_by_limit: bool = False
    reference_samples: int = 0
    scoring_samples: int = 0
    epoch_logs: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_hashes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "updates": self.updates,
            "samples": self.samples,
            "losses": self.losses,
            "optimizer_updated": self.optimizer_updated,
            "stopped_by_limit": self.stopped_by_limit,
            "dev_metrics": self.dev_metrics.to_dict() if self.dev_metrics else None,
            "reference_samples": self.reference_samples,
            "scoring_samples": self.scoring_samples,
            "epoch_logs": self.epoch_logs,
            "checkpoint_hashes": self.checkpoint_hashes,
        }


def _unpack_batch(batch: Any) -> tuple[Any, Any]:
    if hasattr(batch, "image") and hasattr(batch, "label_index"):
        return batch.image, batch.label_index
    if isinstance(batch, Mapping):
        image = batch.get("image", batch.get("features", batch.get("pixel_values")))
        labels = batch.get("label_index", batch.get("labels", batch.get("label")))
        if image is None or labels is None:
            raise TrainingError("batch mapping requires image/features and label_index/labels")
        return image, labels
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TrainingError("unsupported batch format")


def _batch_len(labels: Any) -> int:
    if torch is not None and isinstance(labels, torch.Tensor):
        return int(labels.shape[0]) if labels.ndim else 1
    if isinstance(labels, (list, tuple)):
        return len(labels)
    return 1


def _slice_value(value: Any, count: int) -> Any:
    if torch is not None and isinstance(value, torch.Tensor) and value.ndim > 0:
        return value[:count]
    if isinstance(value, (list, tuple)):
        return value[:count]
    return value


def _prepare_labels(labels: Any, device: Any) -> Any:
    if torch is None:
        raise ClipDependencyError("torch is required for training")
    if isinstance(labels, torch.Tensor):
        result = transfer_tensor(labels, device, dtype=torch.long)
    else:
        result = torch.as_tensor(labels, device=device, dtype=torch.long)
    return result.reshape(-1)


def _prepare_images(images: Any, device: Any) -> Any:
    if torch is not None and isinstance(images, torch.Tensor):
        return transfer_tensor(images, device)
    raise TrainingError("training batches must provide torch tensors after preprocessing")


def evaluate_loader(
    model: Any,
    loader: Iterable[Any],
    *,
    total_classes: int,
    training_counts: Mapping[int, int] | None = None,
    max_batches: int | None = None,
    device: Any = None,
    observer: Any = None,
) -> tuple[ClassificationMetrics, list[int], list[int]]:
    if torch is None:
        raise ClipDependencyError("torch is required for evaluation")
    if device is None:
        device = next(model.parameters()).device
    labels: list[int] = []
    predictions: list[int] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        iterator = iter(loader)
        for _ in (range(max_batches) if max_batches is not None else count()):
            if observer is not None:
                observer.begin_step()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            if observer is not None:
                observer.data_ready()
            images, batch_labels = _unpack_batch(batch)
            images = _prepare_images(images, device)
            batch_labels_tensor = _prepare_labels(batch_labels, device)
            if observer is not None:
                observer.cuda_mark("transfer")
            logits = model(images)
            if observer is not None:
                observer.cuda_mark("compute")
            batch_predictions = logits.argmax(dim=-1)
            labels.extend(int(value) for value in batch_labels_tensor.cpu().tolist())
            predictions.extend(int(value) for value in batch_predictions.cpu().tolist())
            if observer is not None:
                observer.end_step(len(batch_labels_tensor))
    if was_training:
        model.train()
    if not labels:
        raise TrainingError("evaluation loader produced no labels")
    return evaluate_classification(labels, predictions, total_classes=total_classes, training_counts=training_counts), predictions, labels


def train_baseline(
    model: Any,
    train_loader: Iterable[Any],
    *,
    config: TrainConfig,
    class_count: int,
    dev_loader: Iterable[Any] | None = None,
    training_counts: Mapping[int, int] | None = None,
    device: str | None = None,
    policy: RuntimePolicy = LOCAL_POLICY,
    training_labels: Mapping[str, int] | None = None,
    reliability: ReliabilityState | None = None,
    scoring_loader: StatefulBatchLoader | None = None,
    reference_encoder: Any | None = None,
    checkpoint_dir: Path | str | None = None,
    checkpoint_metadata: CheckpointMetadata | None = None,
    resume_from: Path | str | None = None,
    stop_after_updates: int | None = None,
    observer: Any = None,
) -> BaselineResult:
    """Common baseline/research loop; resume only at committed update boundaries.

    Smoke limits are cumulative across resume, never renewed by another call.
    A checkpointed stream must be StatefulBatchLoader (delivered-only cursor).
    ``stop_after_updates`` pauses at an absolute update count without turning
    the pause into an epoch boundary or flushing an incomplete accumulation.
    """

    if torch is None:
        raise ClipDependencyError("install torch before running a baseline")
    resolve_run_config(config.run, policy)
    smoke = config.run.execution_mode == "smoke"
    if stop_after_updates is not None and stop_after_updates <= 0:
        raise TrainingError("stop_after_updates must be positive")
    if smoke and stop_after_updates is not None and stop_after_updates >= config.run.max_updates:
        raise TrainingError("pause before the smoke update cap; omit pause to finish the check")
    if smoke:
        for loader in (train_loader, dev_loader, scoring_loader):
            if loader is None:
                continue
            if getattr(loader, "num_workers", 0):
                raise TrainingError("smoke loaders must disable worker prefetch (num_workers=0)")
            if getattr(loader, "batch_size", config.run.batch_size) > config.run.batch_size:
                raise TrainingError("loader batch size exceeds the resolved smoke batch size")
    dataset = getattr(train_loader, "dataset", None)
    if hasattr(dataset, "purpose") and (dataset.purpose != "train" or dataset.partition != "train"
                                        or dataset.role != "train" or dataset.stage != config.run.stage):
        raise TrainingError("training requires the current-stage training partition")
    if dev_loader is not None:
        dev_dataset = getattr(dev_loader, "dataset", None)
        if hasattr(dev_dataset, "purpose") and (dev_dataset.purpose != "dev" or dev_dataset.stage != config.run.stage):
            raise TrainingError("checkpoint selection accepts current-stage dev only")
    if isinstance(train_loader, StatefulBatchLoader) and smoke:
        if train_loader.max_samples is None or train_loader.max_samples > config.run.max_samples:
            raise TrainingError("bound the loader before starting smoke execution")
    if checkpoint_dir is not None or resume_from is not None:
        if not isinstance(train_loader, StatefulBatchLoader) or checkpoint_metadata is None:
            raise TrainingError("checkpoint/resume requires metadata and StatefulBatchLoader")
        if checkpoint_metadata.configuration_digest != config.digest:
            raise TrainingError("checkpoint metadata must identify the resolved TrainConfig")
        for key in ("stage", "manifest_digest", "split_digest", "class_map_digest",
                    "official_weight_id", "official_weight_revision"):
            if getattr(checkpoint_metadata, key) != getattr(config.run, key):
                raise TrainingError(f"checkpoint metadata/config mismatch: {key}")
    if config.weighting or config.prior_tau:
        if not isinstance(train_loader, StatefulBatchLoader) or training_labels is None:
            raise TrainingError("W/I requires a stateful training stream and explicit training labels")
        active_ids = {train_loader.ids[index] for index in train_loader.order}
        if set(training_labels) != active_ids:
            raise TrainingError("W/I labels must cover exactly the active training IDs")
        records = getattr(dataset, "records", None)
        if records is not None:
            expected = {record.sample_id: dataset.class_to_index[record.class_id]
                        for record in records if record.sample_id in active_ids}
            if dict(training_labels) != expected:
                raise TrainingError("W/I labels differ from the training manifest")
    if config.weighting:
        if scoring_loader is None or scoring_loader is train_loader:
            raise TrainingError("W requires a separate deterministic training scoring stream")
        score_dataset = scoring_loader.dataset
        if hasattr(score_dataset, "purpose") and (score_dataset.purpose != "scoring"
                or score_dataset.stage != config.run.stage or score_dataset.partition != "train"):
            raise TrainingError("reliability scoring is restricted to the training partition")
        score_ids = {scoring_loader.ids[index] for index in scoring_loader.order}
        if scoring_loader.shuffle or score_ids != set(training_labels):
            raise TrainingError("scoring must use the exact active training IDs in a fixed order")
        if getattr(dataset, "records", None) is not None:
            expected_records = {item.sample_id: item for item in dataset.records if item.sample_id in score_ids}
            actual_records = {item.sample_id: item for item in getattr(score_dataset, "records", ()) if item.sample_id in score_ids}
            if expected_records != actual_records or dataset.class_to_index != score_dataset.class_to_index:
                raise TrainingError("scoring records/class map differ from training")
        if smoke and (scoring_loader.max_samples is None or scoring_loader.max_samples > config.run.max_samples):
            raise TrainingError("bound the scoring stream before smoke execution")
        if reliability is None:
            reliability = ReliabilityState(tuple(sorted(training_labels)),
                                           {key: str(value) for key, value in training_labels.items()})
        if set(reliability.sample_ids) != set(training_labels) or dict(reliability.observed_class_ids) != {
                key: str(value) for key, value in training_labels.items()}:
            raise TrainingError("reliability state differs from active training labels")
    prior = observed_prior(training_labels.values(), class_count) if config.prior_tau else None
    if config.lambda_preserve:
        if reference_encoder is None or not hasattr(model, "forward_with_features"):
            raise TrainingError("P requires forward_with_features and a separate original encoder")
        if {id(parameter) for parameter in model.parameters()} & {id(parameter) for parameter in reference_encoder.parameters()}:
            raise TrainingError("reference encoder must not share adapted model parameters")
    if resume_from is None:
        seed_everything(config.run.seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise TrainingError("baseline has no trainable parameters")
    optimizer = torch.optim.AdamW(optimizer_groups(model, head_lr=config.learning_rate,
        lora_lr=config.lora_learning_rate, weight_decay=config.weight_decay), betas=(.9, .999), eps=1e-8)
    scheduler = (torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
                 if config.scheduler == "step" else torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0))
    if config.scheduler == "warmup_cosine":
        steps_per_epoch = math.ceil(len(train_loader) / config.accumulation_steps)
        total_steps = config.run.max_updates if smoke else steps_per_epoch * config.epochs
        warmup_steps = min(steps_per_epoch * config.warmup_epochs, max(0, total_steps - 1))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda update: warmup_cosine_factor(
            update, warmup_updates=warmup_steps, total_updates=total_steps))
    result = BaselineResult(updates=0, samples=0)
    epoch = 0
    best_score = (float("-inf"), float("-inf"), 0)
    if resume_from is not None:
        requested = {key: getattr(checkpoint_metadata, key) for key in (
            "stage", "class_map_digest", "manifest_digest", "split_digest",
            "official_weight_id", "official_weight_revision")}
        for key in ("weight_files_digest", "preprocessing_digest", "initialization_digest", "execution_mode"):
            if getattr(checkpoint_metadata, key):
                requested[key] = getattr(checkpoint_metadata, key)
        restored = load_checkpoint(resume_from, model=model, optimizer=optimizer,
                                   scheduler=scheduler, requested=requested,
                                   expected_configuration_digest=config.digest)
        progress = restored["metadata"].progress
        if restored["metadata"].model_family != checkpoint_metadata.model_family:
            raise TrainingError("resume model family differs from requested model")
        epoch = int(progress["epoch"])
        result.updates, result.samples = int(progress["updates"]), int(progress["samples"])
        saved_score = progress["best_score"]
        if not isinstance(saved_score, (list, tuple)) or len(saved_score) != 3:
            raise TrainingError("legacy checkpoint selection policy is incompatible; restart the run")
        best_score = tuple(saved_score)
        result.reference_samples = int(progress.get("reference_samples", 0))
        result.scoring_samples = int(progress.get("scoring_samples", 0))
        train_loader.load_state_dict(restored["sampler_state"])
        if train_loader.epoch != epoch:
            raise TrainingError("checkpoint epoch and stream cursor disagree")
        state = restored["module_state"]
        result.epoch_logs = list(state.get("epoch_logs", []))
        if state.get("prior") != prior:
            raise TrainingError("resume training prior differs")
        if config.weighting:
            restored_reliability = ReliabilityState.from_state_dict(state["reliability"])
            if restored_reliability.sample_ids != reliability.sample_ids or dict(restored_reliability.observed_class_ids) != dict(reliability.observed_class_ids):
                raise TrainingError("resume reliability IDs/labels differ")
            reliability = restored_reliability
        if smoke and (result.updates > config.run.max_updates or result.samples > config.run.max_samples):
            raise TrainingError("checkpoint exceeds smoke budget")
        if smoke and (result.reference_samples > config.run.max_samples or result.scoring_samples > config.run.max_samples):
            raise TrainingError("checkpoint exceeds auxiliary smoke budget")
    if config.lambda_preserve:
        reference_encoder.to(device).eval()
        for parameter in reference_encoder.parameters():
            parameter.requires_grad_(False)

    def persist(name: str) -> None:
        if checkpoint_dir is None:
            return
        persist_started = time.monotonic()
        metadata = replace(checkpoint_metadata, progress={
            "epoch": epoch, "updates": result.updates, "samples": result.samples,
            "best_score": best_score, "reference_samples": result.reference_samples,
            "scoring_samples": result.scoring_samples,
        }, optimizer_state_present=True, scheduler_state_present=True,
            rng_state_present=True, sampler_state_present=True)
        dependencies = {}
        for package in ("torch", "transformers", "Pillow", "numpy"):
            try:
                dependencies[package] = version(package)
            except PackageNotFoundError:
                dependencies[package] = None
        modules = {"prior": prior, "reliability": reliability.state_dict() if config.weighting else None,
                   "configuration": asdict(config), "dependencies": dependencies,
                   "epoch_logs": result.epoch_logs}
        path = Path(checkpoint_dir) / f"{name}.pt"
        result.checkpoint_hashes[name] = save_checkpoint(path, model=model, metadata=metadata,
            optimizer=optimizer, scheduler=scheduler, sampler_state=train_loader.state_dict(), module_state=modules)
        if observer is not None:
            observer.checkpoint_seconds += time.monotonic() - persist_started

    optimizer.zero_grad(set_to_none=True)
    paused = False
    while epoch < config.epochs:
        if smoke and (result.samples >= config.run.max_samples or result.updates >= config.run.max_updates):
            result.stopped_by_limit = True
            break
        if stop_after_updates is not None and result.updates >= stop_after_updates:
            paused = True
            break
        iterator = iter(train_loader)
        exhausted = False
        while True:
            # Check before next(): no extra image decode, transform or worker fetch.
            if smoke and (result.samples >= config.run.max_samples or result.updates >= config.run.max_updates):
                result.stopped_by_limit = True
                break
            # Buffer only CPU inputs for one effective batch, never graphs.
            # Knowing its exact size/weight mass makes incomplete and weighted
            # accumulation equivalent to the corresponding unsplit objective.
            if observer is not None:
                observer.begin_step()
            group = []
            group_samples = 0
            for _ in range(config.accumulation_steps):
                remaining = config.run.max_samples - result.samples - group_samples if smoke else None
                if remaining == 0:
                    break
                try:
                    batch = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                images, labels = _unpack_batch(batch)
                count = _batch_len(labels)
                if remaining is not None:
                    count = min(count, remaining)
                    images, labels = _slice_value(images, count), _slice_value(labels, count)
                if count <= 0:
                    raise TrainingError("empty training batch")
                weights = None
                if config.weighting:
                    ids = batch["sample_id"][:count]
                    if any(sample_id not in reliability.weights for sample_id in ids):
                        raise TrainingError("non-training sample entered reliability weighting")
                    weights = [reliability.weights[sample_id] for sample_id in ids]
                group.append((images, labels, count, weights))
                group_samples += count
            if not group:
                break
            if observer is not None:
                observer.data_ready()
            mass = sum(sum(weights) if weights is not None else count for _, _, count, weights in group)
            detached_losses = []
            for images, labels, count, weights in group:
                images, labels = _prepare_images(images, device), _prepare_labels(labels, device)
                if observer is not None:
                    observer.cuda_mark("transfer")
                features = frozen_features = None
                if config.lambda_preserve:
                    logits, features = model.forward_with_features(images)
                    with torch.no_grad():
                        frozen_features = reference_encoder(images)
                    result.reference_samples += count
                else:
                    logits = model(images)
                if config.objective == "gce":
                    supervised = gce_loss(logits, labels, q=config.gce_q)
                elif config.objective == "sce":
                    supervised = sce_loss(logits, labels)
                else:
                    supervised = combined_wpi_loss(logits, labels, sample_weights=weights, prior=prior,
                                                  tau=config.prior_tau)
                contribution = supervised * ((sum(weights) if weights is not None else count) / max(mass, 1e-12))
                if config.lambda_preserve:
                    contribution = contribution + config.lambda_preserve * preservation_loss(features, frozen_features) * count / group_samples
                if not torch.isfinite(contribution):
                    raise TrainingError("nonfinite loss; run stopped without retry")
                contribution.backward()
                detached_losses.append(contribution.detach())
                result.samples += count
                if observer is not None:
                    observer.cuda_mark("compute")
            finite_gradients = [torch.isfinite(parameter.grad).all() for parameter in trainable if parameter.grad is not None]
            if finite_gradients and not torch.stack(finite_gradients).all():
                raise TrainingError("nonfinite gradient; run stopped without retry")
            result.losses.extend(torch.stack(detached_losses).cpu().tolist())
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            result.updates += 1
            result.optimizer_updated = True
            if observer is not None:
                observer.cuda_mark("optimizer")
                observer.end_step(group_samples)
            if stop_after_updates is not None and result.updates >= stop_after_updates:
                paused = True
                break
            if smoke and result.updates >= config.run.max_updates:
                result.stopped_by_limit = True
                break
        if paused:
            break
        exhausted = exhausted or (isinstance(train_loader, StatefulBatchLoader) and train_loader.exhausted)
        if config.weighting:
            scoring_loader.reset(0)
            losses_by_id: dict[str, float] = {}
            model.eval()
            with torch.no_grad():
                for batch in scoring_loader:
                    images, labels = _unpack_batch(batch)
                    logits = model(_prepare_images(images, device))
                    scores = torch.nn.functional.cross_entropy(logits, _prepare_labels(labels, device), reduction="none")
                    for sample_id, value in zip(batch["sample_id"], scores.cpu().tolist()):
                        if sample_id in losses_by_id:
                            raise TrainingError("scoring stream repeated a sample")
                        losses_by_id[sample_id] = value
                    result.scoring_samples += len(scores)
            model.train()
            if exhausted:
                reliability.finish_epoch(losses_by_id)
            else:
                # A partial smoke pass is not a completed formal warm-up epoch.
                reliability.update_losses(losses_by_id)
                reliability.next_epoch_weights()
        selected = False
        if dev_loader is not None:
            if isinstance(dev_loader, StatefulBatchLoader):
                dev_loader.reset(0)
            result.dev_metrics, result.dev_predictions, result.dev_labels = evaluate_loader(
                model, dev_loader, total_classes=class_count, training_counts=training_counts,
                max_batches=config.run.max_eval_batches if smoke else None, device=device)
            score = selection_key(result.dev_metrics, epoch)
            selected = score > best_score
            best_score = max(best_score, score)
        result.epoch_logs.append({"epoch": epoch, "complete": exhausted,
            "updates": result.updates, "samples": result.samples,
            "dev_metrics": result.dev_metrics.to_dict() if result.dev_metrics else None})
        if checkpoint_dir is not None:
            write_json(Path(checkpoint_dir) / "epochs.json", result.epoch_logs)
            if dev_loader is not None and hasattr(getattr(dev_loader, "dataset", None), "records"):
                records = dev_loader.dataset.records
                write_json(Path(checkpoint_dir) / f"dev-epoch-{epoch:04d}.json", [
                    {"sample_id": record.sample_id, "label": label, "prediction": prediction}
                    for record, label, prediction in zip(records, result.dev_labels, result.dev_predictions)])
        if exhausted:
            epoch += 1
            if isinstance(train_loader, StatefulBatchLoader):
                train_loader.reset(epoch)
        if selected:
            persist("best")
        persist("last")
        if result.stopped_by_limit or smoke:
            break
    if paused:
        persist("last")
    if smoke:
        assert_bounded_startup(updates=result.updates, samples=result.samples, config=config.run)
        if result.reference_samples > config.run.max_samples or result.scoring_samples > config.run.max_samples:
            raise TrainingError("auxiliary smoke passes exceeded their sample bounds")
    return result
