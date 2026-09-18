"""Bounded, remote-only FP32 measurements; never a model-selection result."""
from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack
from itertools import islice
from pathlib import Path
import os
import time

from .contracts import write_json
from .configuration import load_config, prepare
from .data.cache import feature_batches
from .environment import environment_report
from .models.clip import torch
from .runtime import RuntimeLimitError, current_code_revision
from .training.baseline import evaluate_loader, train_baseline


class StepTimer:
    """CUDA events are read after the window, never synchronized per region.

    Region times include stream idle/host dispatch gaps; they are not kernel
    profiler timings. End-to-end wall time is the throughput denominator.
    """

    def __init__(self, warmup_steps, measure_steps, *, cuda=True):
        self.warmup_steps, self.measure_steps, self.cuda = warmup_steps, measure_steps, cuda
        self.steps = self.samples = 0
        self.data_wait_seconds = self.checkpoint_seconds = 0.
        self.cold_started = time.monotonic()
        self.measure_started = self.measure_finished = None
        self.cold_seconds = None
        self.events = []
        self._last_event = None

    def _sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    def begin_step(self):
        if self.steps >= self.warmup_steps + self.measure_steps:
            raise RuntimeError("benchmark step limit reached")
        if self.steps == self.warmup_steps and self.measure_started is None:
            self._sync()
            self.measure_started = time.monotonic()
            self.cold_seconds = self.measure_started - self.cold_started
        self._step_started = time.monotonic()
        self._last_event = None

    def data_ready(self):
        if self.steps >= self.warmup_steps:
            self.data_wait_seconds += time.monotonic() - self._step_started
        if self.cuda:
            self._last_event = torch.cuda.Event(enable_timing=True)
            self._last_event.record()

    def cuda_mark(self, name):
        if self.cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            if self.steps >= self.warmup_steps:
                self.events.append((name, self._last_event, event))
            self._last_event = event

    def end_step(self, samples):
        if self.steps >= self.warmup_steps:
            self.samples += samples
        self.steps += 1
        if self.steps == self.warmup_steps + self.measure_steps:
            self._sync()
            self.measure_finished = time.monotonic()

    def report(self):
        if self.measure_finished is None:
            raise ValueError("insufficient batches for the requested benchmark window")
        seconds = self.measure_finished - self.measure_started
        regions = defaultdict(float)
        for name, start, end in self.events:
            regions[name] += start.elapsed_time(end) / 1000
        return {"warmup_steps": self.warmup_steps, "measured_steps": self.measure_steps,
                "measured_samples": self.samples, "cold_start_and_warmup_seconds": self.cold_seconds,
                "measurement_seconds": seconds, "seconds_per_step": seconds / self.measure_steps,
                "samples_per_second": self.samples / seconds,
                "data_wait_seconds": self.data_wait_seconds, "cuda_region_seconds": dict(regions),
                "checkpoint_write_seconds": self.checkpoint_seconds,
                "timing_note": "CUDA regions include stream idle/dispatch gaps; use wall throughput, not their sum"}


def validate_request(phase, warmup_steps, measure_steps):
    if phase not in {"cache", "train", "eval"}:
        raise ValueError("benchmark phase must be cache, train or eval; never confirm/test")
    if type(warmup_steps) is not int or not 0 <= warmup_steps <= 5:
        raise ValueError("warmup_steps must be an integer from 0 to 5")
    if type(measure_steps) is not int or not 1 <= measure_steps <= 20:
        raise ValueError("measure_steps must be an integer from 1 to 20")


def benchmark_command(config_path, phase, output, *, warmup_steps=2, measure_steps=10):
    validate_request(phase, warmup_steps, measure_steps)
    config = load_config(config_path)
    if config.get("execution_mode") != "formal" or config.get("device") != "cuda":
        raise RuntimeLimitError("benchmark requires an enrolled remote CUDA formal configuration")
    if config["recipe"] not in {"B03", "B04"}:
        raise ValueError("v1 benchmark accepts B03/B04 controls only")
    root = Path(output)
    if root.exists():
        raise FileExistsError(root)
    ctx = prepare(config_path)  # host enrollment, full artifact identity checks
    if not ctx.policy.allow_formal or not torch.cuda.is_available():
        raise RuntimeLimitError("benchmark requires an enrolled CUDA training host")
    if any(getattr(ctx.train, name) for name in ("weighting", "lambda_preserve", "prior_tau")) or ctx.train.objective != "ce":
        raise ValueError("v1 benchmark measures unmodified CE controls only")
    effective = ctx.run.batch_size * ctx.train.accumulation_steps
    if (effective != config.get("effective_batch_size", 128) or effective > 128
            or ctx.performance.cache_batch_size > 128 or ctx.performance.eval_batch_size > 128):
        raise ValueError("benchmark requires consistent effective batch size and batches at most 128")
    from .workflow import dataset_for, load_bundle, metadata_for, stream, training_components
    steps = warmup_steps + measure_steps
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "benchmark-only.json", {"kind": "benchmark_only", "phase": phase, "resumable": False})
    write_json(root / "resolved.json", ctx.summary())
    started = time.monotonic()
    timer = StepTimer(warmup_steps, measure_steps)
    try:
        torch.cuda.reset_peak_memory_stats()
        with ExitStack() as resources:
            if phase == "cache":
                bundle = load_bundle(ctx)
                model = bundle.encoder.to("cuda").eval()
                dataset = dataset_for(ctx, "train", bundle.processor)
                resources.callback(dataset.close)
                loader = resources.enter_context(stream(ctx, dataset, shuffle=False,
                    batch_size=ctx.performance.cache_batch_size))
                if len(loader) < steps:
                    raise ValueError("not enough training batches for cache benchmark")
                batches = feature_batches(loader, model, device="cuda", observer=timer)
                try:
                    for _ in islice(batches, steps):
                        pass  # no persistent feature shards or cache index
                finally:
                    batches.close()
            else:
                model, train, dev, scoring, reference, initialization = training_components(ctx)
                for dataset in (train, dev, scoring):
                    if hasattr(dataset, "close"):
                        resources.callback(dataset.close)
                dataset = train if phase == "train" else dev
                loader = resources.enter_context(stream(ctx, dataset, shuffle=phase == "train"))
                if phase == "train":
                    # Never cross an epoch: no validation/selection/auxiliary pass.
                    if len(train) <= steps * effective:
                        raise ValueError("training benchmark must stop strictly before the first epoch boundary")
                    meta = metadata_for(ctx, ctx.train, initialization=initialization, family="BENCHMARK")
                    result = train_baseline(model, loader, config=ctx.train,
                        class_count=len(ctx.class_map.id_to_index), device="cuda", policy=ctx.policy,
                        checkpoint_dir=root, checkpoint_metadata=meta,
                        stop_after_updates=steps, observer=timer)
                    if result.updates != steps:
                        raise ValueError("benchmark did not reach its exact bounded update count")
                else:
                    if len(loader) < steps:
                        raise ValueError("not enough dev batches for evaluation benchmark")
                    model.to("cuda").eval()
                    evaluate_loader(model, loader, total_classes=len(ctx.class_map.id_to_index),
                        max_batches=steps, device="cuda", observer=timer)
            report = {"kind": "benchmark_only", "phase": phase, "precision": "fp32",
                "selection_eligible": False, "configuration": ctx.summary(),
                "source_revision": current_code_revision(), "environment": environment_report(),
                "execution_environment": {name: os.environ.get(name) for name in (
                    "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE", "AIC_ARCHIVE_LOCATIONS")},
                "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                **timer.report(), "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                "prefetch_extra_samples_upper_bound": loader.num_workers * loader.prefetch_factor * loader.batch_size,
                "cache_shard_write_included": False if phase == "cache" else None}
        report["total_seconds_including_cleanup"] = time.monotonic() - started
        write_json(root / "benchmark.json", report)
        return report
    except BaseException as exc:
        write_json(root / "failure.json", {"kind": "benchmark_only", "error": str(exc), "auto_retry": False})
        raise


def main():
    from .pipeline_cli import benchmark_main
    return benchmark_main()


if __name__ == "__main__":
    raise SystemExit(main())
