"""Four independent T4 jobs, explicit performance gate, no downloads or auto-promotion.

Run with ``python -m aic_robust_clip.t4_runner --help``. Preparation is metadata
only; benchmark and training require the existing remote host enrollment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import copy
import fcntl
import math
from importlib.metadata import version
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import threading

from .configuration import load_config, prepare
from .contracts import read_json, sha256_json, write_json
from .environment import machine_fingerprint, machine_policy
from .models.provision import file_sha256
from .performance import PerformanceConfig
from .runtime import current_code_revision


JOBS = {"N00": "B03", "N01": "R01", "N02": "F100", "N03": "F010"}
# Each profile is applied to ALL methods, never just the baseline. eval and W
# scoring remain FP32; fp16 changes training arithmetic and is opt-in.
PROFILES = {
    "fp32-m16-e0": (16, 4, 0, "fp32"),
    "fp32-m16-e4": (16, 4, 4, "fp32"),
    "fp32-m32-e4": (32, 4, 4, "fp32"),
    "fp32-m32-w2-e2": (32, 2, 2, "fp32"),
    "fp16-m32-e4": (32, 4, 4, "fp16"),
    "fp16-m64-e4": (64, 4, 4, "fp16"),
}


def dependencies():
    return {name: version(name) for name in ("torch", "transformers", "Pillow", "numpy", "huggingface-hub")}


def lineage_report(ctx):
    """Export metadata only, with groups defined exactly as training metrics."""
    from .metrics import class_frequency_groups
    counts = Counter(ctx.class_map.id_to_index[item.class_id] for item in ctx.split.records if item.partition == "train")
    return {"stage": ctx.run.stage, "manifest_digest": ctx.run.manifest_digest,
        "split_digest": ctx.run.split_digest, "class_map_digest": ctx.run.class_map_digest,
        "class_map": dict(ctx.class_map.id_to_index), "train_counts": dict(counts),
        "frequency_groups": {name: sorted(ids) for name, ids in class_frequency_groups(
            counts, len(ctx.class_map.id_to_index)).items()},
        "dev_groups": {item.sample_id: item.group_id for item in ctx.split.records if item.partition == "dev"},
        "images_read": 0, "confirm_access": "metadata_only_no_scoring"}


def prepare_bundle(config_path, output):
    source = load_config(config_path)
    PerformanceConfig.from_config(source)
    p = source.get("parameters", {})
    if (source.get("execution_mode") != "formal" or source.get("device") != "cuda"
            or source["recipe"] not in {"B03", "B04"} or source.get("effective_batch_size", 128) != 128
            or p.get("objective", "ce") != "ce"
            or any(p.get(k, 0) for k in ("weighting", "lambda_preserve", "prior_tau"))):
        raise ValueError("bundle needs the operator's formal CUDA plain-CE B03/B04 config, effective batch 128")
    if p.get("epochs", p.get("formal_epochs", 10)) != 10:
        raise ValueError("bundle preserves the full ten-epoch schedule")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    configs = {}
    for name, (micro, workers, eval_workers, precision) in PROFILES.items():
        configs[name] = {}
        for job, recipe in JOBS.items():
            config = copy.deepcopy(source)
            config.update(recipe=recipe, batch_size=micro, effective_batch_size=128, seed=17,
                          output=str(root / "runs" / name / job))
            params = config.setdefault("parameters", {})
            for key in ("accumulation_steps", "objective", "gce_q", "weighting", "lambda_preserve", "prior_tau", "epochs"):
                params.pop(key, None)
            params.update(formal_epochs=10, profile="ONLINE10", scheduler="warmup_cosine", warmup_epochs=1,
                          learning_rate=1e-3, lora_learning_rate=1e-4, weight_decay=1e-4)
            if precision == "fp16":
                params["precision"] = precision
            else:
                params.pop("precision", None)
            config["performance"] = {"cache_batch_size": 64, "head_batch_size": 128,
                "eval_batch_size": 64, "scoring_batch_size": 128, "num_workers": workers,
                "eval_num_workers": eval_workers, "scoring_num_workers": eval_workers,
                "prefetch_factor": 2, "pin_memory": True}
            PerformanceConfig.from_config(config)
            path = root / "configs" / name / f"{job}.json"
            write_json(path, config)
            configs[name][job] = str(path)
    value = {"schema_version": 1, "source": str(Path(config_path).resolve()), "configs": configs,
             "status": "unmeasured_not_authorized_for_training", "source_revision": current_code_revision()}
    write_json(root / "bundle.json", value)
    return {"bundle": str(root / "bundle.json"), "profiles": list(configs), "jobs_started": 0}


def gpu_inventory(indices, count=4):
    if len(indices) != count or len(set(indices)) != count or any(not x.isdigit() for x in indices):
        raise ValueError("one distinct physical GPU index per selected job is required")
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=15)
    devices = {}
    for line in result.stdout.splitlines():
        index, uuid, name = [part.strip() for part in line.split(",", 2)]
        devices[index] = {"uuid": uuid, "name": name}
    chosen = [devices[index] for index in indices]
    if any(device["name"] not in {"Tesla T4", "NVIDIA Tesla T4", "NVIDIA T4"} for device in chosen):
        raise ValueError("this runner is restricted to the four T4s; it does not enroll hosts")
    busy = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                          check=True, capture_output=True, text=True, timeout=15).stdout
    if any(device["uuid"] in busy for device in chosen):
        raise ValueError("a selected GPU has an existing compute process; do not interrupt it")
    return chosen


def profile_configs(bundle, profile):
    paths = read_json(bundle)["configs"][profile]
    if set(paths) != set(JOBS):
        raise ValueError("bundle must contain N00/N01/N02/N03 exactly")
    values = {job: load_config(paths[job]) for job in JOBS}
    for job, config in values.items():
        if config["recipe"] != JOBS[job] or config.get("seed") != 17:
            raise ValueError("bundle recipe/seed mismatch")
        # Local development host stays prohibited even with fabricated files.
        if not machine_policy(config.get("machine_config")).allow_formal:
            raise ValueError("remote training enrollment is required")
    return paths, values


def benchmark_worker(config, output):
    from .benchmark import benchmark_command
    value = load_config(config)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    phases = [("train", 1), ("eval", 3)]
    if value["recipe"] == "F100":
        phases.append(("scoring", 3))
    reports = []
    for phase, repeats in phases:
        for repeat in range(repeats):
            target = root / f"{phase}-{repeat}"
            benchmark_command(config, phase, target, warmup_steps=2, measure_steps=10)
            reports.append(str(target / "benchmark.json"))
    return {"reports": reports, "formal_training": False}


def benchmark_summary(directory, configs):
    """Validate all reports, including repeated cleanup, and estimate whole epochs."""
    root = Path(directory).resolve()
    receipt = read_json(root / "dispatch.json")
    if receipt.get("mode") != "bench" or receipt.get("status") != "complete" or not receipt.get("concurrent"):
        raise ValueError("acceptance requires a completed four-way concurrent benchmark")
    estimates, hashes = {}, {}
    for job, config in configs.items():
        phases = {"train": 1, "eval": 3, **({"scoring": 3} if config["recipe"] == "F100" else {})}
        rates, checkpoint = {}, 0.
        for phase, repeats in phases.items():
            phase_rates = []
            for repeat in range(repeats):
                path = root / job / f"{phase}-{repeat}" / "benchmark.json"
                report = read_json(path)
                expected_batch = 128 if phase == "train" else config["performance"][
                    "scoring_batch_size" if phase == "scoring" else "eval_batch_size"]
                after = report["loader_after_cleanup"]
                env = report["environment"]
                if (report.get("kind") != "benchmark_only" or report.get("selection_eligible") is not False
                        or report["configuration"]["config"] != config or report["phase"] != phase
                        or report["source_revision"] != current_code_revision()
                        or env["fingerprint"] != machine_fingerprint()
                        or env["dependencies"] != dependencies()
                        or report["measured_steps"] != 10 or report["warmup_steps"] != 2
                        or report["measured_samples"] != 10 * expected_batch
                        or after["delivered_samples"] != 12 * expected_batch
                        or report["prefetch_extra_samples_upper_bound"] != 0
                        or after["worker_failed"] or after["workers"]
                        or any(w["exitcode"] != 0 or w["alive"] or w.get("forced") for w in after["worker_exits"])):
                    raise ValueError(f"invalid/stale/incomplete benchmark: {path}")
                speed = report["samples_per_second"]
                if not math.isfinite(speed) or speed <= 0:
                    raise ValueError("invalid measured throughput")
                if report["peak_gpu_reserved_bytes"] > .9 * env["gpu"]["memory_bytes"]:
                    raise ValueError("less than 10% reserved-memory headroom")
                phase_rates.append(speed)
                checkpoint = max(checkpoint, report["checkpoint_write_seconds"])
                hashes[str(path)] = file_sha256(path)
                counts = report["configuration"]["split_counts"]
            rates[phase] = min(phase_rates)  # conservative, not best of repeats
        seconds = counts["train"] / rates["train"] + counts["dev"] / rates["eval"] + 2 * checkpoint
        if "scoring" in rates:
            seconds += counts["train"] / rates["scoring"]
        estimates[job] = {"rates": rates, "estimated_epoch_seconds": seconds,
                          "estimated_three_epoch_seconds": 3 * seconds}
    return {"estimates": estimates, "report_hashes": hashes}


def accept(bundle, profile, directory, comparison, output, rationale):
    if not rationale.strip():
        raise ValueError("record why this measured profile was selected")
    if Path(output).exists():
        raise FileExistsError(output)
    _, configs = profile_configs(bundle, profile)
    selected = benchmark_summary(directory, configs)
    comparisons = []
    # At least one measured alternative is mandatory. The operator additionally
    # checks AMP and higher batch candidates as specified in the handoff.
    for other in comparison:
        dispatch = read_json(Path(other) / "dispatch.json")
        _, other_configs = profile_configs(bundle, dispatch["profile"])
        comparisons.append({"profile": dispatch["profile"], **benchmark_summary(other, other_configs)})
    if not comparisons or not any(item["profile"] != profile for item in comparisons):
        raise ValueError("at least one different measured engineering profile is required")
    selected_total = sum(x["estimated_epoch_seconds"] for x in selected["estimates"].values())
    best_total = min(sum(x["estimated_epoch_seconds"] for x in item["estimates"].values()) for item in comparisons)
    if selected_total > best_total * 1.10:
        raise ValueError("selected profile is >10% slower in aggregate estimated epoch cost; investigate before training")
    value = {"schema_version": 1, "status": "performance_accepted_not_model_selected", "profile": profile,
        "fingerprint": machine_fingerprint(), "source_revision": current_code_revision(),
        "dependencies": dependencies(),
        "config_digests": {job: sha256_json(config) for job, config in configs.items()},
        "rationale": rationale, **selected, "comparisons": comparisons,
        "note": "Short-window estimates exclude cold start; epoch timings remain the production check. No convergence claim."}
    write_json(output, value)
    return value


def validate_acceptance(path, profile, configs):
    receipt = read_json(path)
    if (receipt.get("status") != "performance_accepted_not_model_selected" or receipt["profile"] != profile
            or receipt["fingerprint"] != machine_fingerprint() or receipt["source_revision"] != current_code_revision()
            or receipt["dependencies"] != dependencies()
            or receipt["config_digests"] != {job: sha256_json(config) for job, config in configs.items()}):
        raise ValueError("acceptance receipt does not match this source/host/configuration")
    for report, digest in receipt["report_hashes"].items():
        if file_sha256(report) != digest:
            raise ValueError("benchmark evidence changed after acceptance")


def dispatch(bundle, profile, output, indices, *, mode, through=3, resume=False, acceptance=None, jobs=None):
    if mode not in {"bench", "train"} or through not in {3, 6, 10}:
        raise ValueError("explicit bench/train and epoch boundary 3/6/10 required")
    paths, configs = profile_configs(bundle, profile)
    jobs = list(JOBS) if jobs is None else jobs
    if not jobs or len(set(jobs)) != len(jobs) or set(jobs) - set(JOBS):
        raise ValueError("select unique known job IDs")
    if mode == "bench" and jobs != list(JOBS):
        raise ValueError("acceptance benchmark must cover all four jobs concurrently")
    if mode == "train":
        if not acceptance:
            raise ValueError("formal dispatch requires a matching performance acceptance receipt")
        validate_acceptance(acceptance, profile, configs)
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError(root)
    devices = gpu_inventory(indices, len(jobs))
    # Check all jobs before starting any of them. prepare reads metadata, never
    # trains, enrolls, creates caches or downloads missing weights.
    shared = lineage = None
    for job in jobs:
        ctx = prepare(paths[job])
        if not ctx.policy.allow_formal or ctx.run.execution_mode != "formal" or ctx.config["device"] != "cuda":
            raise ValueError("all jobs must be enrolled formal CUDA configurations")
        from .workflow import load_head
        _, head_digest = load_head(ctx)
        identity = (ctx.run.stage, ctx.run.manifest_digest, ctx.run.split_digest, ctx.run.class_map_digest,
                    ctx.weights["digest"], ctx.preprocessing_digest, head_digest, ctx.train.precision,
                    ctx.run.batch_size, ctx.train.accumulation_steps, ctx.train.epochs)
        if shared is not None and shared != identity:
            raise ValueError("paired jobs differ in data/head/precision/batch/schedule identity")
        shared = identity
        if lineage is None:
            lineage = lineage_report(ctx)
        if mode == "train":
            run = Path(configs[job]["output"])
            if resume:
                previous = read_json(run / "result.json")
                if previous.get("status") != "paused" or previous.get("completed_epochs", 0) >= through:
                    raise ValueError("resume requires a paused run before the requested boundary")
                if not (run / "last.pt").is_file():
                    raise ValueError("missing resumable last.pt")
            elif run.exists():
                raise FileExistsError(run)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "data-lineage.json", lineage)
    started = time.monotonic()
    report = {"mode": mode, "profile": profile, "status": "running", "concurrent": True,
              "devices": devices, "through": through if mode == "train" else None, "jobs": {}}
    write_json(root / "dispatch.json", report)
    children = []
    cancelled = threading.Event()
    # Locks are shared by every bundle of this checkout, not per output folder.
    lock_root = Path(__file__).resolve().parents[2] / "outputs" / "t4-gpu-locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for device in devices:
            handle = stack.enter_context((lock_root / f"{device['uuid']}.lock").open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def run_job(pair):
            job, device = pair
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device["uuid"], HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       OPENBLAS_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
            if mode == "bench":
                command = [sys.executable, "-m", "aic_robust_clip.t4_runner", "_bench-worker",
                           "--config", paths[job], "--output", str(root / job)]
            else:
                command = [sys.executable, "-m", "aic_robust_clip.t4_runner", "_train-worker",
                           "--config", paths[job], "--through", str(through)]
                if resume:
                    command += ["--resume"]
            with (root / f"{job}.log").open("x") as log:
                if cancelled.is_set():
                    raise RuntimeError("dispatch cancelled before launch")
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                children.append(child)
                while True:
                    if cancelled.is_set() and child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                    try:
                        code = child.wait(timeout=1)
                        return job, {"exit_code": code, "command": command, "gpu_uuid": device["uuid"]}
                    except subprocess.TimeoutExpired:
                        continue
        executor = ThreadPoolExecutor(max_workers=4)
        try:
            futures = [executor.submit(run_job, pair) for pair in zip(jobs, devices)]
            for future in futures:
                job, result = future.result()
                report["jobs"][job] = result
                write_json(root / "dispatch.json", report)
            report["status"] = "complete" if all(r["exit_code"] == 0 for r in report["jobs"].values()) else "failed"
        except BaseException:
            report["status"] = "interrupted_or_failed"
            cancelled.set()
            raise
        finally:
            executor.shutdown(wait=True)
            report["wall_seconds"] = time.monotonic() - started
            write_json(root / "dispatch.json", report)
    if report["status"] != "complete":
        raise RuntimeError("at least one job failed; see logs; no automatic retry or promotion")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="四张 T4 独立任务；无下载、无自动晋级")
    subs = parser.add_subparsers(dest="action", required=True)
    p = subs.add_parser("prepare", help="只生成配置，不读取图像")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    for action in ("bench", "train", "accept"):
        p = subs.add_parser(action)
        p.add_argument("--bundle", required=True)
        p.add_argument("--profile", choices=tuple(PROFILES), required=True)
        p.add_argument("--output", required=True)
        if action in {"bench", "train"}:
            p.add_argument("--gpus", default="0,1,2,3")
        if action == "train":
            p.add_argument("--through", type=int, choices=(3, 6, 10), default=3)
            p.add_argument("--resume", action="store_true")
            p.add_argument("--acceptance", required=True)
            p.add_argument("--jobs", default=",".join(JOBS), help="selected survivors; pair with the same number of --gpus")
        if action == "accept":
            p.add_argument("--bench-dir", required=True)
            p.add_argument("--comparison", action="append", required=True)
            p.add_argument("--rationale", required=True)
    for action in ("_bench-worker", "_train-worker"):
        p = subs.add_parser(action)
        p.add_argument("--config", required=True)
        if action == "_bench-worker":
            p.add_argument("--output", required=True)
        else:
            p.add_argument("--through", type=int, required=True)
            p.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    from .pipeline_cli import _execute
    def action():
        if args.action == "prepare":
            return prepare_bundle(args.config, args.output)
        if args.action == "_bench-worker":
            return benchmark_worker(args.config, args.output)
        if args.action == "_train-worker":
            from .workflow import train_command
            value = load_config(args.config)
            result = train_command(args.config, stop_after_epochs=args.through,
                                   resume=Path(value["output"]) / "last.pt" if args.resume else None)
            # Don't dump every microbatch loss into a many-MB stdout log.
            return {key: value for key, value in result.items() if key != "losses"}
        if args.action == "accept":
            return accept(args.bundle, args.profile, args.bench_dir, args.comparison, args.output, args.rationale)
        return dispatch(args.bundle, args.profile, args.output, args.gpus.split(","), mode=args.action,
                        through=getattr(args, "through", 3), resume=getattr(args, "resume", False),
                        acceptance=getattr(args, "acceptance", None),
                        jobs=args.jobs.split(",") if hasattr(args, "jobs") else None)
    return _execute(action)


if __name__ == "__main__":
    raise SystemExit(main())
