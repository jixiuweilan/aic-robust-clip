"""Independent 4060 preparation and evidence gate; never imports the T4 runner."""
from __future__ import annotations

import argparse
import copy
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from .configuration import load_config
from .contracts import read_json, sha256_json, write_json
from .environment import environment_report, machine_policy
from .models.provision import file_sha256
from .performance import PerformanceConfig
from .runtime import current_code_revision

PROFILES = {
    "fp32-m16-w4-e0": ("fp32", 16, 4, 0),
    "fp32-m32-w4-e4": ("fp32", 32, 4, 4),
    "fp16-m32-w4-e4": ("fp16", 32, 4, 4),
    "fp16-m64-w4-e4": ("fp16", 64, 4, 4),
    "fp16-m128-w4-e4": ("fp16", 128, 4, 4),
    "fp16-m64-w2-e2": ("fp16", 64, 2, 2),
}


def _new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        import json
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return value


def _binding():
    env = environment_report()
    return {"source_revision": current_code_revision(), "fingerprint": env["fingerprint"],
            "python": env["python"], "dependencies": env["dependencies"],
            "torch_cuda": env.get("torch_cuda"), "gpu": env.get("gpu")}


def _assets(config):
    # Metadata only: no ZIP, weights, checkpoint tensor or feature reads.
    head_path = Path(config["head"]) / "head.json"
    head = read_json(head_path)
    result = head["result"]
    if (head["identity"]["profile"] != "HEAD3" or result["stopped_by_limit"]
            or len(result["epoch_logs"]) != 3 or not all(e["complete"] for e in result["epoch_logs"])
            or re.fullmatch(r"[0-9a-f]{64}", head["sha256"]) is None):
        raise ValueError("reuse a complete, verified HEAD3; never rebuild it here")
    return {**{name: file_sha256(config[name]) for name in ("manifest", "split", "class_map")},
            "head_json": file_sha256(head_path), "head_sha256": head["sha256"],
            "weight_revision": config["weight_revision"], "stage": config["stage"]}


def prepare_bundle(config_path, output, member):
    if member not in {"A", "B"}:
        raise ValueError("member must be A or B")
    source = load_config(config_path)
    p = source.get("parameters", {})
    if (source.get("execution_mode") != "formal" or source.get("device") != "cuda"
            or source["recipe"] != "B03" or source.get("seed", 17) != 17
            or source.get("effective_batch_size", 128) != 128
            or p.get("epochs", 10) != 10 or p.get("formal_epochs", 10) != 10
            or p.get("profile", "ONLINE10") != "ONLINE10" or p.get("objective", "ce") != "ce"
            or p.get("learning_rate", 1e-3) != 1e-3
            or p.get("lora_learning_rate", 1e-4) != 1e-4
            or p.get("scheduler", "warmup_cosine") != "warmup_cosine"
            or p.get("warmup_epochs", 1) != 1 or p.get("weight_decay", 1e-4) != 1e-4
            or any(p.get(key, 0) for key in ("weighting", "lambda_preserve", "prior_tau"))):
        raise ValueError("require seed17 B03 plain CE, original optimizer and full ONLINE10 control")
    PerformanceConfig.from_config(source)
    assets = _assets(source)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    configs = {}
    for name, (precision, micro, workers, eval_workers) in PROFILES.items():
        configs[name] = {}
        for role, lr in (("control", 1e-4), ("candidate", 3e-5 if member == "A" else 3e-4)):
            config = copy.deepcopy(source)
            config.update(seed=17, batch_size=micro, effective_batch_size=128,
                          output=str(root / "runs" / name / role))
            config.setdefault("parameters", {}).update(precision=precision, learning_rate=1e-3,
                lora_learning_rate=lr, formal_epochs=10, profile="ONLINE10", accumulation_steps=128 // micro)
            config["performance"] = {**source.get("performance", {}), "cache_batch_size": 64,
                "head_batch_size": 128, "eval_batch_size": 64, "num_workers": workers,
                "eval_num_workers": eval_workers, "prefetch_factor": 2, "pin_memory": True}
            PerformanceConfig.from_config(config)
            path = root / name / f"{role}.json"
            write_json(path, config)
            configs[name][role] = str(path)
    # This normalized recipe contract is portable across the two hosts.
    recipe = {k: v for k, v in source.items() if k not in {
        "machine_config", "manifest", "split", "class_map", "weights", "head", "output",
        "train_cache", "dev_cache", "performance", "parameters", "batch_size"}}
    recipe["parameters"] = {k: v for k, v in p.items() if k not in {"precision", "accumulation_steps"}}
    return _new_json(root / "bundle.json", {"schema_version": 1, "member": member, "configs": configs,
        "assets": assets, "recipe": recipe, "source_revision": current_code_revision(),
        "config_digests": {name: {role: sha256_json(load_config(p)) for role, p in paths.items()}
                           for name, paths in configs.items()},
        "stop_after_epochs": 3, "status": "prepared_only_no_training", "selection_eligible": False})


def _bundle(path):
    bundle = read_json(path)
    if bundle["source_revision"] != current_code_revision() or bundle["stop_after_epochs"] != 3:
        raise ValueError("bundle is stale; prepare a new directory for this source")
    for name, paths in bundle["configs"].items():
        for role, path in paths.items():
            config = load_config(path)
            if sha256_json(config) != bundle["config_digests"][name][role]:
                raise ValueError("prepared configuration changed; generate a new bundle")
            if _assets(config) != bundle["assets"]:
                raise ValueError("bundle identity metadata changed")
    return bundle


def check(config_path, output):
    """Owner-operated remote acceptance, including the original B04 failed test."""
    config = load_config(config_path)
    binding = _binding()
    if (not machine_policy(config.get("machine_config")).allow_formal
            or "RTX 4060" not in (binding["gpu"] or {}).get("name", "")):
        raise ValueError("check requires an enrolled separate 4060")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    commands = [
        ("relocation", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_relocation.py", "-v"]),
        ("suite", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]),
        ("compile", [sys.executable, "-m", "compileall", "-q", "src", "tests"]),
        ("pip-check", [sys.executable, "-m", "pip", "check"]),
        ("diff-check", ["git", "diff", "--check"]),
    ]
    hashes = {}
    try:
        for name, command in commands:
            result = subprocess.run(command, capture_output=True, text=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
            log = root / f"{name}.log"
            log.write_text(result.stdout + result.stderr, encoding="utf-8")
            if (result.returncode or (name in {"relocation", "suite"} and (
                    not re.search(r"\nOK\s*$", result.stderr) or not re.search(r"Ran [1-9][0-9]* tests?", result.stderr)))):
                raise ValueError(f"{name} failed or skipped tests; retain {log}")
            if name == "relocation" and not re.search(
                    r"test_verification_cached_only_for_unchanged_file_and_process .* \.\.\. ok", result.stderr):
                raise ValueError("original B04 failed regression was not executed successfully")
            hashes[str(log)] = file_sha256(log)
        # Both arithmetic modes get exactly two synthetic updates, no real images.
        from .training.model_smoke import check_official_model
        for precision in ("fp32", "fp16"):
            report = check_official_model(config["weights"], config["weight_revision"], "B03",
                                          device="cuda", precision=precision)
            if (report["result"]["updates"] != 2 or not report["result"]["optimizer_updated"]
                    or not report["result"]["stopped_by_limit"]):
                raise ValueError("synthetic startup did not pass its two-update limit")
            path = root / f"startup-{precision}.json"
            write_json(path, report)
            hashes[str(path)] = file_sha256(path)
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True)
        lock = root / "pip-freeze.txt"
        lock.write_text(freeze.stdout, encoding="utf-8")
        hashes[str(lock)] = file_sha256(lock)
        if binding != _binding():
            raise ValueError("source/environment changed during checks")
        return _new_json(root / "checks.json", {"binding": binding, "status": "passed_zero_skips",
            "original_b04_regression": "test_verification_cached_only_for_unchanged_file_and_process",
            "hashes": hashes})
    except BaseException as exc:
        write_json(root / "failure.json", {"error": str(exc), "auto_retry": False})
        raise


def _hashes_unchanged(hashes):
    for path, digest in hashes.items():
        if file_sha256(path) != digest:
            raise ValueError(f"evidence changed: {path}")


def _report(path, config, binding):
    path = Path(path)
    if (path.parent / "failure.json").exists():
        raise ValueError(f"failed window: {path}")
    r = read_json(path)
    phase = r["phase"]
    batch = 128 if phase == "train" else 64
    workers = config["performance"]["num_workers" if phase == "train" else "eval_num_workers"]
    after, env = r["loader_after_cleanup"], r["environment"]
    expected_binding = {k: env.get(k) for k in binding if k != "source_revision"}
    expected_binding["source_revision"] = r["source_revision"]
    if (expected_binding != binding or r["configuration"]["config"] != config
            or r["kind"] != "benchmark_only" or r["selection_eligible"] is not False
            or phase not in {"train", "eval"} or r["warmup_steps"] != 2 or r["measured_steps"] != 10
            or r["measured_samples"] != batch * 10 or after["delivered_samples"] != batch * 12
            or after["dispatch_stop"] != batch * 12 or after["num_workers"] != workers
            or after["workers"] or after["worker_failed"] or len(after["worker_exits"]) != workers
            or any(w["exitcode"] != 0 or w["alive"] or w["forced"] for w in after["worker_exits"])
            or r["prefetch_extra_samples_upper_bound"] != 0
            or r["precision"] != (config["parameters"]["precision"] if phase == "train" else "fp32")):
        raise ValueError(f"stale/incomplete/unclean benchmark: {path}")
    for key in ("samples_per_second", "checkpoint_write_seconds", "cold_start_and_warmup_seconds", "data_wait_seconds"):
        if not math.isfinite(r[key]) or r[key] < 0 or (key == "samples_per_second" and r[key] == 0):
            raise ValueError(f"invalid timing: {path}")
    if phase == "train" and r["checkpoint_write_seconds"] <= 0:
        raise ValueError("training benchmark must include checkpoint persistence")
    if max(r["peak_gpu_allocated_bytes"], r["peak_gpu_reserved_bytes"]) > .85 * env["gpu"]["memory_bytes"]:
        raise ValueError(f"less than 15% GPU memory headroom: {path}")
    return r


def summarize(bundle_path, directory, checks_path, output):
    bundle, binding = _bundle(bundle_path), _binding()
    checks = read_json(checks_path)
    if checks["binding"] != binding or checks["status"] != "passed_zero_skips":
        raise ValueError("fresh local checks including B04 regression are required")
    _hashes_unchanged(checks["hashes"])
    root = Path(directory).resolve()
    entries, hashes = {}, {str(Path(checks_path).resolve()): file_sha256(checks_path)}
    for name, paths in bundle["configs"].items():
        config = load_config(paths["control"])
        reports = []
        try:
            for phase in ("train", "eval"):
                path = root / name / phase / "benchmark.json"
                r = _report(path, config, binding)
                if r["phase"] != phase:
                    raise ValueError("wrong phase")
                reports.append(r)
                hashes[str(path)] = file_sha256(path)
            train, dev = reports
            counts = train["configuration"]["split_counts"]
            if counts != dev["configuration"]["split_counts"]:
                raise ValueError("split counts differ")
            seconds = (counts["train"] / train["samples_per_second"] + counts["dev"] / dev["samples_per_second"]
                       + 2 * train["checkpoint_write_seconds"])
            entries[name] = {"passed": True, "estimated_epoch_seconds": seconds,
                "cold_seconds": [r["cold_start_and_warmup_seconds"] for r in reports],
                "data_wait_seconds": [r["data_wait_seconds"] for r in reports],
                "peak_reserved_bytes": max(r["peak_gpu_reserved_bytes"] for r in reports)}
        except (OSError, ValueError, KeyError) as exc:
            # Retain original failed reports; missing windows are never passes.
            entries[name] = {"passed": False, "reason": str(exc)}
    for path in sorted(root.rglob("*")):
        if path.is_file() and (path.suffix in {".json", ".log", ".exitcode"}):
            hashes[str(path)] = file_sha256(path)
    return _new_json(output, {"binding": binding, "member": bundle["member"], "assets": bundle["assets"],
        "recipe": bundle["recipe"], "bundle_digest": sha256_json(bundle), "profiles": entries, "hashes": hashes,
        "config_digests": {n: {role: sha256_json(load_config(p)) for role, p in paths.items()}
                           for n, paths in bundle["configs"].items()},
        "estimate_note": "train + dev + two checkpoint writes; excludes cold start; not a time budget"})


def select(summary_a, summary_b, output):
    a, b = read_json(summary_a), read_json(summary_b)
    if ({a["member"], b["member"]} != {"A", "B"} or a["binding"]["fingerprint"] == b["binding"]["fingerprint"]
            or a["assets"] != b["assets"] or a["recipe"] != b["recipe"]
            or a["binding"]["source_revision"] != current_code_revision()
            or b["binding"]["source_revision"] != current_code_revision()
            or a["binding"]["dependencies"] != b["binding"]["dependencies"]):
        raise ValueError("require two distinct matched 4060 hosts, source, dependencies, assets and recipe")
    ranking = []
    for name in PROFILES:
        x, y = a["profiles"][name], b["profiles"][name]
        if x["passed"] and y["passed"]:
            cost = x["estimated_epoch_seconds"] + y["estimated_epoch_seconds"]
            if not math.isfinite(cost) or cost <= 0:
                raise ValueError("invalid estimated time")
            ranking.append((cost, name))
    if not ranking:
        raise ValueError("no common passing profile; stop formal runs and report")
    ranking.sort()
    best = ranking[0][0]
    tied = [(cost, name) for cost, name in ranking if (cost - best) / best < .05]
    _, chosen = min(tied, key=lambda item: (PROFILES[item[1]][2], PROFILES[item[1]][3],
                                           PROFILES[item[1]][1], item[0], item[1]))
    return _new_json(output, {"profile": chosen, "ranking": ranking,
        "summaries": {s["member"]: s for s in (a, b)},
        "summary_hashes": {s["member"]: file_sha256(p) for s, p in ((a, summary_a), (b, summary_b))},
        "status": "engineering_selection_requires_three_local_eval_windows", "selection_eligible": False})


def accept(bundle_path, selection_path, eval_directory, checks_path, output):
    bundle, selection, binding = _bundle(bundle_path), read_json(selection_path), _binding()
    summary = selection["summaries"][bundle["member"]]
    checks = read_json(checks_path)
    if (summary["binding"] != binding or checks["binding"] != binding
            or checks["status"] != "passed_zero_skips" or summary["bundle_digest"] != sha256_json(bundle)):
        raise ValueError("stale or foreign checks/selection/bundle")
    _hashes_unchanged(checks["hashes"])
    _hashes_unchanged(summary["hashes"])
    name = selection["profile"]
    digests = {role: sha256_json(load_config(p)) for role, p in bundle["configs"][name].items()}
    if digests != summary["config_digests"][name]:
        raise ValueError("configuration changed after measurements")
    hashes = {**checks["hashes"], **summary["hashes"],
              str(Path(selection_path).resolve()): file_sha256(selection_path)}
    config = load_config(bundle["configs"][name]["control"])
    for repeat in range(1, 4):
        path = Path(eval_directory).resolve() / f"eval-{repeat}" / "benchmark.json"
        report = _report(path, config, binding)
        if report["phase"] != "eval":
            raise ValueError("acceptance requires three independent eval windows")
        hashes[str(path)] = file_sha256(path)
    return _new_json(output, {"binding": binding, "bundle_digest": sha256_json(bundle), "profile": name,
        "config_digests": digests, "hashes": hashes, "status": "accepted_engineering_only",
        "selection_eligible": False, "stop_after_epochs": 3})


def run(bundle_path, receipt_path, role, *, resume=False):
    """One explicit job, pauses after epoch 3; no promotion or continuation loop."""
    bundle, receipt = _bundle(bundle_path), read_json(receipt_path)
    if (receipt["status"] != "accepted_engineering_only" or receipt["binding"] != _binding()
            or receipt["bundle_digest"] != sha256_json(bundle) or receipt["stop_after_epochs"] != 3):
        raise ValueError("new source/dependencies/host require new acceptance")
    _hashes_unchanged(receipt["hashes"])
    paths = bundle["configs"][receipt["profile"]]
    if {r: sha256_json(load_config(p)) for r, p in paths.items()} != receipt["config_digests"]:
        raise ValueError("accepted configurations changed")
    config = load_config(paths[role])
    if role == "candidate":
        control_root = Path(load_config(paths["control"])["output"])
        if (control_root / "failure.json").exists():
            raise ValueError("control has failure evidence; stop and report")
        control = read_json(control_root / "result.json")
        logs = control["epoch_logs"]
        if (len(logs) != 3 or not all(e["complete"] for e in logs) or not control["paused"]
                or control["completed_epochs"] != 3 or control["stopped_by_limit"]):
            raise ValueError("complete the three-epoch local control first")
    from .workflow import train_command
    return train_command(paths[role], resume=Path(config["output"]) / "last.pt" if resume else None,
                         stop_after_epochs=3)


def main(argv=None):
    parser = argparse.ArgumentParser(description="4060 独立任务包：准备、验收、三轮暂停；不下载，不操作 T4")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", required=True); p.add_argument("--output", required=True)
    p.add_argument("--member", choices=("A", "B"), required=True)
    p = sub.add_parser("check")
    p.add_argument("--config", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("summarize")
    for name in ("bundle", "directory", "checks", "output"):
        p.add_argument("--" + name, required=True)
    p = sub.add_parser("select")
    for name in ("summary-a", "summary-b", "output"):
        p.add_argument("--" + name, required=True)
    p = sub.add_parser("accept")
    for name in ("bundle", "selection", "eval-directory", "checks", "output"):
        p.add_argument("--" + name, required=True)
    p = sub.add_parser("run")
    p.add_argument("--bundle", required=True); p.add_argument("--receipt", required=True)
    p.add_argument("--role", choices=("control", "candidate"), required=True)
    p.add_argument("--resume", action="store_true")
    a = parser.parse_args(argv)
    from .pipeline_cli import _execute
    actions = {
        "prepare": lambda: prepare_bundle(a.config, a.output, a.member),
        "check": lambda: check(a.config, a.output),
        "summarize": lambda: summarize(a.bundle, a.directory, a.checks, a.output),
        "select": lambda: select(a.summary_a, a.summary_b, a.output),
        "accept": lambda: accept(a.bundle, a.selection, a.eval_directory, a.checks, a.output),
        "run": lambda: run(a.bundle, a.receipt, a.role, resume=a.resume),
    }
    return _execute(actions[a.command])


if __name__ == "__main__":
    raise SystemExit(main())
