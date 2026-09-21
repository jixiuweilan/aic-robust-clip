"""Isolated preliminary LoRA comparisons and admission-only 4060 preparation.

No downloads, no round2 assets, no implicit training from metadata preparation.
The legacy one-command TURN pilot remains a separate entry point.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch

from . import preliminary_pilot as pilot
from .contracts import read_json, write_json, sha256_json
from .gpu_identity import normalize_gpu_uuid, cuda_gpu_uuids
from .models.provision import file_sha256
from .round2 import admission, engine
from .round2.config import RECIPE, sealed, verify_seal, stage_path
from .round2.methods import MethodError

VERSION = "preliminary-comparison-v1"
METHODS = ("ce", "turn", "fine", "snscl")
VARIANTS = {"ce": "CE-B32-v1", "turn": pilot.VARIANT, "fine": "FINE-B32-v1", "snscl": "SNSCL-B32-v1"}
FIXED = {"microbatch": 32, "workers": 2}


def policy(method):
    if method not in METHODS:
        raise ValueError("unknown preliminary method")
    return "retain_observed" if method == "turn" else "error"


def recipe(method):
    return {**copy.deepcopy(RECIPE), "stage": "preliminary", "initializer": "verified_preliminary_HEAD3",
            "method_variant": VARIANTS[method], "zero_selection_policy": policy(method)}


def plan(output):
    """Metadata only: never prepare assets, fit HEAD, profile or download."""
    root = stage_path(output, stage="preliminary")
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for method in METHODS:
        value = sealed({"version": VERSION, "stage": "preliminary", "method": method, "adaptation": "lora",
                        "recipe": recipe(method), "engineering": FIXED,
                        "status": "blocked_on_machine_admission", "purpose": pilot.PURPOSE,
                        "round2_eligible": False})
        write_json(root / f"{method}.json", value)
        rows.append(value)
    return rows


def source_context(source_config, *, family):
    source = pilot.load_config(source_config)
    if source.get("stage") != "preliminary":
        raise ValueError("only preliminary source configurations are accepted")
    engine.require_machine(source.get("machine_config"), stage="preliminary")
    runtime = admission.runtime_identity()
    if torch.cuda.device_count() != 1 or family not in runtime.get("gpu", {}).get("name", ""):
        raise ValueError(f"exactly one visible {family} GPU required")
    ctx = pilot.prepare(source_config)
    pilot.assert_source(ctx)
    _, head = pilot.load_head(ctx)
    assets = pilot.PilotAssets(ctx, head)
    return source, assets, runtime


def check_profile(value, *, method, assets, runtime, engineering, windows):
    verify_seal(value)
    if (value.get("kind") != "profile" or value.get("status") != "passed"
            or value.get("stage") != "preliminary" or value.get("runtime") != runtime
            or value.get("method") != method or value.get("adaptation") != "lora"
            or value.get("zero_selection_policy") != policy(method)
            or value.get("asset_digest") != assets.descriptor["digest"] or value.get("head_sha256") != assets.head_sha256
            or any(value.get(k) != v for k, v in engineering.items())
            or len(value.get("eval_seconds", [])) != windows
            or value.get("warmup") != 2 or value.get("measured") != 10
            or any(not math.isfinite(t) or t <= 0 for t in value.get("eval_seconds", []))
            or not value.get("workers_closed") or not value.get("dev_student_replay")
            or not 0 < value.get("peak_occupied_bytes", 0) <= .85 * value.get("total_bytes", 0)):
        raise ValueError("preliminary profile identity/evidence mismatch")


def run_profile(source_config, output, *, method, family="T4", engineering=None, barrier=None, checks=True):
    settings = engineering or FIXED
    source, assets, runtime = source_context(source_config, family=family)
    root = stage_path(output, stage="preliminary")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "provenance.json", sealed({"version": VERSION, "method": method,
        "source_configuration": str(Path(source_config).resolve()), "source_sha256": file_sha256(source_config),
        "assets": assets.descriptor, "runtime": runtime, "purpose": pilot.PURPOSE}))
    try:
        if checks:
            pilot.software_checks(root, method=method)
        value = admission._profile_one(None, method=method, adaptation="lora", **settings,
            machine=source.get("machine_config"), output=root / "profile", eval_windows=3,
            barrier=barrier, _assets=assets, _student_factory=pilot.student_factory,
            _stage="preliminary", _zero_selection_policy=policy(method))
        check_profile(value, method=method, assets=assets, runtime=runtime, engineering=settings, windows=3)
        return value
    except BaseException as exc:
        engine.record_failure(root, exc)
        raise


def evidence(path):
    return {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}


def checked_evidence(entry):
    if file_sha256(entry["path"]) != entry["sha256"]:
        raise ValueError("admission evidence changed")
    value = read_json(entry["path"])
    verify_seal(value)
    return value


def check_software(value, method, runtime):
    verify_seal(value)
    suite = value.get("suite", {})
    expected = {(method, precision, policy(method)) for precision in ("fp32", "fp16")}
    actual = {(r.get("method"), r.get("precision"), r.get("zero_selection_policy")) for r in value.get("startup", [])}
    if (value.get("runtime") != runtime or suite.get("tests", 0) < 1
            or any(suite.get(k) != 0 for k in ("failures", "errors", "skips"))
            or actual != expected or len(value.get("startup", [])) != 2
            or any(r.get("status") != "passed" or r.get("updates") != 2 or r.get("adaptation") != "lora"
                   or r.get("device") != "cuda" for r in value.get("startup", []))):
        raise ValueError("fresh matching CUDA/software checks required")


def admit_t4(source_config, output, *, methods, gpu_uuids, owner):
    """Concurrent profiles only. Never launches training; failed group stops."""
    gpu_uuids = cuda_gpu_uuids(gpu_uuids)
    if (not owner.strip() or not methods or len(set(methods)) != len(methods)
            or any(m not in METHODS for m in methods) or len(gpu_uuids) != len(methods)
            or len(set(gpu_uuids)) != len(gpu_uuids) or any(not u.startswith("GPU-") for u in gpu_uuids)):
        raise ValueError("explicit owner and distinct method/GPU assignments required")
    source = pilot.load_config(source_config)
    if source.get("stage") != "preliminary":
        raise ValueError("only preliminary source configurations are accepted")
    engine.require_machine(source.get("machine_config"), stage="preliminary")
    root = stage_path(output, stage="preliminary")
    root.mkdir(parents=True, exist_ok=False)
    barrier = root / "barrier"
    barrier.mkdir()
    processes, handles = [], []
    try:
        for method, uuid in zip(methods, gpu_uuids):
            handle = (root / f"{method}.log").open("w")
            handles.append(handle)
            command = [sys.executable, "-m", "aic_robust_clip.preliminary_experiments", "profile",
                       "--source-config", str(Path(source_config).resolve()), "--output", str(root / method),
                       "--method", method, "--barrier", str(barrier)]
            processes.append(subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": uuid},
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True))
        while not all((barrier / f"{m}.ready.json").is_file() for m in methods):
            if any(p.poll() is not None for p in processes):
                raise ValueError("a profile exited before shared start; preserve all failures")
            time.sleep(.1)
        write_json(barrier / "go.json", {"released": time.time()})
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise ValueError("concurrent profile failed; no automatic retry")
            time.sleep(.1)
        if any(p.returncode != 0 for p in processes):
            raise ValueError("concurrent profile failed")
        rows, entries = [], {}
        for method, uuid in zip(methods, gpu_uuids):
            child = root / method
            row = read_json(child / "profile/profile.json")
            info = read_json(child / "provenance.json")
            verify_seal(info)
            verify_seal(row)
            checks_value = read_json(child / "checks.json")
            check_software(checks_value, method, row["runtime"])
            if (normalize_gpu_uuid(row["runtime"]["gpu"]["uuid"]) != normalize_gpu_uuid(uuid) or row["method"] != method
                    or info["runtime"] != row["runtime"] or info["assets"]["digest"] != row["asset_digest"]
                    or info["source_sha256"] != file_sha256(source_config)):
                raise ValueError("concurrent profile source/GPU identity mismatch")
            rows.append(row)
            entries[method] = {"profile": evidence(child / "profile/profile.json"),
                               "checks": evidence(child / "checks.json"), "provenance": evidence(child / "provenance.json")}
        overlap = min(r["train_window_ended"] for r in rows) - max(r["train_window_started"] for r in rows)
        hosts = {sha256_json({k: v for k, v in r["runtime"].items() if k != "gpu"}) for r in rows}
        if overlap <= 0 or len(hosts) != 1 or len({r["asset_digest"] for r in rows}) != 1 or len({r["head_sha256"] for r in rows}) != 1:
            raise ValueError("no shared training window or shared initialization/assets")
        value = sealed({"version": VERSION, "kind": "t4_admission", "status": "passed", "stage": "preliminary",
            "owner": owner, "methods": list(methods), "gpu_uuids": list(gpu_uuids), "engineering": FIXED,
            "source_configuration": str(Path(source_config).resolve()), "source_sha256": file_sha256(source_config),
            "entries": entries, "overlap_seconds": overlap, "formal_training_started": False})
        write_json(root / "admission.json", value)
        return value
    except BaseException as exc:
        engine.record_failure(root, exc)
        raise
    finally:
        for p in processes:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
        for h in handles:
            h.close()


def comparison_identity(assets, engineering):
    common = recipe("ce")
    for key in ("method_variant", "zero_selection_policy"):
        del common[key]
    return sha256_json({"asset_digest": assets.descriptor["digest"], "head_sha256": assets.head_sha256,
                        "recipe": common, "engineering": engineering, "adaptation": "lora"})


def compare(control_root, candidate_root):
    """Only finished pairs with identical comparison conditions are matched."""
    configs, results = [], []
    for root in (Path(control_root), Path(candidate_root)):
        cfg = read_json(root / "config.json")
        verify_seal(cfg)
        result = read_json(root / "run/result.json")
        replay = read_json(root / "replay-dev/replay.json")
        if (cfg.get("version") != VERSION or cfg.get("stage") != "preliminary"
                or cfg.get("recipe") != recipe(cfg["method"])
                or result.get("status") != "paused_at_epoch10" or result.get("completed_epochs") != 10
                or result.get("full_epochs") != 30 or not replay.get("passed")
                or replay.get("partition") != "dev" or replay.get("epoch") != 10
                or replay.get("identity", {}).get("checkpoint_sha256") != result.get("checkpoint_sha256")
                or replay.get("student_sha256") != file_sha256(root / "replay-dev/student.pt")
                or result.get("checkpoint_sha256") != file_sha256(root / "run/last.pt")
                or result.get("best_sha256") != file_sha256(root / "run/best.pt")):
            raise ValueError("complete epoch10 checkpoints/dev replay required")
        configs.append(cfg)
        results.append(result)
    if (configs[0]["method"] != "ce" or configs[1]["method"] == "ce"
            or configs[0].get("comparison_digest") != configs[1].get("comparison_digest")
            or any(configs[0].get(k) != configs[1].get(k) for k in ("asset_digest", "head_sha256", "engineering", "adaptation"))
            or configs[0]["runtime"]["host"] != configs[1]["runtime"]["host"]
            or configs[0]["runtime"]["source"] != configs[1]["runtime"]["source"]):
        raise ValueError("not a same-host, same-source matched CE/candidate comparison")
    return {"stage": "preliminary", "candidate": configs[1]["method"], "matched": True,
            "best_macro_delta": results[1]["best_metrics"]["macro_recall"] - results[0]["best_metrics"]["macro_recall"],
            "last_macro_delta": results[1]["last"]["macro_recall"] - results[0]["last"]["macro_recall"],
            "comparison_digest": configs[0]["comparison_digest"], "single_seed_only": True}


def run(receipt_path, output, *, method, resume=False):
    receipt = read_json(receipt_path)
    verify_seal(receipt)
    if (receipt.get("version") != VERSION or receipt.get("kind") != "t4_admission"
            or receipt.get("status") != "passed" or receipt.get("stage") != "preliminary"
            or method not in receipt.get("methods", []) or receipt.get("engineering") != FIXED):
        raise ValueError("new matching T4 admission required; 4060 admission cannot launch training")
    source_config = receipt["source_configuration"]
    if file_sha256(source_config) != receipt["source_sha256"]:
        raise ValueError("source configuration changed")
    _, assets, runtime = source_context(source_config, family="T4")
    entry = receipt["entries"][method]
    profile = checked_evidence(entry["profile"])
    checks = checked_evidence(entry["checks"])
    info = checked_evidence(entry["provenance"])
    check_software(checks, method, runtime)
    check_profile(profile, method=method, assets=assets, runtime=runtime, engineering=FIXED, windows=3)
    if info["assets"] != assets.descriptor or info["runtime"] != runtime:
        raise ValueError("admitted assets/runtime changed")
    root = stage_path(output, stage="preliminary")
    config = sealed({"version": VERSION, "stage": "preliminary", "purpose": pilot.PURPOSE,
        "method": method, "adaptation": "lora", "group": "preliminary-t4", "recipe": recipe(method),
        "engineering": FIXED, "output": str(root / "run"), "asset_digest": assets.descriptor["digest"],
        "head_sha256": assets.head_sha256, "runtime": runtime, "receipt_digest": receipt["digest"],
        "comparison_digest": comparison_identity(assets, FIXED), "round2_eligible": False})
    if resume:
        if read_json(root / "config.json") != config:
            raise ValueError("resume configuration/receipt identity changed")
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "config.json", config)
    try:
        result = engine._run_prepared(config, assets, resume=resume, student_factory=pilot.student_factory,
                                      expected_stage="preliminary", purpose=pilot.PURPOSE)
        replay = engine._replay_prepared(config, assets, output=root / "replay-dev", student_factory=pilot.student_factory,
                                         expected_stage="preliminary", purpose=pilot.PURPOSE)
        summary = {"version": VERSION, "stage": "preliminary", "method": method, "method_variant": VARIANTS[method],
                   "comparison_digest": config["comparison_digest"], "result": result, "replay": replay,
                   "round2_eligible": False, "source": runtime["source"]}
        write_json(root / "summary.json", summary)
        (root / "RESULT.zh-CN.md").write_text(
            f"# 初赛 {VARIANTS[method]}\n\n30轮调度，第10轮评分、验证、保存后暂停。\n\n"
            f"best macro recall：{result['best_metrics']['macro_recall']:.6%}；"
            f"last：{result['last']['macro_recall']:.6%}。\n\ndev回放通过：{replay['passed']}。"
            "逐类/逐图、筛选覆盖、软标签/队列和开销见run目录；不自动续跑。\n", encoding="utf-8")
        return summary
    except BaseException as exc:
        engine.record_failure(root, exc)
        raise


def check_history(history, prior, runtime):
    if (history.get("host") != runtime["host"] or normalize_gpu_uuid(history.get("gpu_uuid")) != normalize_gpu_uuid(runtime["gpu"]["uuid"])
            or type(history.get("original_b04_failure")) is not bool
            or not history.get("reviewer") or not history.get("basis")):
        raise ValueError("4060 history must bind reviewer, basis, host and GPU; unknown is blocked")
    if history["original_b04_failure"] and not prior:
        raise ValueError("original B04 failure evidence/retest required")
    if prior and (prior.get("host") != runtime["host"] or normalize_gpu_uuid(prior.get("gpu_uuid")) != normalize_gpu_uuid(runtime["gpu"]["uuid"])
                  or prior.get("test") != "test_verification_cached_only_for_unchanged_file_and_process"
                  or len(prior.get("original_evidence_sha256", "")) != 64):
        raise ValueError("original B04 failure identity/evidence incomplete")


def admit_4060(source_config, output, *, candidate, owner, history_path, previous_failure=None):
    """Full grid + common final windows. No training configs or runs are released."""
    if candidate not in {"turn", "fine"} or not owner.strip():
        raise ValueError("4060 admission requires owner and TURN/FINE candidate")
    source, assets, runtime = source_context(source_config, family="4060")
    history = read_json(history_path)
    prior = read_json(previous_failure) if previous_failure else None
    check_history(history, prior, runtime)
    root = stage_path(output, stage="preliminary")
    root.mkdir(parents=True, exist_ok=False)
    methods, rows, failures, entries = ("ce", candidate), [], [], []
    try:
        for method in methods:
            check_root = root / f"checks-{method}"
            check_root.mkdir()
            check_software(pilot.software_checks(check_root, method=method), method, runtime)
            entries.append(evidence(check_root / "checks.json"))
        # Full suite includes the named regression. On the original failed
        # machine additionally run that exact test by name and preserve output.
        if history["original_b04_failure"]:
            command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v", "-k",
                       "test_verification_cached_only_for_unchanged_file_and_process"]
            with (root / "b04-retest.log").open("w") as handle:
                result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
            log = (root / "b04-retest.log").read_text()
            if result.returncode or "Ran 1 test" not in log or "\nOK\n" not in log or "skipped" in log:
                raise ValueError("original B04 regression did not run and pass exactly once")
        for method in methods:
            for batch, workers in admission.GRID:
                target = root / "grid" / f"{method}-m{batch}-w{workers}"
                try:
                    row = admission._profile_one(None, method=method, adaptation="lora", microbatch=batch, workers=workers,
                        machine=source.get("machine_config"), output=target, _assets=assets,
                        _student_factory=pilot.student_factory, _stage="preliminary", _zero_selection_policy=policy(method))
                    check_profile(row, method=method, assets=assets, runtime=runtime,
                                  engineering={"microbatch": batch, "workers": workers}, windows=1)
                    rows.append(row)
                    entries.append(evidence(target / "profile.json"))
                except MethodError:
                    raise  # Method failure is not an engineering grid miss.
                except Exception:
                    if not (target / "failure.json").is_file():
                        raise
                    failures.append(evidence(target / "failure.json"))
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        choice = admission.choose_common(rows, methods)
        for method in methods:
            target = root / "final" / method
            row = admission._profile_one(None, method=method, adaptation="lora", **choice, eval_windows=3,
                machine=source.get("machine_config"), output=target, _assets=assets, _student_factory=pilot.student_factory,
                _stage="preliminary", _zero_selection_policy=policy(method))
            check_profile(row, method=method, assets=assets, runtime=runtime, engineering=choice, windows=3)
            entries.append(evidence(target / "profile.json"))
        value = sealed({"version": VERSION, "kind": "4060_admission_only", "status": "passed", "stage": "preliminary",
            "owner": owner, "runtime": runtime, "methods": methods, "engineering": choice, "machine_history": history,
            "previous_failure": prior, "source_sha256": file_sha256(source_config), "assets": assets.descriptor,
            "history_evidence": evidence(history_path),
            "previous_failure_evidence": evidence(previous_failure) if previous_failure else None,
            "b04_retest": evidence(root / "b04-retest.log") if history["original_b04_failure"] else None,
            "evidence": entries, "failures": failures, "training_authorized_by_receipt": False})
        write_json(root / "admission.json", value)
        return value
    except BaseException as exc:
        engine.record_failure(root, exc)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="初赛同期LoRA对照；T4训练、4060仅准入；不下载")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="仅元数据，保持准入阻塞")
    p.add_argument("--output", required=True)
    p = sub.add_parser("startup-check", help="仅四张合成图、两次更新，不自动训练")
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    for command in ("profile", "admit-t4", "admit-4060"):
        p = sub.add_parser(command)
        p.add_argument("--source-config", required=True)
        p.add_argument("--output", required=True)
        if command == "profile":
            p.add_argument("--method", choices=METHODS, required=True)
            p.add_argument("--barrier", help=argparse.SUPPRESS)
        else:
            p.add_argument("--owner", required=True)
        if command == "admit-t4":
            p.add_argument("--methods", choices=METHODS, nargs="+", default=["ce", "fine", "snscl"])
            p.add_argument("--gpu-uuids", nargs="+", required=True)
        if command == "admit-4060":
            p.add_argument("--candidate", choices=("turn", "fine"), required=True)
            p.add_argument("--machine-history", required=True)
            p.add_argument("--previous-failure")
    p = sub.add_parser("run")
    p.add_argument("--receipt", required=True)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true", help="仅负责人显式恢复中断run；第10轮拒绝续跑")
    p = sub.add_parser("compare")
    p.add_argument("--control", required=True)
    p.add_argument("--candidate", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = plan(args.output)
        elif args.command == "startup-check":
            result = admission.synthetic_check(args.method, "lora", device=args.device, precision=args.precision,
                                               zero_selection_policy=policy(args.method))
        elif args.command == "profile":
            result = run_profile(args.source_config, args.output, method=args.method, barrier=args.barrier)
        elif args.command == "admit-t4":
            result = admit_t4(args.source_config, args.output, methods=args.methods, gpu_uuids=args.gpu_uuids, owner=args.owner)
        elif args.command == "admit-4060":
            result = admit_4060(args.source_config, args.output, candidate=args.candidate, owner=args.owner,
                               history_path=args.machine_history, previous_failure=args.previous_failure)
        elif args.command == "compare":
            result = compare(args.control, args.candidate)
        else:
            result = run(args.receipt, args.output, method=args.method, resume=args.resume)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"preliminary comparison stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
