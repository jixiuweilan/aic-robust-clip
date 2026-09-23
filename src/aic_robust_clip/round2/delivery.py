"""Offline round2 jobs with explicit stages, durable logs and fail-closed gates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..contracts import read_json, write_json
from ..gpu_identity import cuda_gpu_uuids
from ..models.provision import file_sha256
from ..runtime import current_code_revision
from .config import RUNS, RECIPE, CLOUD_GROUPS, GROUPS, adaptation_for, stage_path
from .journal import atomic_json, task_journal

SCHEMA = "round2-delivery-v1"
_children = []


def validate_job(job):
    if job.get("schema") != SCHEMA or job.get("stage") != "second_round":
        raise ValueError("复赛专用任务描述必填；拒绝初赛启动包")
    if job.get("action") not in {"public-assets", "checks", "group", "train"}:
        raise ValueError("未知任务阶段")
    stage_path(job["output"])
    if job["action"] == "public-assets":
        cuda_gpu_uuids([job["gpu_uuid"]])
        stage_path(job["archive"])
        if Path(job["archive"]).name.lower() != "train.zip":
            raise ValueError("公共资产入口只接收 train.zip")
        digest = job["expected_sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("填写已登记的 train.zip SHA256")
        for name in ("source_url", "retrieved_at", "organizer_version", "weights", "weight_revision", "machine"):
            if not job.get(name):
                raise ValueError(f"缺少 {name}")
    elif job["action"] in {"checks", "group"}:
        group = job["group"]
        if group not in GROUPS:
            raise ValueError("未知准入组")
        from .admission import methods_for
        methods_for(group, job.get("methods"))
        uuids = cuda_gpu_uuids(job["gpu_uuids"])
        if len(uuids) != (4 if group == "t4" else 1):
            raise ValueError("准入组 GPU 数量错误")
        if group in {"4060-a", "4060-b"} and not job.get("machine_history"):
            raise ValueError("4060 必须提供 B04 机器历史核对记录")
        if job["action"] == "group":
            stage_path(job["assets"])
            if not job.get("owner") or not job.get("machine"):
                raise ValueError("准入需要执行人和机器绑定")
    else:
        if job.get("run_id") not in RUNS:
            raise ValueError("未知复赛 run")
        stage_path(job["config"])
        cuda_gpu_uuids([job["gpu_uuid"]])
        if not job.get("machine"):
            raise ValueError("缺少机器绑定")
    return job


def launch(job_path):
    job = validate_job(read_json(job_path))
    job = {**job, "source": current_code_revision()}
    root = stage_path(job["output"])
    root.mkdir(parents=True, exist_ok=False)
    # Copy exact job values. Never depend on a mutable external task file.
    atomic_json(root / "job.json", job)
    atomic_json(root / "status.json", {"status": "launching", "exit_code": None, "started_at": time.time()})
    with (root / "console.log").open("xb") as log:
        process = subprocess.Popen([sys.executable, "-u", "-m", "aic_robust_clip.round2.delivery",
                                    "execute", "--job", str(root / "job.json")],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True)
    _children.append(process)
    atomic_json(root / "launch.json", {"pid": process.pid, "job_sha256": file_sha256(root / "job.json"),
                                       "source": current_code_revision(), "status": "launched_not_completed"})
    return {"output": str(root), "pid": process.pid, "status": "launched_not_completed"}


def execute(job_path):
    job = validate_job(read_json(job_path))
    root = stage_path(job["output"])
    if Path(job_path).resolve() != root / "job.json" or not (root / "status.json").is_file():
        raise ValueError("请使用 launch 创建独立任务目录")
    # Exclusive claim prevents two executors or accidental reruns overwriting evidence.
    with (root / "executor.json").open("x") as handle:
        json.dump({"pid": os.getpid(), "source": current_code_revision()}, handle)
    with task_journal(root) as journal:
        if job.get("source") != current_code_revision():
            raise ValueError("任务启动后源码已变化；必须使用新checkout和新任务")
        ordinal = 0
        def step(name, args, uuid=None):
            nonlocal ordinal
            if job["source"] != current_code_revision():
                raise ValueError("任务执行中源码已变化；停止，重做受影响准入")
            ordinal += 1
            journal.update(phase=name, step=ordinal)
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            if uuid:
                env["CUDA_VISIBLE_DEVICES"] = cuda_gpu_uuids([uuid])[0]
            command = [sys.executable, "-u", "-m", "aic_robust_clip.round2", *map(str, args)]
            log_path = root / f"{ordinal:02d}-{name}.log"
            with log_path.open("x") as handle:
                child = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
                try:
                    code = child.wait()
                finally:
                    if child.poll() is None:
                        child.terminate()
                        child.wait()
            atomic_json(root / f"{ordinal:02d}-{name}.json", {"command": command, "exit_code": code,
                        "log_sha256": file_sha256(log_path), "gpu_uuid": uuid, "source": current_code_revision()})
            if code:
                raise RuntimeError(f"{name} 退出码 {code}；停止所属任务，保留日志")

        if job["action"] == "public-assets":
            target = root / "assets"
            args = ["audit", "--output", target]
            for key in ("archive", "source_url", "retrieved_at", "organizer_version", "weights", "weight_revision", "expected_sha256"):
                args += ["--" + key.replace("_", "-"), job[key]]
            if job.get("member_prefix"):
                args += ["--member-prefix", job["member_prefix"]]
            step("audit", args)
            for command in ("cache", "init-head"):
                step(command, [command, "--assets", target / "assets.json", "--machine", job["machine"]], job.get("gpu_uuid"))
            from .config import load_assets, head_descriptor
            assets = load_assets(target / "assets.json", verify_archives=True)
            atomic_json(root / "public-assets.json", {"asset_digest": assets.descriptor["digest"],
                        "head": head_descriptor(assets), "source": current_code_revision()})
        elif job["action"] in {"checks", "group"}:
            group = job["group"]
            from .admission import methods_for
            methods = methods_for(group, job.get("methods"))
            uuids = cuda_gpu_uuids(job["gpu_uuids"])
            checks = []
            # Existing checks may be reused only if admit verifies current source and runtime.
            for index, uuid in enumerate(uuids):
                target = root / f"checks-{index}"
                args = ["checks", "--group", group, "--output", target]
                if group in CLOUD_GROUPS:
                    args += ["--methods", *methods]
                for key in ("machine_history", "previous_failure"):
                    if job.get(key):
                        args += ["--" + key.replace("_", "-"), job[key]]
                step(f"checks-{index}", args, uuid)
                checks.append(target / "checks.json")
            if job["action"] == "group":
                assigned = dict(zip(methods, uuids if group == "t4" else uuids * len(methods)))
                common = ["--assets", job["assets"], "--machine", job["machine"]]
                feasibility = []
                for method in methods:
                    target = root / f"feasibility-{method}"
                    step(f"feasibility-{method}", ["feasibility", *common, "--method", method,
                         "--adaptation", adaptation_for(group), "--output", target], assigned[method])
                    feasibility.append(target / "feasibility.json")
                profiles, finals = [], []
                for phase, collection in (("grid", profiles), ("final", finals)):
                    for method in methods:
                        target = root / f"{phase}-{method}"
                        args = ["profile", *common, "--group", group, "--method", method, "--output", target]
                        if phase == "final":
                            args += ["--choice", root / "choice.json", "--eval-windows", "3"]
                        step(f"{phase}-{method}", args, assigned[method])
                        collection.append(target / "profiles.json")
                        from .admission import read_profiles
                        if any(row.get("method_failure") for row in read_profiles([collection[-1]])):
                            raise ValueError("方法数值/筛选失败，停止整个准入组")
                    if phase == "grid":
                        args = ["choose", "--group", group, "--profiles", *profiles, "--output", root / "choice.json"]
                        if group in CLOUD_GROUPS:
                            args += ["--methods", *methods]
                        step("choose", args)
                if group == "t4":
                    step("concurrent", ["concurrent", *common, "--choice", root / "choice.json", "--gpu-uuids", *uuids,
                                        "--output", root / "concurrent"])
                args = ["admit", "--assets", job["assets"], "--group", group, "--owner", job["owner"],
                        "--checks", *checks, "--profiles", *profiles, "--final-profiles", *finals,
                        "--feasibility", *feasibility, "--output", root / "admission.json"]
                if group == "t4":
                    args += ["--concurrency", root / "concurrent/concurrency.json"]
                if group in CLOUD_GROUPS:
                    args += ["--methods", *methods]
                step("admit", args)
                step("prepare", ["prepare", "--assets", job["assets"], "--receipt", root / "admission.json",
                                 "--output", root / "configs"])
        else:
            cfg = read_json(job["config"])
            if cfg.get("run_id") != job["run_id"]:
                raise ValueError("任务与配置 run_id 不一致")
            resume = bool(job.get("resume", False))
            completed = 0
            if resume:
                import torch
                saved = torch.load(Path(cfg["output"]) / "last.pt", weights_only=False, map_location="cpu")
                completed = saved["method"]["completed_epochs"]
                if completed >= 10:
                    raise ValueError("第十轮已暂停，禁止自动续跑")
            boundaries = [1, 5, 6, 10] if cfg["method"] == "snscl" else [1, 10]
            for epoch in boundaries:
                if epoch <= completed:
                    continue
                args = ["run", "--config", job["config"], "--machine", job["machine"], "--stop-after-epoch", epoch]
                if resume:
                    args += ["--resume"]
                step(f"epoch-{epoch}", args, job["gpu_uuid"])
                observation = verify_observation(cfg, epoch)
                atomic_json(root / f"observation-{epoch:02d}.json", observation)
                resume = True
            step("replay-dev", ["replay-dev", "--config", job["config"], "--machine", job["machine"],
                                "--output", root / "replay-dev"], job["gpu_uuid"])
        journal.update(phase="verified")
    return read_json(root / "status.json")


def verify_observation(config, epoch):
    """Inspect complete epoch artifacts before allowing the next observation."""
    import torch
    root = Path(config["output"])
    log = read_json(root / f"epoch-{epoch:02d}.json")
    result = read_json(root / "result.json")
    payload = torch.load(root / "last.pt", weights_only=False, map_location="cpu")
    required = {"student", "optimizer", "scaler", "method", "auxiliary", "method_rng", "rng", "sampler", "scheduler", "history"}
    if (not required <= set(payload) or payload["method"]["completed_epochs"] != epoch
            or payload["scheduler"] != {"total_epochs": 30, "completed_epochs": epoch}
            or payload["identity"] != {"configuration": config["digest"], "source": current_code_revision(), "purpose": "formal"}
            or len(payload["history"]) != epoch or log["optimizer_updates"] <= 0
            or result["completed_epochs"] != epoch or result["full_epochs"] != 30
            or result["status"] != ("paused_at_epoch10" if epoch == 10 else "paused_at_observation")
            or result["checkpoint_sha256"] != file_sha256(root / "last.pt")
            or log["checkpoint_sha256"] != result["checkpoint_sha256"]
            or result["best_sha256"] != file_sha256(root / "best.pt")):
        raise ValueError("完整检查点/调度/首轮更新/摘要校验失败")
    dev = read_json(root / f"dev-epoch-{epoch:02d}.json")
    if dev["metrics"] != log["metrics"] or not (root / f"dev-epoch-{epoch:02d}.pt").is_file():
        raise ValueError("轮末 dev 证据不完整")
    if config["method"] == "snscl" and epoch >= 5:
        if payload["auxiliary"] is None or not payload["method"]["report"]:
            raise ValueError("SNSCL 第五轮评分/辅助状态缺失")
        if epoch >= 6 and sum(log["queue_counts"]) <= 0:
            raise ValueError("SNSCL 第六轮队列未更新")
    return {"epoch": epoch, "status": "verified", "checkpoint_sha256": result["checkpoint_sha256"],
            "metrics": log["metrics"], "optimizer_updates": log["optimizer_updates"], "full_epochs": 30}


def status(output):
    for child in list(_children):
        if child.poll() is not None:
            _children.remove(child)
    root = stage_path(output)
    value = read_json(root / "status.json")
    # No PID liveness claim: after SIGKILL/reboot 'running' is explicitly unverified.
    return {**value, "completion_verified": value.get("status") == "succeeded" and value.get("exit_code") == 0,
            "note": "仅 succeeded 且退出码0表示本阶段完成；running/launching不证明进程存活或成功"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="复赛离线交付任务；launch 不等于任务完成")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("launch", "execute", "validate"):
        p = sub.add_parser(command)
        p.add_argument("--job", required=True)
    p = sub.add_parser("status")
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        value = status(args.output) if args.command == "status" else (
            validate_job(read_json(args.job)) if args.command == "validate" else
            launch(args.job) if args.command == "launch" else execute(args.job))
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"round2 delivery stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
