"""Run with python -m aic_robust_clip.round2; all provisioning is explicit."""
import argparse
import json
import sys
from pathlib import Path
from ..contracts import read_json
from .config import prepare_configs, check_config


def parser():
    p = argparse.ArgumentParser(description="复赛独立入口；配置准备不训练、不下载。")
    sub = p.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="生成八个配置；缺资产/准入时保留 blocked 状态")
    prepare.add_argument("--assets")
    prepare.add_argument("--receipt")
    prepare.add_argument("--output", required=True)
    check = sub.add_parser("check")
    check.add_argument("--config", required=True)
    audit = sub.add_parser("audit", help="仅审计用户取得的复赛训练 ZIP")
    for name in ("archive", "output", "source-url", "retrieved-at", "organizer-version", "weights", "weight-revision"):
        audit.add_argument("--" + name, required=True)
    audit.add_argument("--member-prefix", help="训练ZIP内部显式根目录，如train；不改写成员路径或图片")
    for name in ("cache", "init-head"):
        q = sub.add_parser(name)
        q.add_argument("--assets", required=True)
        q.add_argument("--machine", required=True)
    q = sub.add_parser("checks")
    q.add_argument("--group", choices=("t4", "4060-a", "4060-b"), required=True)
    q.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    q.add_argument("--previous-failure")
    q.add_argument("--machine-history", help="4060 必填：原 B04 失败机器身份核对记录")
    q.add_argument("--output", required=True)
    q = sub.add_parser("startup-check", help="四张合成图、两次更新；不自动继续")
    q.add_argument("--method", choices=("ce", "turn", "fine", "snscl"), required=True)
    q.add_argument("--adaptation", choices=("full_visual", "lora"), required=True)
    q.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    q.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    for name in ("profile", "profile-one"):
        q = sub.add_parser(name)
        for key in ("assets", "machine", "output"):
            q.add_argument("--" + key, required=True)
        q.add_argument("--method", choices=("ce", "turn", "fine", "snscl"), required=name == "profile-one")
        q.add_argument("--eval-windows", type=int, choices=(1, 3), default=1)
        if name == "profile":
            q.add_argument("--group", choices=("t4", "4060-a", "4060-b"), required=True)
            q.add_argument("--choice", help="通过 choose 得到的共同配置，固定后做三次 eval 窗口")
        else:
            q.add_argument("--adaptation", choices=("full_visual", "lora"), required=True)
            q.add_argument("--microbatch", type=int, choices=(4, 8, 16, 32), required=True)
            q.add_argument("--workers", type=int, choices=(2, 4), required=True)
            q.add_argument("--barrier", help=argparse.SUPPRESS)
    q = sub.add_parser("choose")
    q.add_argument("--profiles", nargs="+", required=True)
    q.add_argument("--group", choices=("t4", "4060-a", "4060-b"), required=True)
    q.add_argument("--output", required=True)
    q = sub.add_parser("concurrent")
    for key in ("assets", "machine", "choice", "output"):
        q.add_argument("--" + key, required=True)
    q.add_argument("--gpu-uuids", nargs=4, required=True)
    q = sub.add_parser("admit")
    for key in ("assets", "owner", "output"):
        q.add_argument("--" + key, required=True)
    q.add_argument("--group", choices=("t4", "4060-a", "4060-b"), required=True)
    for key in ("checks", "profiles", "final-profiles"):
        q.add_argument("--" + key, nargs="+", required=True)
    q.add_argument("--concurrency")
    q.add_argument("--previous-failure-required", action="store_true")
    q = sub.add_parser("run", help="完整30轮配方；本次只允许执行到第10轮保存后暂停")
    q.add_argument("--config", required=True)
    q.add_argument("--machine", required=True)
    q.add_argument("--resume", action="store_true", help="只恢复本 run 的 last.pt；不能越过第10轮")
    q = sub.add_parser("replay-dev")
    for key in ("config", "machine", "output"):
        q.add_argument("--" + key, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        from . import admission, assets, engine
        c = args.command
        if c == "prepare":
            value = prepare_configs(args.output, args.assets, args.receipt)
        elif c == "check":
            value, _ = check_config(args.config, ready=False)
        elif c == "audit":
            value = assets.audit(args.archive, args.output, source_url=args.source_url, retrieved_at=args.retrieved_at,
                organizer_version=args.organizer_version, weights=args.weights, weight_revision=args.weight_revision,
                member_prefix=args.member_prefix)
        elif c in {"cache", "init-head"}:
            value = (assets.cache if c == "cache" else assets.init_head)(args.assets, machine=args.machine)
        elif c == "checks":
            value = admission.checks(args.output, group=args.group, device=args.device,
                                     previous_failure=args.previous_failure, machine_history=args.machine_history)
        elif c == "startup-check":
            value = admission.synthetic_check(args.method, args.adaptation, device=args.device, precision=args.precision)
        elif c == "profile":
            choice = read_json(args.choice) if args.choice else None
            if choice:
                from .config import verify_seal
                verify_seal(choice)
                if choice["group"] != args.group:
                    raise ValueError("engineering choice group mismatch")
            engineering = (choice["engineering"]["microbatch"], choice["engineering"]["workers"]) if choice else None
            value = admission.profile(args.assets, group=args.group, machine=args.machine, output=args.output,
                                      method=args.method, engineering=engineering, eval_windows=args.eval_windows)
        elif c == "profile-one":
            value = admission._profile_one(args.assets, method=args.method, adaptation=args.adaptation,
                microbatch=args.microbatch, workers=args.workers, machine=args.machine, output=args.output,
                eval_windows=args.eval_windows, barrier=args.barrier)
        elif c == "choose":
            value = admission.choose(args.profiles, group=args.group, output=args.output)
        elif c == "concurrent":
            value = admission.concurrent(args.assets, machine=args.machine, choice_path=args.choice,
                                         gpu_uuids=args.gpu_uuids, output=args.output)
        elif c == "admit":
            value = admission.admit(args.assets, group=args.group, owner=args.owner, check_paths=args.checks,
                profile_paths=args.profiles, final_paths=args.final_profiles, output=args.output,
                concurrency=args.concurrency, previous_failure_required=args.previous_failure_required)
        elif c == "run":
            value = engine.run(args.config, machine=args.machine, resume=args.resume)
        else:
            value = engine.replay(args.config, machine=args.machine, output=args.output)
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"round2 stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
