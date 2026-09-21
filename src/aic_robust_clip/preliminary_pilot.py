"""Explicitly authorized preliminary TURN pilot, never a round2 asset source.

One command completes software checks, real-model profiling, ten full epochs
under a thirty-epoch schedule, student replay and a morning report. No downloads.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from .configuration import load_config, prepare
from .contracts import read_json, write_json
from .models.provision import file_sha256
from .workflow import dataset_for, load_bundle, load_head
from .round2 import admission, engine
from .round2.config import RECIPE, sealed, stage_path
from .round2.model import Student

VERSION = "preliminary-turn-abstain-pilot-v2"
PURPOSE = "preliminary_method_validation_only"
ZERO_SELECTION_POLICY = "retain_observed"
VARIANT = "TURN-ABSTAIN-B32-v1"


def assert_source(ctx):
    if ctx.run.stage != "preliminary" or ctx.run.execution_mode != "formal" or ctx.run.seed != 17:
        raise ValueError("pilot requires the existing seed17 preliminary formal source")
    if (ctx.config["recipe"] != "B03" or ctx.train.objective != "ce" or ctx.train.weighting
            or ctx.train.lambda_preserve or ctx.train.prior_tau):
        raise ValueError("use the existing plain CE B03 source, not an old research candidate")
    if ctx.config.get("effective_batch_size", 128) != 128:
        raise ValueError("source effective batch must be 128")


class PilotAssets:
    def __init__(self, ctx, head_sha256):
        assert_source(ctx)
        self.ctx, self.class_map = ctx, ctx.class_map
        self.head_sha256 = head_sha256
        self.descriptor = sealed({"version": VERSION, "stage": "preliminary", "purpose": PURPOSE,
            "manifest": ctx.run.manifest_digest, "split": ctx.run.split_digest,
            "class_map": ctx.run.class_map_digest, "weights": ctx.weights,
            "preprocessing": ctx.preprocessing_digest, "head_sha256": head_sha256,
            "initializer": "verified_preliminary_HEAD3", "seed": 17})

    def dataset(self, partition, processor=None, online=False, purpose=None):
        if partition not in {"train", "dev"}:
            raise ValueError("pilot accepts preliminary train/dev only")
        return dataset_for(self.ctx, partition, processor, online=online, purpose=purpose)


def student_factory(assets, adaptation):
    if adaptation != "lora":
        raise ValueError("night pilot is fixed to Q/V LoRA")
    head, digest = load_head(assets.ctx)
    if digest != assets.head_sha256:
        raise ValueError("preliminary initializer changed")
    bundle = load_bundle(assets.ctx)
    student = Student(bundle.encoder, len(assets.class_map.id_to_index), "lora")
    student.classifier.load_state_dict(head)
    return student, bundle.processor, {"sha256": digest}


def software_checks(root):
    script = '''import json,sys,unittest
r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover("tests"))
with open(sys.argv[1],"w") as f: json.dump({"tests":r.testsRun,"failures":len(r.failures),"errors":len(r.errors),"skips":len(r.skipped)},f)
sys.exit(not r.wasSuccessful() or bool(r.skipped))
'''
    commands = [[sys.executable, "-c", script, str(root / "suite.json")],
                [sys.executable, "-m", "compileall", "-q", "src", "tests"],
                [sys.executable, "-m", "pip", "check"], ["git", "diff", "--check"]]
    for index, command in enumerate(commands):
        with (root / f"check-{index}.log").open("w") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            raise ValueError(f"software check {index} failed; see preserved log")
    startup = [admission.synthetic_check("turn", "lora", device="cuda", precision=precision,
                                       zero_selection_policy=ZERO_SELECTION_POLICY)
               for precision in ("fp32", "fp16")]
    value = sealed({"version": VERSION, "kind": "pilot_checks", "purpose": PURPOSE,
                    "runtime": admission.runtime_identity(), "suite": read_json(root / "suite.json"), "startup": startup,
                    "logs": {f"check-{i}.log": file_sha256(root / f"check-{i}.log") for i in range(4)}})
    write_json(root / "checks.json", value)
    return value


def pilot_config(root, assets, runtime, checks, profile):
    recipe = copy.deepcopy(RECIPE)
    recipe.update(stage="preliminary", initializer="verified_preliminary_HEAD3",
                  zero_selection_policy=ZERO_SELECTION_POLICY, method_variant=VARIANT)
    return sealed({"version": VERSION, "stage": "preliminary", "purpose": PURPOSE, "method": "turn",
        "adaptation": "lora", "group": "preliminary-pilot", "status": "pilot_ready", "recipe": recipe,
        "engineering": {"microbatch": 32, "workers": 2}, "output": str(root / "run"),
        "asset_digest": assets.descriptor["digest"], "head_sha256": assets.head_sha256,
        "runtime": runtime, "checks_digest": checks["digest"], "profile_digest": profile["digest"],
        "round2_eligible": False, "strict_matched_ce_control": False})


def write_morning_report(root, config, result, replay):
    rows = [read_json(root / "run" / f"epoch-{epoch:02d}.json") for epoch in range(1, 11)]
    value = {"version": VERSION, "stage": "preliminary", "purpose": PURPOSE, "status": result["status"],
             "full_epochs": 30, "completed_epochs": 10, "best_epoch": result["best_epoch"],
             "best": result["best_metrics"], "last": result["last"], "configuration_digest": config["digest"],
             "runtime": config["runtime"], "asset_digest": config["asset_digest"], "head_sha256": config["head_sha256"],
             "checkpoint_sha256": result["checkpoint_sha256"], "best_sha256": result["best_sha256"],
             "replay": replay, "epoch_seconds": [r["epoch_seconds"] for r in rows],
             "selected_for_next_epoch": [r["selected"] for r in rows], "round2_eligible": False,
             "method_variant": VARIANT, "zero_selection_policy": ZERO_SELECTION_POLICY,
             "selection_summaries": [r.get("selection", {}).get("summary", {}) for r in rows],
             "comparison": "old ten-epoch CE is contextual only; schedule differs; no causal gain claim"}
    write_json(root / "morning-summary.json", value)
    lines = ["# 初赛 TURN-ABSTAIN 验证结果", "", "仅为初赛验证，不能用作复赛初始化或结果。", "",
             "仅在GMM收敛后零可信样本时弃权筛选、保留原监督；保留数不等于可信数。", "",
             f"完整计划30轮，已完成10轮并暂停；best 轮次：{result['best_epoch']}。",
             f"Best macro recall：{result['best_metrics']['macro_recall']:.6%}；last：{result['last']['macro_recall']:.6%}。",
             "旧 CE 为10轮预算，本任务为30轮预算下的10轮观察点，不能据绝对分数作严格因果对照。", "",
             "| 轮次 | macro recall | micro top1 | 下轮选中数 | 总轮秒数 |",
             "| --- | --- | --- | --- | --- |"]
    lines.extend(f"| {r['epoch']} | {r['metrics']['macro_recall']:.6%} | {r['metrics']['micro_top1']:.6%} | {r['selected']} | {r['epoch_seconds']:.2f} |" for r in rows)
    lines.extend(["", f"学生 dev 回放通过：{replay['passed']}。", "", "完整逐类、逐图、筛选与开销见 run/；摘要见 morning-summary.json。"])
    (root / "MORNING.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return value


def run_night(source_config, output):
    source = load_config(source_config)
    if source.get("stage") != "preliminary":
        raise ValueError("night pilot cannot consume round2 or semifinal source configurations")
    engine.require_machine(source.get("machine_config"), stage="preliminary")
    ctx = prepare(source_config)
    assert_source(ctx)
    runtime = admission.runtime_identity()
    if "T4" not in runtime.get("gpu", {}).get("name", ""):
        raise ValueError("night pilot targets the assigned T4 server")
    _, head_sha = load_head(ctx)
    assets = PilotAssets(ctx, head_sha)
    root = stage_path(output, stage="preliminary")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "provenance.json", {"source_configuration": str(Path(source_config).resolve()),
        "source_configuration_sha256": file_sha256(source_config), "source_summary": ctx.summary(),
        "asset_identity": assets.descriptor, "runtime": runtime, "purpose": PURPOSE})
    try:
        checks = software_checks(root)
        # One declared engineering setting, not the round2 multi-method grid.
        # 2 warmup + 10 real updates, full fixed-view scoring and 3 dev windows.
        profile = admission._profile_one(None, method="turn", adaptation="lora", microbatch=32, workers=2,
            machine=source.get("machine_config"), output=root / "profile", eval_windows=3,
            _assets=assets, _student_factory=student_factory, _stage="preliminary", _zero_selection_policy=ZERO_SELECTION_POLICY)
        if (profile["status"] != "passed" or profile["runtime"] != runtime or profile["head_sha256"] != head_sha
                or profile.get("zero_selection_policy") != ZERO_SELECTION_POLICY):
            raise ValueError("pilot profile identity mismatch")
        config = pilot_config(root, assets, runtime, checks, profile)
        write_json(root / "config.json", config)
        # The loop reloads HEAD3 and resets seed; profile updates are discarded.
        result = engine._run_prepared(config, assets, student_factory=student_factory,
                                      expected_stage="preliminary", purpose=PURPOSE)
        replay = engine._replay_prepared(config, assets, output=root / "replay-dev", student_factory=student_factory,
                                         expected_stage="preliminary", purpose=PURPOSE)
        return write_morning_report(root, config, result, replay)
    except BaseException as exc:
        engine.record_failure(root, exc)
        (root / "MORNING-FAILED.md").write_text(
            "# 初赛夜间验证未完成\n\n" + repr(exc) + "\n\n保留全部日志；未自动重试、改参或替换方法。\n", encoding="utf-8")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="初赛 LoRA+TURN 独立夜间验证；30轮计划，10轮暂停；不下载")
    parser.add_argument("--source-config", help="服务器已有的初赛 seed17 B03/CE 正式配置")
    parser.add_argument("--output", required=True, help="全新的 preliminary 输出目录")
    parser.add_argument("--startup-check", action="store_true", help="仅随机微型模型四张合成图、两次更新；不读取真实资产或继续训练")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda", help="仅启动检查使用")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32", help="仅启动检查使用")
    args = parser.parse_args(argv)
    if not args.startup_check and not args.source_config:
        parser.error("formal pilot requires --source-config")
    try:
        if args.startup_check:
            root = stage_path(args.output, stage="preliminary")
            root.mkdir(parents=True, exist_ok=False)
            result = admission.synthetic_check("turn", "lora", device=args.device, precision=args.precision,
                                               zero_selection_policy=ZERO_SELECTION_POLICY)
            write_json(root / "startup.json", {**result, "stage": "preliminary", "status": "startup_only"})
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        print(json.dumps(run_night(args.source_config, args.output), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"preliminary pilot stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
