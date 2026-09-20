"""Offline 4060 task-package regressions; synthetic metadata, no CUDA jobs."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aic_robust_clip import rtx4060 as r
from aic_robust_clip.configuration import load_config
from aic_robust_clip.contracts import RunConfig, read_json, write_json
from aic_robust_clip.training.baseline import TrainConfig
from aic_robust_clip.performance import PerformanceConfig


def fixture(root, member="A"):
    root.mkdir(parents=True, exist_ok=True)
    for name in ("manifest", "split", "class_map"):
        write_json(root / f"{name}.json", {"synthetic": name})
    write_json(root / "head/head.json", {"identity": {"profile": "HEAD3"}, "sha256": "a" * 64,
        "result": {"stopped_by_limit": False, "epoch_logs": [{"complete": True}] * 3}})
    source = {"schema_version": 2, "recipe": "B03", "stage": "preliminary", "seed": 17,
        "execution_mode": "formal", "device": "cuda", "batch_size": 16, "effective_batch_size": 128,
        "parameters": {}, "head": "head", "output": "old-output", "weights": "weights",
        "weight_revision": "b" * 40, "machine_config": "machine.json",
        **{n: f"{n}.json" for n in ("manifest", "split", "class_map")}}
    write_json(root / "source.json", source)
    r.prepare_bundle(root / "source.json", root / "bundle", member)
    return root / "bundle/bundle.json"


def binding(host="host-A"):
    return {"source_revision": "fixture-source", "fingerprint": host, "python": "fixture-python",
            "dependencies": {"torch": "fixture"}, "torch_cuda": "fixture", "gpu": {"name": "RTX 4060", "memory_bytes": 1000}}


def report(config, phase, bind):
    batch = 128 if phase == "train" else 64
    workers = config["performance"]["num_workers" if phase == "train" else "eval_num_workers"]
    return {"phase": phase, "kind": "benchmark_only", "selection_eligible": False,
        "configuration": {"config": config, "split_counts": {"train": 12800, "dev": 6400}},
        "source_revision": bind["source_revision"], "environment": {k: v for k, v in bind.items() if k != "source_revision"},
        "warmup_steps": 2, "measured_steps": 10, "measured_samples": batch * 10,
        "precision": config["parameters"]["precision"] if phase == "train" else "fp32",
        "prefetch_extra_samples_upper_bound": 0,
        "samples_per_second": batch, "checkpoint_write_seconds": 1 if phase == "train" else 0,
        "cold_start_and_warmup_seconds": 10, "data_wait_seconds": 2,
        "peak_gpu_reserved_bytes": 800, "peak_gpu_allocated_bytes": 750,
        "loader_after_cleanup": {"delivered_samples": batch * 12, "dispatch_stop": batch * 12,
            "num_workers": workers, "workers": [], "worker_failed": False,
            "worker_exits": [{"exitcode": 0, "alive": False, "forced": False}] * workers}}


def measured(root, member, bind):
    bundle_path = fixture(root, member)
    bundle = read_json(bundle_path)
    checks = root / "checks.json"
    write_json(checks, {"binding": bind, "status": "passed_zero_skips", "hashes": {}})
    for name, paths in bundle["configs"].items():
        cfg = load_config(paths["control"])
        for phase in ("train", "eval"):
            write_json(root / "windows" / name / phase / "benchmark.json", report(cfg, phase, bind))
    with patch.object(r, "_binding", return_value=bind):
        r.summarize(bundle_path, root / "windows", checks, root / "summary.json")
    return bundle_path


def joint_ready(root):
    bundle = measured(root / "A", "A", binding())
    measured(root / "B", "B", binding("host-B"))
    selection = root / "selection.json"
    selected = r.select(root / "A/summary.json", root / "B/summary.json", selection)
    config = load_config(read_json(bundle)["configs"][selected["profile"]]["control"])
    for i in range(1, 4):
        write_json(root / f"repeat/eval-{i}/benchmark.json", report(config, "eval", binding()))
    return bundle, selection


class RTX4060Tests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        mock = patch.object(r, "current_code_revision", return_value="fixture-source")
        mock.start(); self.addCleanup(mock.stop)

    def test_prepare_preserves_assets_schedule_and_lrs_with_isolated_outputs(self):
        outputs = []
        for member, lr in (("A", 3e-5), ("B", 3e-4)):
            root = self.root / member
            bundle = read_json(fixture(root, member))
            self.assertEqual(bundle["stop_after_epochs"], 3)
            self.assertEqual(set(bundle["configs"]), set(r.PROFILES))
            for name, paths in bundle["configs"].items():
                precision, micro, workers, eval_workers = r.PROFILES[name]
                for role, path in paths.items():
                    config = load_config(path)
                    p = config["parameters"]
                    perf = PerformanceConfig.from_config(config)
                    self.assertEqual((p["precision"], config["batch_size"], perf.num_workers, perf.eval_num_workers),
                                     (precision, micro, workers, eval_workers))
                    self.assertEqual(config["batch_size"] * p["accumulation_steps"], 128)
                    actual = TrainConfig.from_run(RunConfig(stage="preliminary", execution_mode="formal",
                        batch_size=micro, parameters=p))
                    self.assertEqual(actual.lora_learning_rate, 1e-4 if role == "control" else lr)
                    self.assertEqual((actual.epochs, actual.precision), (10, precision))
                    self.assertEqual(p["lora_learning_rate"], 1e-4 if role == "control" else lr)
                    self.assertEqual((p["learning_rate"], p["formal_epochs"], p["profile"]), (1e-3, 10, "ONLINE10"))
                    self.assertEqual(perf.eval_batch_size, 64)
                    self.assertEqual(config["head"], str(root / "head"))
                    outputs.append(config["output"])
                    self.assertFalse(Path(config["output"]).exists())
            self.assertFalse((root / "weights").exists())
            self.assertFalse((root / "head/head.pt").exists())  # prepare is metadata-only
            with self.assertRaises(FileExistsError):
                r.prepare_bundle(root / "source.json", root / "bundle", member)
        self.assertEqual(len(outputs), len(set(outputs)))

    def test_recipe_changes_and_incomplete_head_are_rejected(self):
        fixture(self.root)
        source = read_json(self.root / "source.json")
        for changes in ({"epochs": 3}, {"formal_epochs": 3}, {"objective": "gce"}, {"weighting": True},
                        {"lora_learning_rate": 3e-4}, {"learning_rate": .01}, {"weight_decay": .1}):
            value = copy.deepcopy(source)
            value["parameters"].update(changes)
            write_json(self.root / "bad.json", value)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                r.prepare_bundle(self.root / "bad.json", self.root / "bad-bundle", "A")
        head = read_json(self.root / "head/head.json")
        head["identity"]["profile"] = "HEAD-SMOKE"
        write_json(self.root / "head/head.json", head)
        with self.assertRaises(ValueError):
            r.prepare_bundle(self.root / "source.json", self.root / "bad-bundle", "A")

    def test_report_rejects_wrong_cleanup_memory_identity_and_bounds(self):
        bundle = read_json(fixture(self.root))
        config = load_config(bundle["configs"]["fp16-m64-w2-e2"]["control"])
        valid = report(config, "eval", binding())
        path = self.root / "window/benchmark.json"
        mutations = [lambda x: x.update(measured_steps=9), lambda x: x.update(precision="fp16"),
            lambda x: x.update(peak_gpu_reserved_bytes=851), lambda x: x.update(source_revision="old"),
            lambda x: x["loader_after_cleanup"].update(worker_exits=[]),
            lambda x: x["loader_after_cleanup"]["worker_exits"][0].update(exitcode=-6),
            lambda x: x["loader_after_cleanup"].update(dispatch_stop=769),
            lambda x: x.update(samples_per_second=float("nan"))]
        for mutate in mutations:
            bad = copy.deepcopy(valid); mutate(bad); write_json(path, bad)
            with self.assertRaises(ValueError):
                r._report(path, config, binding())
        write_json(path, valid)
        self.assertEqual(r._report(path, config, binding()), valid)

    def test_common_selection_tie_break_and_no_common_pass(self):
        measured(self.root / "A", "A", binding())
        measured(self.root / "B", "B", binding("host-B"))
        ap, bp = self.root / "A/summary.json", self.root / "B/summary.json"
        selected = r.select(ap, bp, self.root / "selection.json")
        self.assertEqual(selected["profile"], "fp16-m64-w2-e2")  # fewer train/eval workers
        a, b = read_json(ap), read_json(bp)
        for summary in (a, b):
            summary["profiles"]["fp32-m16-w4-e0"]["passed"] = False
            summary["profiles"]["fp16-m64-w2-e2"]["estimated_epoch_seconds"] *= 1.049
        write_json(ap, a); write_json(bp, b)
        selected = r.select(ap, bp, self.root / "selection2.json")
        self.assertEqual(selected["profile"], "fp16-m64-w2-e2")
        for summary in (a, b):
            summary["profiles"]["fp16-m64-w2-e2"]["estimated_epoch_seconds"] = 202 * 1.051
        write_json(ap, a); write_json(bp, b)
        selected = r.select(ap, bp, self.root / "selection-outside-tie.json")
        self.assertEqual(r.PROFILES[selected["profile"]][1:3], (32, 4))
        for entry in b["profiles"].values():
            entry["passed"] = False
        write_json(bp, b)
        with self.assertRaisesRegex(ValueError, "no common"):
            r.select(ap, bp, self.root / "selection3.json")

    def test_full_receipt_flow_pause_resume_and_stale_source_rejection(self):
        a = measured(self.root / "A", "A", binding())
        measured(self.root / "B", "B", binding("host-B"))
        selection_path = self.root / "selection.json"
        selected = r.select(self.root / "A/summary.json", self.root / "B/summary.json", selection_path)
        bundle = read_json(a)
        paths = bundle["configs"][selected["profile"]]
        cfg = load_config(paths["control"])
        for i in range(1, 4):
            write_json(self.root / f"repeat/eval-{i}/benchmark.json", report(cfg, "eval", binding()))
        receipt_path = self.root / "receipt.json"
        with patch.object(r, "_binding", return_value=binding()):
            receipt = r.accept(a, selection_path, self.root / "repeat", self.root / "A/checks.json", receipt_path)
            self.assertEqual(receipt["stop_after_epochs"], 3)
            with patch("aic_robust_clip.workflow.train_command", return_value={}) as train:
                r.run(a, receipt_path, "control")
                train.assert_called_once_with(paths["control"], resume=None, stop_after_epochs=3)
                r.run(a, receipt_path, "control", resume=True)
                self.assertEqual(train.call_args.kwargs["resume"], Path(cfg["output"]) / "last.pt")
                with self.assertRaises(OSError):
                    r.run(a, receipt_path, "candidate")
                write_json(Path(cfg["output"]) / "result.json", {"epoch_logs": [{"complete": True}] * 3, "paused": True, "completed_epochs": 3, "stopped_by_limit": False})
                r.run(a, receipt_path, "candidate")
                self.assertEqual(train.call_args.args[0], paths["candidate"])
            with patch.object(r, "current_code_revision", return_value="new-source"):
                with self.assertRaisesRegex(ValueError, "stale"):
                    r.run(a, receipt_path, "control")
            write_json(self.root / "repeat/eval-1/benchmark.json", {"changed": True})
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                r.run(a, receipt_path, "control")

    def test_accept_rejects_actual_single_host_selection_shape_and_renamed_copy(self):
        bundle, path = joint_ready(self.root)
        original = read_json(path)
        summary = original["summaries"]["A"]
        # Shape received in the real incident: B contains the exact A summary.
        manual = {"schema_version": 1, "profile": original["profile"],
                  "selected_by": "A_self_single_host", "summaries": {"A": summary, "B": summary}}
        renamed = copy.deepcopy(original)
        renamed["summaries"]["B"] = copy.deepcopy(summary)
        renamed["summaries"]["B"]["member"] = "B"  # relabeling cannot create a second host
        for name, bad in (("original-incident", manual), ("renamed-copy", renamed)):
            with self.subTest(name=name), patch.object(r, "_binding", return_value=binding()):
                write_json(path, bad)
                target = self.root / f"{name}-receipt.json"
                with self.assertRaisesRegex(ValueError, "two distinct"):
                    r.accept(bundle, path, self.root / "repeat", self.root / "A/checks.json", target)
                self.assertFalse(target.exists())

    def test_accept_recomputes_peer_contract_ranking_and_summary_digests(self):
        bundle, path = joint_ready(self.root)
        original = read_json(path)
        cases = {
            "missing-schema": lambda x: x.pop("schema_version"),
            "missing-peer": lambda x: x["summaries"].pop("B"),
            "wrong-member": lambda x: x["summaries"]["B"].update(member="A"),
            "empty-fingerprint": lambda x: x["summaries"]["B"]["binding"].update(fingerprint=""),
            "wrong-source": lambda x: x["summaries"]["B"]["binding"].update(source_revision="older"),
            "wrong-deps": lambda x: x["summaries"]["B"]["binding"].update(dependencies={"torch": "other"}),
            "wrong-python": lambda x: x["summaries"]["B"]["binding"].update(python="other"),
            "wrong-cuda": lambda x: x["summaries"]["B"]["binding"].update(torch_cuda="other"),
            "wrong-assets": lambda x: x["summaries"]["B"]["assets"].update(head_sha256="c"*64),
            "wrong-recipe": lambda x: x["summaries"]["B"]["recipe"].update(seed=29),
            "missing-profile": lambda x: x["summaries"]["B"]["profiles"].pop(x["profile"]),
            "peer-failed": lambda x: x["summaries"]["B"]["profiles"][x["profile"]].update(passed=False),
            "string-passed": lambda x: x["summaries"]["B"]["profiles"][x["profile"]].update(passed="false"),
            "negative-time": lambda x: x["summaries"]["B"]["profiles"][x["profile"]].update(estimated_epoch_seconds=-1),
            "infinite-time": lambda x: x["summaries"]["B"]["profiles"][x["profile"]].update(estimated_epoch_seconds=float("inf")),
            "nonpreferred-profile": lambda x: x.update(profile="fp32-m32-w4-e4"),
            "changed-ranking": lambda x: x.update(ranking=[]),
            "changed-digest": lambda x: x["summary_digests"].update(B="f"*64),
            "missing-raw-hash": lambda x: x["summary_hashes"].pop("B"),
            "wrong-status": lambda x: x.update(status="approved"),
            "eligible": lambda x: x.update(selection_eligible=True),
        }
        for name, mutate in cases.items():
            bad = copy.deepcopy(original)
            mutate(bad)
            write_json(path, bad)
            target = self.root / f"{name}.json"
            with self.subTest(name=name), patch.object(r, "_binding", return_value=binding()):
                with self.assertRaises(ValueError):
                    r.accept(bundle, path, self.root / "repeat", self.root / "A/checks.json", target)
                self.assertFalse(target.exists())

    def test_run_rejects_invalid_selection_even_after_receipt_hash_is_rewritten(self):
        bundle, path = joint_ready(self.root)
        selection = read_json(path)
        receipt_path = self.root / "receipt.json"
        with patch.object(r, "_binding", return_value=binding()):
            receipt = r.accept(bundle, path, self.root / "repeat", self.root / "A/checks.json", receipt_path)
            duplicate = copy.deepcopy(selection)
            duplicate["summaries"]["B"] = copy.deepcopy(duplicate["summaries"]["A"])
            duplicate["summaries"]["B"]["member"] = "B"
            duplicate["summary_digests"]["B"] = r.sha256_json(duplicate["summaries"]["B"])
            failed = copy.deepcopy(selection)
            failed["summaries"]["B"]["profiles"][failed["profile"]]["passed"] = False
            failed["summary_digests"]["B"] = r.sha256_json(failed["summaries"]["B"])
            for name, bad in (("duplicate", duplicate), ("failed-profile", failed)):
                write_json(path, bad)
                altered = copy.deepcopy(receipt)
                altered["selection_sha256"] = r.file_sha256(path)
                altered["hashes"][str(path)] = altered["selection_sha256"]
                write_json(receipt_path, altered)
                with self.subTest(name=name), patch("aic_robust_clip.workflow.train_command") as train:
                    with self.assertRaises(ValueError):
                        r.run(bundle, receipt_path, "control")
                    train.assert_not_called()

    def test_run_requires_explicit_selection_binding_and_matches_receipt_profile(self):
        bundle, path = joint_ready(self.root)
        receipt_path = self.root / "receipt.json"
        with patch.object(r, "_binding", return_value=binding()):
            receipt = r.accept(bundle, path, self.root / "repeat", self.root / "A/checks.json", receipt_path)
            for name, mutate in {
                "legacy": lambda x: x.pop("schema_version"),
                "missing-path": lambda x: x.pop("selection_path"),
                "relative-path": lambda x: x.update(selection_path="selection.json"),
                "missing-digest": lambda x: x.pop("selection_sha256"),
                "missing-evidence": lambda x: x["hashes"].pop(str(path)),
                "other-profile": lambda x: x.update(profile="fp32-m32-w4-e4"),
            }.items():
                altered = copy.deepcopy(receipt)
                mutate(altered)
                write_json(receipt_path, altered)
                with self.subTest(name=name), patch("aic_robust_clip.workflow.train_command") as train:
                    with self.assertRaises(ValueError):
                        r.run(bundle, receipt_path, "control")
                    train.assert_not_called()
            write_json(receipt_path, receipt)
            selection = read_json(path)
            selection["status"] = "changed"
            write_json(path, selection)
            with patch("aic_robust_clip.workflow.train_command") as train:
                with self.assertRaisesRegex(ValueError, "evidence changed"):
                    r.run(bundle, receipt_path, "control")
                train.assert_not_called()

    def test_valid_joint_selection_is_portable_to_b_without_reading_a_host_paths(self):
        _, path = joint_ready(self.root)
        bundle = self.root / "B/bundle/bundle.json"
        selected = read_json(path)
        cfg = load_config(read_json(bundle)["configs"][selected["profile"]]["control"])
        for i in range(1, 4):
            write_json(self.root / f"repeat-b/eval-{i}/benchmark.json", report(cfg, "eval", binding("host-B")))
        # The peer's original summary file is on another machine, not a prerequisite here.
        (self.root / "A/summary.json").unlink()
        with patch.object(r, "_binding", return_value=binding("host-B")):
            receipt = self.root / "receipt-b.json"
            r.accept(bundle, path, self.root / "repeat-b", self.root / "B/checks.json", receipt)
            with patch("aic_robust_clip.workflow.train_command") as train:
                r.run(bundle, receipt, "control")
                train.assert_called_once()

    def test_failed_candidates_are_retained_and_not_selected(self):
        a = measured(self.root, "A", binding())
        name = next(iter(r.PROFILES))
        path = self.root / "windows" / name / "eval/benchmark.json"
        original = path.read_bytes()
        write_json(path.parent / "failure.json", {"error": "synthetic failure"})
        with patch.object(r, "_binding", return_value=binding()):
            summary = r.summarize(a, self.root / "windows", self.root / "checks.json", self.root / "summary-new.json")
        self.assertFalse(summary["profiles"][name]["passed"])
        self.assertEqual(path.read_bytes(), original)

    def test_remote_check_records_original_regression_and_bounded_startups(self):
        from types import SimpleNamespace
        fixture(self.root)
        passed = "test_verification_cached_only_for_unchanged_file_and_process (fixture) ... ok\nRan 1 test in 0.1s\n\nOK\n"
        with patch.object(r, "_binding", return_value=binding()), \
                patch.object(r, "machine_policy", return_value=SimpleNamespace(allow_formal=True)), \
                patch.object(r.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr=passed)), \
                patch("aic_robust_clip.training.model_smoke.check_official_model", return_value={
                    "result": {"updates": 2, "optimizer_updated": True, "stopped_by_limit": True}}) as startup:
            receipt = r.check(self.root / "source.json", self.root / "check")
            self.assertEqual(receipt["status"], "passed_zero_skips")
            self.assertEqual(startup.call_count, 2)
            self.assertEqual([call.kwargs for call in startup.call_args_list],
                [{"device": "cuda", "precision": "fp32"}, {"device": "cuda", "precision": "fp16"}])
        for index, stderr in enumerate((passed.replace("OK", "OK (skipped=1)"),
                                       passed.replace("test_verification_cached_only_for_unchanged_file_and_process", "unrelated"))):
            with patch.object(r, "_binding", return_value=binding()), \
                    patch.object(r, "machine_policy", return_value=SimpleNamespace(allow_formal=True)), \
                    patch.object(r.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr=stderr)), \
                    patch("aic_robust_clip.training.model_smoke.check_official_model", side_effect=AssertionError("started")):
                target = self.root / f"bad-check-{index}"
                with self.assertRaises(ValueError):
                    r.check(self.root / "source.json", target)
                self.assertTrue((target / "failure.json").exists())
                self.assertFalse((target / "checks.json").exists())

    def test_check_rejects_local_host_before_running_commands(self):
        fixture(self.root)
        with patch.object(r, "_binding", return_value=binding()), \
                patch.object(r, "machine_policy", return_value=type("Policy", (), {"allow_formal": False})()), \
                patch.object(r.subprocess, "run", side_effect=AssertionError("executed")):
            with self.assertRaisesRegex(ValueError, "separate 4060"):
                r.check(self.root / "source.json", self.root / "check")
        self.assertFalse((self.root / "check").exists())


if __name__ == "__main__":
    unittest.main()
