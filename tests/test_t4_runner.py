"""Offline orchestration fixtures: no SSH, CUDA, weights or real archives."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from aic_robust_clip import t4_runner as runner
from aic_robust_clip.contracts import read_json, write_json
from aic_robust_clip.performance import PerformanceConfig


def source(root):
    config = {"schema_version": 2, "recipe": "B03", "stage": "preliminary", "execution_mode": "formal",
              "device": "cuda", "batch_size": 1, "manifest": "manifest.json", "split": "split.json",
              "head": "head", "output": "old-run", "parameters": {"formal_epochs": 10}, "machine_config": "host.json"}
    path = root / "source.json"
    write_json(path, config)
    runner.prepare_bundle(path, root / "bundle")
    return root / "bundle/bundle.json"


class T4Tests(unittest.TestCase):
    def test_lineage_uses_train_frequency_and_exports_dev_groups_only(self):
        rows = [SimpleNamespace(partition=p, class_id=c, sample_id=s, group_id="g-" + s)
                for p, c, s in (("train", "0000", "a"), ("train", "0000", "b"),
                                ("train", "0001", "c"), ("dev", "0001", "d"),
                                ("confirm", "0001", "e"))]
        ctx = SimpleNamespace(run=SimpleNamespace(stage="preliminary", manifest_digest="a", split_digest="b", class_map_digest="c"),
                              class_map=SimpleNamespace(id_to_index={"0000": 0, "0001": 1}), split=SimpleNamespace(records=rows))
        report = runner.lineage_report(ctx)
        self.assertEqual(report["train_counts"], {0: 2, 1: 1})
        self.assertEqual(report["dev_groups"], {"d": "g-d"})
        self.assertEqual(report["images_read"], 0)

    def test_bundle_reuses_assets_isolated_configs_and_no_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = read_json(source(root))
            self.assertEqual(len(bundle["configs"]), 6)
            outputs = []
            for profile, paths in bundle["configs"].items():
                for job, path in paths.items():
                    config = read_json(path)
                    self.assertEqual(config["recipe"], runner.JOBS[job])
                    self.assertEqual(config["head"], str(root / "head"))
                    self.assertEqual(config["parameters"]["formal_epochs"], 10)
                    self.assertEqual(config["effective_batch_size"], 128)
                    perf = PerformanceConfig.from_config(config)
                    self.assertEqual(perf.scoring_batch_size, 128)
                    self.assertEqual(perf.scoring_num_workers, perf.eval_num_workers)
                    outputs.append(config["output"])
                    self.assertFalse(Path(config["output"]).exists())
            self.assertEqual(len(outputs), len(set(outputs)))
            self.assertFalse((root / "old-run").exists())
            with self.assertRaises(FileExistsError):
                runner.prepare_bundle(root / "source.json", root / "bundle")

    def test_scoring_defaults_and_smoke_boundary(self):
        self.assertEqual(PerformanceConfig.from_config({"batch_size": 16, "execution_mode": "formal",
            "performance": {"eval_batch_size": 64, "eval_num_workers": 2}}).scoring_batch_size, 64)
        for field, value in (("scoring_num_workers", True), ("scoring_batch_size", 0)):
            with self.assertRaises(ValueError):
                PerformanceConfig.from_config({"performance": {field: value}})
        with self.assertRaisesRegex(ValueError, "smoke"):
            PerformanceConfig.from_config({"performance": {"scoring_batch_size": 2}})

    def test_inventory_rejects_busy_wrong_or_duplicate_devices(self):
        inventory = "\n".join(f"{i}, GPU-{i}, Tesla T4" for i in range(4))
        with patch.object(runner.subprocess, "run", side_effect=[SimpleNamespace(stdout=inventory), SimpleNamespace(stdout="")]):
            self.assertEqual(len(runner.gpu_inventory(["0", "1", "2", "3"])), 4)
        with self.assertRaises(ValueError):
            runner.gpu_inventory(["0", "0", "2", "3"])
        with patch.object(runner.subprocess, "run", side_effect=[SimpleNamespace(stdout=inventory), SimpleNamespace(stdout="GPU-1, 123")]):
            with self.assertRaisesRegex(ValueError, "existing compute"):
                runner.gpu_inventory(["0", "1", "2", "3"])
        with patch.object(runner.subprocess, "run", return_value=SimpleNamespace(stdout=inventory.replace("Tesla T4", "RTX 4060"))):
            with self.assertRaisesRegex(ValueError, "four T4s"):
                runner.gpu_inventory(["0", "1", "2", "3"])

    def test_local_refusal_precedes_gpu_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = source(root)
            with patch.object(runner, "machine_policy", return_value=SimpleNamespace(allow_formal=False)), \
                    patch.object(runner, "gpu_inventory", side_effect=AssertionError("GPU touched")):
                with self.assertRaisesRegex(ValueError, "enrollment"):
                    runner.dispatch(bundle, "fp32-m16-e4", root / "dispatch", ["0", "1", "2", "3"], mode="bench")
            self.assertFalse((root / "dispatch").exists())

    def test_worker_phases_include_three_eval_and_three_w_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = read_json(source(root))
            for job in runner.JOBS:
                with patch("aic_robust_clip.benchmark.benchmark_command") as benchmark:
                    runner.benchmark_worker(bundle["configs"]["fp32-m16-e4"][job], root / job)
                    phases = [call.args[1] for call in benchmark.call_args_list]
                    self.assertEqual(phases, ["train"] + ["eval"] * 3 + (["scoring"] * 3 if job == "N02" else []))
                    self.assertTrue(all(c.kwargs == {"warmup_steps": 2, "measure_steps": 10} for c in benchmark.call_args_list))

    def test_acceptance_reports_hashes_bounds_and_speed_gate(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, "machine_policy", return_value=SimpleNamespace(allow_formal=True)), \
                patch.object(runner, "machine_fingerprint", return_value="host"), \
                patch.object(runner, "dependencies", return_value={"torch": "fixture"}), \
                patch.object(runner, "current_code_revision", return_value="source"):
            root = Path(directory)
            bundle = source(root)
            selected = "fp32-m16-e4"

            def reports(profile, speed):
                output = root / profile
                write_json(output / "dispatch.json", {"status": "complete", "mode": "bench", "profile": profile, "concurrent": True})
                _, configs = runner.profile_configs(bundle, profile)
                for job, cfg in configs.items():
                    for phase, repeats in {"train": 1, "eval": 3, **({"scoring": 3} if job == "N02" else {})}.items():
                        batch = 128 if phase in {"train", "scoring"} else 64
                        for repeat in range(repeats):
                            write_json(output / job / f"{phase}-{repeat}" / "benchmark.json", {
                                "kind": "benchmark_only", "selection_eligible": False, "configuration": {
                                    "config": cfg, "split_counts": {"train": 82586, "dev": 10378}},
                                "phase": phase, "source_revision": "source", "measured_steps": 10,
                                "warmup_steps": 2, "measured_samples": 10 * batch, "prefetch_extra_samples_upper_bound": 0,
                                "environment": {"fingerprint": "host", "gpu": {"memory_bytes": 1000}, "dependencies": {"torch": "fixture"}},
                                "peak_gpu_reserved_bytes": 100, "samples_per_second": speed,
                                "checkpoint_write_seconds": 1, "loader_after_cleanup": {
                                    "delivered_samples": 12 * batch, "workers": [], "worker_exits": [], "worker_failed": False}})
                return output, configs

            slow, _ = reports("fp32-m16-e0", 20)
            fast, configs = reports(selected, 100)
            receipt = root / "accepted.json"
            result = runner.accept(bundle, selected, fast, [slow], receipt, "synthetic test only")
            self.assertEqual(len(result["report_hashes"]), 19)
            runner.validate_acceptance(receipt, selected, configs)
            with self.assertRaisesRegex(ValueError, "slower"):
                runner.accept(bundle, "fp32-m16-e0", slow, [fast], root / "bad.json", "test")
            path = fast / "N02/scoring-2/benchmark.json"
            changed = read_json(path)
            changed["loader_after_cleanup"]["worker_exits"] = [{"exitcode": -6, "alive": False}]
            write_json(path, changed)
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                runner.validate_acceptance(receipt, selected, configs)
            with self.assertRaisesRegex(ValueError, "invalid/stale"):
                runner.benchmark_summary(fast, configs)

    def test_formal_missing_acceptance_and_invalid_subset_fail_before_gpu(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, "machine_policy", return_value=SimpleNamespace(allow_formal=True)), \
                patch.object(runner, "gpu_inventory", side_effect=AssertionError("GPU touched")):
            root = Path(directory)
            bundle = source(root)
            with self.assertRaisesRegex(ValueError, "acceptance"):
                runner.dispatch(bundle, "fp32-m16-e4", root / "out", ["0", "1", "2", "3"], mode="train")
            with self.assertRaisesRegex(ValueError, "all four"):
                runner.dispatch(bundle, "fp32-m16-e4", root / "out", ["0"], mode="bench", jobs=["N00"])
            self.assertFalse((root / "out").exists())

    def test_dispatch_four_distinct_masks_offline_and_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(runner, "machine_policy", return_value=SimpleNamespace(allow_formal=True)):
            root = Path(directory)
            bundle = source(root)
            configs = read_json(bundle)["configs"]["fp32-m16-e4"]
            calls = []
            def fake_prepare(path):
                value = read_json(path)
                run = SimpleNamespace(stage="preliminary", manifest_digest="a", split_digest="b", class_map_digest="c",
                                      execution_mode="formal", batch_size=16)
                return SimpleNamespace(config=value, run=run, weights={"digest": "w"}, preprocessing_digest="p",
                    train=SimpleNamespace(precision="fp32", accumulation_steps=8, epochs=10),
                    policy=SimpleNamespace(allow_formal=True))
            def fake_child(command, **kwargs):
                calls.append((command, kwargs["env"]))
                return SimpleNamespace(wait=lambda **kw: 1 if configs["N02"] in command else 0)
            with patch.object(runner, "gpu_inventory", return_value=[{"uuid": f"GPU-{i}", "name": "Tesla T4"} for i in range(4)]), \
                    patch.object(runner, "prepare", side_effect=fake_prepare), \
                    patch.object(runner, "lineage_report", return_value={"images_read": 0}), \
                    patch("aic_robust_clip.workflow.load_head", return_value=({}, "head")), \
                    patch.object(runner.subprocess, "Popen", side_effect=fake_child), \
                    patch.object(runner, "__file__", str(root / "src/pkg/t4_runner.py")):
                with self.assertRaisesRegex(RuntimeError, "one job failed"):
                    runner.dispatch(bundle, "fp32-m16-e4", root / "dispatch", ["0", "1", "2", "3"], mode="bench")
            self.assertEqual(len(calls), 4)
            self.assertEqual({env["CUDA_VISIBLE_DEVICES"] for _, env in calls}, {f"GPU-{i}" for i in range(4)})
            self.assertTrue(all(env["HF_HUB_OFFLINE"] == "1" and env["OMP_NUM_THREADS"] == "2" for _, env in calls))
            self.assertEqual(read_json(root / "dispatch/dispatch.json")["status"], "failed")


if __name__ == "__main__":
    unittest.main()
