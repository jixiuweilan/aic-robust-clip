"""Preliminary orchestration correctness; mocks/generated images, no real assets."""
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import zipfile

import numpy as np
import torch

from aic_robust_clip import preliminary_experiments as exp, preliminary_pilot as pilot
from aic_robust_clip.contracts import read_json, write_json
from aic_robust_clip.round2 import admission, engine
from aic_robust_clip.round2.config import sealed, RECIPE
from aic_robust_clip.round2.methods import MethodError
from test_preliminary_pilot import context


RUNTIME = {"host": "fixture-host", "source": "fixture-source", "gpu": {"name": "Tesla T4", "uuid": "GPU-fixture"}}


def profile(method, assets, runtime=RUNTIME, settings=None, windows=3):
    return sealed({"kind": "profile", "status": "passed", "stage": "preliminary", "method": method,
        "runtime": runtime, "adaptation": "lora", "zero_selection_policy": exp.policy(method),
        "asset_digest": assets.descriptor["digest"], "head_sha256": assets.head_sha256,
        **(settings or exp.FIXED), "eval_seconds": [.1] * windows, "warmup": 2, "measured": 10,
        "workers_closed": True, "dev_student_replay": True, "peak_occupied_bytes": 60, "total_bytes": 100,
        "projected_epoch_seconds": 10., "train_window_started": 1., "train_window_ended": 2.})


def checks(method, runtime=RUNTIME):
    return sealed({"runtime": runtime, "suite": {"tests": 1, "failures": 0, "errors": 0, "skips": 0},
                   "startup": [{"method": method, "precision": p, "adaptation": "lora", "device": "cuda",
                                "zero_selection_policy": exp.policy(method), "updates": 2, "status": "passed"}
                               for p in ("fp32", "fp16")]})


class PreliminaryExperimentsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "preliminary"
        self.root.mkdir()
        self.assets = pilot.PilotAssets(context(), "head")
        self.source = self.root / "source.json"
        write_json(self.source, {"schema_version": 2, "recipe": "B03", "stage": "preliminary", "machine_config": "fixture"})

    def receipt(self, method, root=None):
        root = root or self.root
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "profile.json", profile(method, self.assets))
        write_json(root / "checks.json", checks(method))
        write_json(root / "provenance.json", sealed({"assets": self.assets.descriptor, "runtime": RUNTIME}))
        value = sealed({"version": exp.VERSION, "kind": "t4_admission", "status": "passed", "stage": "preliminary",
                        "methods": [method], "engineering": exp.FIXED,
                        "source_configuration": str(self.source), "source_sha256": exp.file_sha256(self.source),
                        "entries": {method: {k: exp.evidence(root / f"{k}.json") for k in ("profile", "checks", "provenance")}}})
        write_json(root / "admission.json", value)
        return root / "admission.json"

    def test_prepare_is_metadata_only_and_preserves_round2_and_turn_defaults(self):
        before = copy.deepcopy(RECIPE)
        with patch.object(pilot, "prepare", side_effect=AssertionError("real assets")), \
             patch.object(pilot, "load_head", side_effect=AssertionError("head fitting/loading")), \
             patch.object(admission, "_profile_one", side_effect=AssertionError("profile")):
            rows = exp.plan(self.root / "plans")
        self.assertEqual({r["method"] for r in rows}, set(exp.METHODS))
        for row in rows:
            self.assertEqual(row["status"], "blocked_on_machine_admission")
            self.assertEqual((row["recipe"]["epochs"], row["recipe"]["pause_after_epoch"]), (30, 10))
            self.assertEqual(row["recipe"]["zero_selection_policy"], "retain_observed" if row["method"] == "turn" else "error")
        self.assertEqual(RECIPE, before)
        self.assertEqual(pilot.VERSION, "preliminary-turn-abstain-pilot-v2")

    def test_source_guard_precedes_asset_access_and_requires_one_gpu(self):
        with patch.object(pilot, "prepare", side_effect=AssertionError("assets opened")), \
             patch.object(pilot, "load_config", return_value={"stage": "second_round"}):
            with self.assertRaisesRegex(ValueError, "preliminary"):
                exp.source_context("unused", family="T4")
        with patch.object(pilot, "prepare", side_effect=AssertionError("assets opened")), \
             patch.object(engine, "require_machine"), patch.object(admission, "runtime_identity", return_value=RUNTIME), \
             patch.object(torch.cuda, "device_count", return_value=2):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                exp.source_context(self.source, family="T4")

    def test_synthetic_startup_all_methods_never_enters_real_flow(self):
        with patch.object(exp, "source_context", side_effect=AssertionError("real assets")), \
             patch.object(exp, "run", side_effect=AssertionError("formal run")):
            for method in exp.METHODS:
                self.assertEqual(exp.main(["startup-check", "--method", method, "--device", "cpu"]), 0)

    def test_profile_evidence_rejects_method_policy_precision_and_headroom(self):
        valid = profile("fine", self.assets)
        exp.check_profile(valid, method="fine", assets=self.assets, runtime=RUNTIME, engineering=exp.FIXED, windows=3)
        for key, value in (("method", "turn"), ("zero_selection_policy", "retain_observed"),
                           ("stage", "second_round"), ("peak_occupied_bytes", 86), ("workers_closed", False),
                           ("eval_seconds", [float("nan")] * 3)):
            bad = sealed({k: v for k, v in {**valid, key: value}.items() if k != "digest"})
            with self.assertRaises(ValueError):
                exp.check_profile(bad, method="fine", assets=self.assets, runtime=RUNTIME, engineering=exp.FIXED, windows=3)
        for kind in ("wrong-method", "missing-fp16", "skipped"):
            value = checks("ce")
            if kind == "wrong-method": value["startup"][0]["method"] = "turn"
            if kind == "missing-fp16": value["startup"].pop()
            if kind == "skipped": value["suite"]["skips"] = 1
            with self.assertRaises(ValueError):
                exp.check_software(sealed({k: v for k, v in value.items() if k != "digest"}), "ce", RUNTIME)

    def test_profile_passes_method_and_stops_on_method_failure(self):
        for method in ("ce", "fine", "snscl"):
            with patch.object(exp, "source_context", return_value=({"machine_config": "machine"}, self.assets, RUNTIME)), \
                 patch.object(pilot, "software_checks") as software, \
                 patch.object(admission, "_profile_one", return_value=profile(method, self.assets)) as numerical:
                exp.run_profile(self.source, self.root / method, method=method)
                self.assertEqual(software.call_args.kwargs["method"], method)
                self.assertEqual(numerical.call_args.kwargs["_zero_selection_policy"], "error")
                self.assertEqual(numerical.call_args.kwargs["eval_windows"], 3)
        with patch.object(exp, "source_context", return_value=({"machine_config": "machine"}, self.assets, RUNTIME)), \
             patch.object(pilot, "software_checks"), \
             patch.object(admission, "_profile_one", side_effect=MethodError("zero selected")) as numerical:
            with self.assertRaises(MethodError):
                exp.run_profile(self.source, self.root / "fail", method="fine")
            self.assertEqual(numerical.call_count, 1)
            self.assertTrue((self.root / "fail/failure.json").is_file())

    def test_run_rejects_old_4060_wrong_gpu_and_changed_evidence_before_training(self):
        receipt = self.receipt("ce")
        with patch.object(engine, "_run_prepared", side_effect=AssertionError("formal run")), \
             patch.object(exp, "source_context", return_value=({}, self.assets, {**RUNTIME, "gpu": {"uuid": "different"}})):
            with self.assertRaisesRegex(ValueError, "checks"):
                exp.run(receipt, self.root / "run", method="ce")
        value = read_json(receipt)
        value["kind"] = "4060_admission_only"
        write_json(receipt, sealed({k: v for k, v in value.items() if k != "digest"}))
        with self.assertRaisesRegex(ValueError, "T4 admission"):
            exp.run(receipt, self.root / "run", method="ce")
        receipt = self.receipt("ce")
        write_json(self.root / "profile.json", {"changed": True})
        with patch.object(exp, "source_context", return_value=({}, self.assets, RUNTIME)):
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                exp.run(receipt, self.root / "run", method="ce")

    def test_concurrency_failing_child_stops_without_go_or_training(self):
        process = SimpleNamespace(pid=123, returncode=1, poll=lambda: 1)
        with patch.object(engine, "require_machine"), patch.object(exp.subprocess, "Popen", return_value=process), \
             patch.object(engine, "_run_prepared", side_effect=AssertionError("formal run")):
            with self.assertRaisesRegex(ValueError, "before shared start"):
                exp.admit_t4(self.source, self.root / "admit-fail", methods=["ce"], gpu_uuids=["GPU-one"], owner="operator")
        self.assertFalse((self.root / "admit-fail/barrier/go.json").exists())
        self.assertTrue((self.root / "admit-fail/failure.json").exists())

    def test_4060_history_grid_and_admission_never_launch_training(self):
        runtime = {**RUNTIME, "gpu": {"name": "RTX 4060", "uuid": "GPU-fixture"}}
        history = {"host": runtime["host"], "gpu_uuid": "GPU-fixture", "original_b04_failure": False,
                   "reviewer": "operator", "basis": "fixture inventory"}
        for change in ({"host": "other"}, {"original_b04_failure": None}, {"original_b04_failure": True}):
            with self.assertRaises(ValueError):
                exp.check_history({**history, **change}, None, runtime)
        write_json(self.root / "history.json", history)
        calls = []
        def software(root, *, method):
            value = checks(method, runtime)
            write_json(root / "checks.json", value)
            return value
        def numerical(*args, **kwargs):
            calls.append((kwargs["method"], kwargs["microbatch"], kwargs["workers"], kwargs.get("eval_windows", 1)))
            target = kwargs["output"]
            value = profile(kwargs["method"], self.assets, runtime,
                            {k: kwargs[k] for k in ("microbatch", "workers")}, kwargs.get("eval_windows", 1))
            write_json(target / "profile.json", value)
            return value
        with patch.object(exp, "source_context", return_value=({"machine_config": "machine"}, self.assets, runtime)), \
             patch.object(pilot, "software_checks", side_effect=software), \
             patch.object(admission, "_profile_one", side_effect=numerical), \
             patch.object(engine, "_run_prepared", side_effect=AssertionError("formal run")):
            value = exp.admit_4060(self.source, self.root / "4060", candidate="fine", owner="operator",
                                   history_path=self.root / "history.json")
        self.assertEqual(len(calls), 18)
        self.assertEqual(value["engineering"], {"microbatch": 4, "workers": 2})
        self.assertFalse(value["training_authorized_by_receipt"])
        self.assertEqual(calls[-2:], [("ce", 4, 2, 3), ("fine", 4, 2, 3)])

    def test_concurrent_success_binds_gpu_assets_and_distinct_methods(self):
        root = self.root / "concurrent"
        def child(command, **kwargs):
            method = command[command.index("--method") + 1]
            target = Path(command[command.index("--output") + 1])
            runtime = {**RUNTIME, "gpu": {"name": "Tesla T4", "uuid": kwargs["env"]["CUDA_VISIBLE_DEVICES"]}}
            write_json(target / "profile/profile.json", profile(method, self.assets, runtime))
            write_json(target / "checks.json", checks(method, runtime))
            write_json(target / "provenance.json", sealed({"assets": self.assets.descriptor, "runtime": runtime,
                       "source_sha256": exp.file_sha256(self.source)}))
            write_json(root / "barrier" / f"{method}.ready.json", {"uuid": runtime["gpu"]["uuid"]})
            return SimpleNamespace(pid=123, returncode=0, poll=lambda: 0)
        with patch.object(engine, "require_machine"), patch.object(exp.subprocess, "Popen", side_effect=child):
            result = exp.admit_t4(self.source, root, methods=["ce", "fine"], gpu_uuids=["GPU-one", "GPU-two"], owner="operator")
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["formal_training_started"])
        self.assertEqual(result["overlap_seconds"], 1.)
        for entry in result["entries"].values():
            for item in entry.values(): exp.checked_evidence(item)

    def test_comparison_rejects_unmatched_source_budget_and_corrupt_checkpoint(self):
        roots = [self.root / m for m in ("ce", "fine")]
        for index, (method, root) in enumerate(zip(("ce", "fine"), roots)):
            (root / "run").mkdir(parents=True)
            for name in ("last.pt", "best.pt"):
                (root / "run" / name).write_bytes(b"synthetic checkpoint bytes")
            cfg = sealed({"version": exp.VERSION, "stage": "preliminary", "method": method,
                          "recipe": exp.recipe(method), "asset_digest": "assets", "head_sha256": "head",
                          "engineering": exp.FIXED, "adaptation": "lora",
                          "comparison_digest": "same", "runtime": RUNTIME, "receipt_digest": str(index)})
            write_json(root / "config.json", cfg)
            write_json(root / "run/result.json", {"status": "paused_at_epoch10", "completed_epochs": 10, "full_epochs": 30,
                "checkpoint_sha256": exp.file_sha256(root / "run/last.pt"), "best_sha256": exp.file_sha256(root / "run/best.pt"),
                "best_metrics": {"macro_recall": .5 + .1 * index}, "last": {"macro_recall": .4 + .1 * index}})
            (root / "replay-dev").mkdir()
            (root / "replay-dev/student.pt").write_bytes(b"synthetic student")
            write_json(root / "replay-dev/replay.json", {"passed": True, "partition": "dev", "epoch": 10,
                "identity": {"checkpoint_sha256": exp.file_sha256(root / "run/last.pt")},
                "student_sha256": exp.file_sha256(root / "replay-dev/student.pt")})
        self.assertAlmostEqual(exp.compare(*roots)["best_macro_delta"], .1)
        cfg = read_json(roots[1] / "config.json")
        cfg["runtime"] = {**RUNTIME, "source": "old"}
        write_json(roots[1] / "config.json", sealed({k: v for k, v in cfg.items() if k != "digest"}))
        with self.assertRaisesRegex(ValueError, "matched"):
            exp.compare(*roots)
        (roots[0] / "run/last.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoints"):
            exp.compare(*roots)

    def test_three_methods_tiny_loader_epoch10_pause_and_student_replay(self):
        # Small generated fixture under a finite 10-update-style loop; no real data/scores.
        from PIL import Image
        from transformers import CLIPImageProcessor
        from aic_robust_clip.data.audit import audit_archive, class_map_from_records
        from aic_robust_clip.data.splits import make_grouped_split
        with zipfile.ZipFile(self.root / "train.zip", "w") as archive:
            for index in range(18):
                data = io.BytesIO()
                Image.new("RGB", (index + 10, 12), (index * 10, 50, 200)).save(data, format="PNG")
                archive.writestr(f"{index % 3:04}/{index}.png", data.getvalue())
        audit = audit_archive(self.root / "train.zip", stage="preliminary", role="train")
        ctx = context()
        ctx.records, ctx.split, ctx.class_map = audit.records, make_grouped_split(audit.records), class_map_from_records(audit.records)
        self.assets = pilot.PilotAssets(ctx, "head")
        trainer_type = engine.Trainer
        initial = admission.tiny_student("full_visual", image_size=224)
        def factory(assets, adaptation):
            from aic_robust_clip.round2.model import Student
            return Student(copy.deepcopy(initial.encoder), 3, adaptation), CLIPImageProcessor(), {"sha256": "head"}
        def cpu_trainer(*args, **kwargs):
            kwargs.update(device="cpu", precision="fp32")
            return trainer_type(*args, **kwargs)
        def synthetic_reliability(scores, **kwargs):
            return np.linspace(.1, .9, len(scores))
        replay_impl = engine._replay_prepared
        def cpu_replay(*args, **kwargs):
            return replay_impl(*args, **kwargs, device="cpu")
        with patch.object(exp, "FIXED", {"microbatch": 4, "workers": 0}), \
             patch.object(exp, "source_context", return_value=({}, self.assets, RUNTIME)), \
             patch.object(pilot, "student_factory", side_effect=factory), \
             patch.object(engine, "Trainer", side_effect=cpu_trainer), \
             patch.object(engine, "_replay_prepared", side_effect=cpu_replay), \
             patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(torch.cuda, "max_memory_reserved", return_value=0), \
             patch("aic_robust_clip.round2.methods.gmm", side_effect=synthetic_reliability):
            for method in ("ce", "fine", "snscl"):
                receipt = self.receipt(method, self.root / f"evidence-{method}")
                target = self.root / f"run-{method}"
                result = exp.run(receipt, target, method=method)
                self.assertEqual(result["result"]["completed_epochs"], 10)
                self.assertTrue(result["replay"]["passed"])
                self.assertEqual(read_json(target / "config.json")["recipe"]["epochs"], 30)
                saved = torch.load(target / "run/last.pt", weights_only=False)
                self.assertEqual(saved["method"]["method"], method)
                if method == "snscl":
                    self.assertEqual(read_json(target / "run/epoch-04.json")["scoring_samples"], 0)
                    self.assertGreater(read_json(target / "run/epoch-05.json")["scoring_samples"], 0)
                    self.assertGreater(sum(read_json(target / "run/epoch-06.json")["queue_counts"]), 0)
                with self.assertRaisesRegex(ValueError, "epoch10"):
                    exp.run(receipt, target, method=method, resume=True)


if __name__ == "__main__":
    unittest.main()
