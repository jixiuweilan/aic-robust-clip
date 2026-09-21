"""Stage-isolated night pilot tests; generated data and mocks only."""
import copy
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch

from aic_robust_clip import preliminary_pilot as pilot
from aic_robust_clip.contracts import read_json, write_json
from aic_robust_clip.round2 import config, engine, admission
from aic_robust_clip.round2.methods import MethodError


def context():
    return SimpleNamespace(run=SimpleNamespace(stage="preliminary", execution_mode="formal", seed=17,
        manifest_digest="manifest", split_digest="split", class_map_digest="classes"),
        config={"recipe": "B03", "effective_batch_size": 128},
        train=SimpleNamespace(objective="ce", weighting=False, lambda_preserve=0., prior_tau=0.),
        class_map=SimpleNamespace(id_to_index={"0000": 0, "0001": 1, "0002": 2}),
        weights={"digest": "official"}, preprocessing_digest="fixed-transform", summary=lambda: {"fixture": True})


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_stage_and_purpose_do_not_relax_round2_guards(self):
        ctx = context()
        assets = pilot.PilotAssets(ctx, "head3")
        self.assertEqual(assets.descriptor["stage"], "preliminary")
        self.assertEqual(assets.descriptor["initializer"], "verified_preliminary_HEAD3")
        for stage in ("second_round", "semifinal"):
            ctx.run.stage = stage
            with self.assertRaises(ValueError):
                pilot.assert_source(ctx)
        for partition in ("confirm", "test"):
            with self.assertRaises(ValueError):
                assets.dataset(partition)
        with tempfile.TemporaryDirectory() as directory:
            preliminary = Path(directory) / "preliminary/run"
            with self.assertRaises(ValueError):
                config.stage_path(preliminary)
            self.assertEqual(config.stage_path(preliminary, stage="preliminary"), preliminary)
            with self.assertRaises(ValueError):
                config.stage_path(Path(directory) / "second_round/run", stage="preliminary")

    def test_source_rejects_old_research_candidate_and_local_execution(self):
        ctx = context()
        ctx.train.weighting = True
        with self.assertRaises(ValueError):
            pilot.assert_source(ctx)
        with patch.object(pilot, "load_config", return_value={"stage": "preliminary"}), \
             patch.object(pilot, "prepare", side_effect=AssertionError("opened training assets")):
            with self.assertRaises(Exception) as caught:
                pilot.run_night("missing", "outputs/preliminary/not-created")
            self.assertNotIsInstance(caught.exception, AssertionError)
        with patch.object(pilot, "load_config", return_value={"stage": "second_round"}):
            with self.assertRaisesRegex(ValueError, "cannot consume"):
                pilot.run_night("missing", "outputs/preliminary/not-created")

    def test_no_silent_scoring_or_dev_stage_alias(self):
        ds = SimpleNamespace(stage="preliminary", role="train", partition="train", purpose="scoring")
        with self.assertRaises(MethodError):
            engine.scoring(None, SimpleNamespace(dataset=ds), device="cpu", features_required=False)
        ds.partition = "dev"
        with self.assertRaises(MethodError):
            engine.evaluate(None, SimpleNamespace(dataset=ds), device="cpu", classes=3, training_counts={})

    def test_fixed_recipe_does_not_mutate_round2_or_claim_matched_control(self):
        before = copy.deepcopy(config.RECIPE)
        assets = pilot.PilotAssets(context(), "head3")
        value = pilot.pilot_config(Path("/tmp/preliminary/test"), assets, {}, {"digest": "checks"}, {"digest": "profile"})
        self.assertEqual(value["recipe"]["epochs"], 30)
        self.assertEqual(value["recipe"]["pause_after_epoch"], 10)
        self.assertEqual(value["recipe"]["stage"], "preliminary")
        self.assertEqual(value["recipe"]["zero_selection_policy"], "retain_observed")
        self.assertEqual(value["recipe"]["method_variant"], "TURN-ABSTAIN-B32-v1")
        self.assertEqual(value["engineering"], {"microbatch": 32, "workers": 2})
        self.assertFalse(value["round2_eligible"])
        self.assertFalse(value["strict_matched_ce_control"])
        self.assertEqual(config.RECIPE, before)

    def test_startup_is_bounded_and_never_enters_formal_flow(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(pilot, "run_night", side_effect=AssertionError("formal flow")), \
             patch.object(pilot, "load_config", side_effect=AssertionError("real asset input")):
            root = Path(directory) / "preliminary/startup"
            self.assertEqual(pilot.main(["--startup-check", "--device", "cpu", "--output", str(root)]), 0)
            result = read_json(root / "startup.json")
            self.assertEqual((result["status"], result["updates"], result["samples"]), ("startup_only", 2, 4))

    def test_preliminary_turn_real_loader_pause_and_export_replay(self):
        # Generated images and a random tiny CLIP only: no competition assets.
        from PIL import Image
        from transformers import CLIPImageProcessor
        from aic_robust_clip.data.audit import audit_archive, class_map_from_records
        from aic_robust_clip.data.splits import make_grouped_split
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "preliminary"
            root.mkdir()
            with zipfile.ZipFile(root / "train.zip", "w") as archive:
                for index in range(18):
                    data = io.BytesIO()
                    Image.new("RGB", (index + 10, 12), (index * 10, 50, 200)).save(data, format="PNG")
                    archive.writestr(f"{index % 3:04}/{index}.png", data.getvalue())
            report = audit_archive(root / "train.zip", stage="preliminary", role="train")
            ctx = context()
            ctx.records = report.records
            ctx.split = make_grouped_split(report.records)
            ctx.class_map = class_map_from_records(report.records)
            assets = pilot.PilotAssets(ctx, "head3")
            initial = admission.tiny_student("full_visual", image_size=224)
            head = copy.deepcopy(initial.classifier.state_dict())
            def bundle(_):
                return SimpleNamespace(encoder=copy.deepcopy(initial.encoder), processor=CLIPImageProcessor())
            cfg = pilot.pilot_config(root, assets, {}, {"digest": "checks"}, {"digest": "profile"})
            cfg["engineering"] = {"microbatch": 4, "workers": 0}
            trainer_type = engine.Trainer
            def cpu_trainer(*args, **kwargs):
                kwargs.update(device="cpu", precision="fp32")
                return trainer_type(*args, **kwargs)
            with patch.object(pilot, "load_head", return_value=(head, "head3")), \
                 patch.object(pilot, "load_bundle", side_effect=bundle), \
                 patch.object(engine, "Trainer", side_effect=cpu_trainer), \
                 patch.object(torch.cuda, "is_available", return_value=False), \
                 patch.object(torch.cuda, "max_memory_reserved", return_value=0):
                result = engine._run_prepared(cfg, assets, student_factory=pilot.student_factory,
                    expected_stage="preliminary", purpose=pilot.PURPOSE)
                self.assertEqual((result["completed_epochs"], result["full_epochs"]), (10, 30))
                saved = torch.load(root / "run/last.pt", weights_only=False)
                self.assertEqual(saved["identity"]["purpose"], pilot.PURPOSE)
                self.assertGreater(saved["optimizer"]["param_groups"][0]["lr"], 0)
                replay = engine._replay_prepared(cfg, assets, output=root / "replay", student_factory=pilot.student_factory,
                    expected_stage="preliminary", purpose=pilot.PURPOSE, device="cpu")
                self.assertTrue(replay["passed"])
                self.assertEqual(replay["identity"]["stage"], "preliminary")
                exported = torch.load(root / "replay/student.pt", weights_only=True)
                self.assertFalse(any("text_model" in name for name in exported["student"]))
                with self.assertRaisesRegex(ValueError, "epoch10"):
                    engine._run_prepared(cfg, assets, resume=True, student_factory=pilot.student_factory,
                        expected_stage="preliminary", purpose=pilot.PURPOSE)

    def test_night_flow_and_failure_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "preliminary"
            source_path = Path(directory) / "source.json"
            write_json(source_path, {"fixture": True})
            runtime = {"gpu": {"name": "Tesla T4", "uuid": "fixture"}}
            profile = {"status": "passed", "runtime": runtime, "head_sha256": "head3", "digest": "profile",
                       "zero_selection_policy": "retain_observed"}
            calls = []
            with patch.object(pilot, "load_config", return_value={"stage": "preliminary", "machine_config": "machine"}), \
                 patch.object(engine, "require_machine"), patch.object(pilot, "prepare", return_value=context()), \
                 patch.object(admission, "runtime_identity", return_value=runtime), \
                 patch.object(pilot, "load_head", return_value=({}, "head3")), \
                 patch.object(pilot, "software_checks", side_effect=lambda root: calls.append("checks") or {"digest": "checks"}), \
                 patch.object(admission, "_profile_one", side_effect=lambda *a, **kw: calls.append("profile") or profile) as prof, \
                 patch.object(engine, "_run_prepared", side_effect=lambda *a, **kw: calls.append("train") or {}) as train, \
                 patch.object(engine, "_replay_prepared", side_effect=lambda *a, **kw: calls.append("replay") or {}) as replay, \
                 patch.object(pilot, "write_morning_report", side_effect=lambda *a: calls.append("report") or {"done": True}):
                self.assertEqual(pilot.run_night(source_path, root / "pass"), {"done": True})
                self.assertEqual(calls, ["checks", "profile", "train", "replay", "report"])
                self.assertEqual(prof.call_args.kwargs["_stage"], "preliminary")
                self.assertEqual(prof.call_args.kwargs["_zero_selection_policy"], "retain_observed")
                self.assertEqual(train.call_args.kwargs["expected_stage"], "preliminary")
                self.assertEqual(train.call_args.kwargs["purpose"], pilot.PURPOSE)
                self.assertEqual(replay.call_args.kwargs["expected_stage"], "preliminary")
                calls.clear()
                profile["zero_selection_policy"] = "error"
                with self.assertRaisesRegex(ValueError, "profile identity mismatch"):
                    pilot.run_night(source_path, root / "wrong-policy")
                self.assertEqual(calls, ["checks", "profile"])
                profile["zero_selection_policy"] = "retain_observed"
                calls.clear()
                prof.side_effect = MethodError("synthetic GMM failure")
                with self.assertRaises(MethodError):
                    pilot.run_night(source_path, root / "fail")
                self.assertEqual(calls, ["checks"])
                self.assertTrue((root / "fail/MORNING-FAILED.md").is_file())
                self.assertTrue((root / "fail/failure-001.json").is_file())

    def test_morning_report_contains_all_ten_epochs_and_actual_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for epoch in range(1, 11):
                write_json(root / "run" / f"epoch-{epoch:02d}.json", {"epoch": epoch, "metrics": {"macro_recall": .5, "micro_top1": .6},
                    "selected": 123, "epoch_seconds": 10.,
                    "selection": {"summary": {"confident_selected": 100, "retained_observed": 23}}})
            result = {"status": "paused_at_epoch10", "best_epoch": 7, "best_metrics": {"macro_recall": .6},
                      "last": {"macro_recall": .5}, "checkpoint_sha256": "last", "best_sha256": "best"}
            cfg = {"digest": "config", "runtime": {}, "asset_digest": "assets", "head_sha256": "head3"}
            report = pilot.write_morning_report(root, cfg, result, {"passed": True})
            self.assertEqual(report["stage"], "preliminary")
            self.assertEqual(len(report["epoch_seconds"]), 10)
            self.assertEqual(report["method_variant"], pilot.VARIANT)
            self.assertEqual(report["selection_summaries"], [{"confident_selected": 100, "retained_observed": 23}] * 10)
            self.assertIn("不能据绝对分数作严格因果对照", (root / "MORNING.md").read_text())
            self.assertFalse(read_json(root / "morning-summary.json")["round2_eligible"])


if __name__ == "__main__":
    unittest.main()
