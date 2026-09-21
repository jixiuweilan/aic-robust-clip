"""Synthetic correctness only: no competition assets, downloads or GPU skips."""
import copy
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image

from aic_robust_clip.contracts import read_json, write_json
from aic_robust_clip.round2 import admission, assets, config, engine
from aic_robust_clip.round2.methods import (MethodError, MethodState, ClassQueue, StochasticFeature,
                                           corrected_labels, fine_scores, gmm, select)
from aic_robust_clip.round2.model import optimizer_groups
from aic_robust_clip.runtime import seed_everything


class MethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_gmm_determinism_component_and_failures(self):
        x = np.r_[np.linspace(.1, .2, 12), np.linspace(2., 3., 12)]
        p = gmm(x)
        np.testing.assert_array_equal(p, gmm(x))
        self.assertTrue((p[:12] > .99).all())
        self.assertTrue((p[12:] < .01).all())
        np.testing.assert_allclose(p + gmm(x, high=True), np.ones(24))
        for bad in ([1] * 8, [1, 2], [float("nan")] * 8):
            with self.assertRaises(MethodError):
                gmm(bad)
        with self.assertRaisesRegex(MethodError, "converge"):
            gmm(x, max_iter=1)
        repeated = gmm([0.] * 20 + [1.] * 2)
        np.testing.assert_array_equal(repeated, [1.] * 20 + [0.] * 2)

    def test_fine_geometry_and_classwise_degeneracy(self):
        features = np.array([[1., 0]] * 12 + [[0., 1]] * 8)
        scores = fine_scores(features)
        np.testing.assert_allclose(scores[:12], 1)
        np.testing.assert_allclose(scores[12:], 0)
        np.testing.assert_allclose(fine_scores(np.pad(features, ((0, 0), (0, 32)))), scores)
        keep, probability, report = select(np.zeros(20), np.ones(20), features, method="fine")
        self.assertEqual(keep.tolist(), [True] * 12 + [False] * 8)
        keep, probability, report = select([0] * 4 + [1] * 8, np.ones(12), method="turn")
        self.assertTrue(keep.all())
        self.assertTrue(np.isnan(probability).all())
        self.assertEqual(report["0"]["reason"], "fewer_than_8")
        self.assertEqual(report["1"]["reason"], "constant")
        with patch("aic_robust_clip.round2.methods.gmm", return_value=np.zeros(20)):
            with self.assertRaisesRegex(MethodError, "zero selected"):
                select(np.zeros(20), np.arange(20), method="turn")
        with self.assertRaises(MethodError):
            select(np.zeros(12), np.ones(12), method="snscl")

    def test_soft_labels_equations_and_stochastic_reparameterization(self):
        observed = torch.eye(3)
        predictions = torch.full((3, 3), 1 / 3)
        gamma = torch.tensor([1., .5, 0.])
        soft, weights = corrected_labels(observed, predictions, observed, gamma)
        torch.testing.assert_close(weights, torch.tensor([1., .5, 0.]))
        torch.testing.assert_close(soft[0], observed[0])
        torch.testing.assert_close(soft[1], .995 * observed[1] + .005 * predictions[1])
        torch.testing.assert_close(soft[2], .99 * observed[2] + .01 * predictions[2])
        module = StochasticFeature(2)
        for p in module.parameters():
            torch.nn.init.zeros_(p)
        module.network[-1].bias.data.copy_(torch.tensor([2., 3., np.log(4), np.log(9)]))
        gen = torch.Generator().manual_seed(4)
        expected = torch.tensor([2., 3.]) + torch.tensor([2., 3.]) * torch.randn((3, 2), generator=gen)
        actual, kl = module(torch.ones(3, 2), generator=torch.Generator().manual_seed(4))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(kl, torch.full((3,), .5 * (4 + 9 + 4 + 9 - 2 - np.log(4) - np.log(9))))

    def test_queue_empty_rejection_wrap_and_valid_keys_only(self):
        queue = ClassQueue(2, dimension=2, capacity=2)
        g = torch.Generator().manual_seed(5)
        queries = torch.eye(2, requires_grad=True)
        torch.testing.assert_close(queue.loss(queries, torch.tensor([0, 1])), torch.zeros(2))
        queue.enqueue(torch.eye(2), [0, 1], torch.zeros(2), generator=g)
        self.assertEqual(int(queue.count.sum()), 0)
        self.assertEqual(int(queue.pointer.sum()), 0)
        self.assertEqual(float(queue.features.sum()), 0)
        queue.enqueue(torch.tensor([[1., 0], [0, 1], [-1, 0]]), [0, 0, 0], torch.ones(3), generator=g)
        self.assertEqual(queue.count.tolist(), [2, 0])
        self.assertEqual(queue.pointer.tolist(), [1, 0])
        torch.testing.assert_close(queue.features[0], torch.tensor([[-1., 0], [0, 1]]))
        self.assertEqual(float(queue.loss(queries, torch.tensor([1, 1])).sum()), 0)
        expected = -torch.log_softmax(queries @ queue.features[0].T / .07, dim=1).mean(1)
        torch.testing.assert_close(queue.loss(queries, torch.tensor([0, 0])), expected)

    def test_nontrain_and_incomplete_method_states_rejected(self):
        state = MethodState("turn", ["train-a", "train-b"], [0, 1], 2)
        with self.assertRaisesRegex(MethodError, "non-train"):
            state.indices(["dev-a"])
        with self.assertRaisesRegex(MethodError, "exactly"):
            state.rescore(["train-a"], [1.], None, None, completed_epochs=1)
        for field in state.state_dict():
            bad = copy.deepcopy(state.state_dict())
            del bad[field]
            with self.assertRaises(MethodError):
                state.load_state_dict(bad)
        bad = state.state_dict()
        bad["soft"] = torch.ones(2, 2)
        with self.assertRaises(MethodError):
            state.load_state_dict(bad)

    def test_optimizer_groups_and_text_frozen(self):
        for adaptation in ("full_visual", "lora"):
            student = admission.tiny_student(adaptation)
            groups = optimizer_groups(student)
            grouped = {id(p): (g["group_name"], g["lr"], g["weight_decay"]) for g in groups for p in g["params"]}
            self.assertEqual(len(grouped), sum(len(g["params"]) for g in groups))
            for name, p in student.named_parameters():
                if "text_model" in name or "text_projection" in name or "logit_scale" in name:
                    self.assertFalse(p.requires_grad)
                    self.assertNotIn(id(p), grouped)
                if p.requires_grad:
                    self.assertIn(id(p), grouped)
                    self.assertEqual(grouped[id(p)][1], .001 if name.startswith("classifier.") else .0001 if "lora_" in name else .00001)
                    if name.endswith("bias") or "layer_norm" in name or "layernorm" in name:
                        self.assertEqual(grouped[id(p)][2], 0)
            student.unowned = torch.nn.Parameter(torch.ones(1))
            with self.assertRaisesRegex(ValueError, "unowned"):
                optimizer_groups(student)

    def test_synthetic_two_updates_all_methods_and_adaptations(self):
        for method in ("ce", "turn", "fine", "snscl"):
            for adaptation in ("full_visual", "lora"):
                report = admission.synthetic_check(method, adaptation, device="cpu", precision="fp32")
                self.assertEqual((report["updates"], report["samples"]), (2, 4))

    def trainer(self, method="snscl"):
        return engine.Trainer(admission.tiny_student("lora"), MethodState(method, ["a", "b"], [0, 1], 3),
                              identity={"fixture": True}, effective_batch=2)

    def test_amp_skip_does_not_advance_momentum_queue_or_method_rng(self):
        class SkipScaler:
            value = 8.
            def scale(self, loss): return loss
            def get_scale(self): return self.value
            def step(self, optimizer): pass
            def update(self): self.value /= 2
        trainer = self.trainer()
        before = copy.deepcopy(trainer.auxiliary.state_dict())
        rng = trainer.auxiliary.generator.get_state()
        trainer.scaler = SkipScaler()
        batch = {"image": torch.randn(2, 3, 32, 32), "sample_id": ["a", "b"], "label_index": torch.tensor([0, 1])}
        trainer.update_window([batch], epoch=5, fraction=1.)
        self.assertEqual((trainer.updates, trainer.skipped), (0, 1))
        for k, v in before.items():
            torch.testing.assert_close(v, trainer.auxiliary.state_dict()[k], rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, trainer.auxiliary.generator.get_state()))

    def test_resume_restores_student_auxiliary_rng_and_next_update(self):
        for method in ("ce", "turn", "fine", "snscl"):
            seed_everything(17)
            full = self.trainer(method)
            batch = {"image": torch.randn(2, 3, 32, 32), "sample_id": ["a", "b"], "label_index": torch.tensor([0, 1])}
            full.update_window([batch], epoch=5, fraction=.5)
            full.state.completed_epochs = 5
            full.history = [{"epoch": i + 1} for i in range(5)]
            saved = copy.deepcopy(full.checkpoint())
            full.update_window([batch], epoch=5, fraction=1)
            restored = self.trainer(method)
            restored.restore(saved)
            restored.update_window([batch], epoch=5, fraction=1)
            for k, v in full.student.state_dict().items():
                torch.testing.assert_close(v, restored.student.state_dict()[k], atol=0, rtol=0)
            if method == "snscl":
                for k, v in full.auxiliary.state_dict().items():
                    torch.testing.assert_close(v, restored.auxiliary.state_dict()[k], atol=0, rtol=0)
                self.assertTrue(torch.equal(full.auxiliary.generator.get_state(), restored.auxiliary.generator.get_state()))
            for key in ("method", "rng", "sampler", "scheduler", "optimizer", "auxiliary"):
                bad = copy.deepcopy(saved)
                del bad[key]
                with self.assertRaises(ValueError):
                    restored.restore(bad)

    def test_epoch_fraction_schedule_preserves_full_budget(self):
        self.assertAlmostEqual(engine.schedule(.5), .5)
        self.assertAlmostEqual(engine.schedule(1), 1)
        self.assertGreater(engine.schedule(10), .7)
        self.assertEqual(engine.schedule(30), 0.)

    def test_snscl_activates_after_five_epochs_and_momentum_once_per_update(self):
        trainer = self.trainer()
        batches = [{"image": torch.randn(1, 3, 32, 32), "sample_id": [s], "label_index": torch.tensor([i])}
                   for i, s in enumerate(["a", "b"])]
        with patch.object(trainer.auxiliary, "successful_update", wraps=trainer.auxiliary.successful_update) as update:
            trainer.update_window(batches, epoch=4, fraction=1)
            self.assertEqual(update.call_count, 0)
            self.assertEqual(int(trainer.auxiliary.queue.count.sum()), 0)
            trainer.update_window(batches, epoch=5, fraction=.5)
            self.assertEqual(update.call_count, 1)
            self.assertEqual(int(trainer.auxiliary.queue.count.sum()), 2)


class AssetConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixtures(self, root):
        archive = root / "train.zip"
        with zipfile.ZipFile(archive, "w") as z:
            for i in range(36):
                out = io.BytesIO()
                Image.new("RGB", (i + 8, 12), (i * 6, 70, 255 - i * 5)).save(out, format="PNG")
                z.writestr(f"{i % 3:04d}/{i}.png", out.getvalue())
        identity = {"model_id": "openai/clip-vit-base-patch32", "revision": "a" * 40,
                    "digest": "b" * 64, "preprocessing_digest": "c" * 64}
        with patch.object(assets, "inspect_weights", return_value=identity):
            assets.audit(archive, root / "assets", source_url="https://organizer.example/round2", retrieved_at="2026-09-20",
                         organizer_version="synthetic-only", weights=root / "weights", weight_revision="a" * 40)
        return root / "assets/assets.json", identity

    def test_prepare_is_metadata_only_and_eight_blocked_configs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "second_round/configs"
            with patch.object(engine, "student_for", side_effect=AssertionError("allocated model")), \
                 patch.object(assets, "cache", side_effect=AssertionError("built cache")), \
                 patch.object(assets, "init_head", side_effect=AssertionError("trained head")):
                rows = config.prepare_configs(output)
                self.assertEqual(len(rows), 8)
                self.assertTrue(all(r["status"] == "blocked_on_round2_assets" for r in rows))
                for row in rows:
                    path = output / (row["run_id"] + ".json")
                    value, a = config.check_config(path, ready=False)
                    self.assertEqual(value["recipe"]["epochs"], 30)
                    self.assertEqual(value["recipe"]["pause_after_epoch"], 10)
                    with self.assertRaisesRegex(ValueError, "blocked_on_round2_assets"):
                        config.check_config(path)
            with self.assertRaises(ValueError):
                config.stage_path(Path(directory) / "preliminary/configs")

    def test_assets_enforce_stage_split_head_and_content_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "second_round"
            root.mkdir()
            path, identity = self.fixtures(root)
            with patch.object(config, "inspect_weights", return_value=identity):
                a = config.load_assets(path, verify_archives=True)
                self.assertEqual(a.split.counts(), {"train": 30, "dev": 3, "confirm": 3})
                self.assertEqual(len(a.dataset("train")), 30)
                for partition in ("confirm", "test"):
                    with self.assertRaises(ValueError):
                        a.dataset(partition)
                head_root = a.root / "HEAD20-GCE"
                head_root.mkdir()
                digest = engine.atomic_save({"x": torch.ones(1)}, head_root / "head.pt")
                write_json(head_root / "head.json", {"identity": {**a.head_identity(), "profile": "HEAD3"}, "sha256": digest})
                with self.assertRaisesRegex(ValueError, "HEAD20"):
                    config.head_descriptor(a)
                write_json(head_root / "head.json", {"identity": a.head_identity(), "sha256": digest})
                self.assertEqual(config.head_descriptor(a)["sha256"], digest)
                original = read_json(path)
                write_json(path, config.sealed({**config.verify_seal(original), "stage": "preliminary"}))
                with self.assertRaises(ValueError):
                    config.load_assets(path)
                write_json(path, original)
                (a.root / "split.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "hash"):
                    config.load_assets(path)

    def test_old_receipts_and_mismatched_config_recipe_rejected(self):
        with self.assertRaises(ValueError):
            admission.validate_receipt({"schema_version": 2, "recipe": "B03"}, None, live=False)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "second_round/configs"
            config.prepare_configs(out)
            path = out / "T4-0-CE.json"
            value = config.verify_seal(read_json(path))
            value["recipe"]["epochs"] = 10
            write_json(path, config.sealed(value))
            with self.assertRaises(ValueError):
                config.check_config(path, ready=False)

    def test_common_engineering_no_silent_fallback_and_tie_order(self):
        rows = []
        for b, w in admission.GRID:
            for method in ("ce", "turn"):
                rows.append({"microbatch": b, "workers": w, "method": method, "status": "passed",
                             "peak_occupied_bytes": 80, "total_bytes": 100, "projected_epoch_seconds": 100 if b == 32 else 104})
        self.assertEqual(admission.choose_common(rows, ["ce", "turn"]), {"microbatch": 4, "workers": 2})
        for row in rows:
            if row["method"] == "turn": row["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "no common"):
            admission.choose_common(rows, ["ce", "turn"])

    def test_scoring_dev_and_test_never_accepted(self):
        for partition in ("dev", "confirm", "test"):
            loader = SimpleNamespace(dataset=SimpleNamespace(stage="second_round", role="train", partition=partition, purpose="scoring"))
            with self.assertRaises(MethodError):
                engine.scoring(None, loader, device="cpu", features_required=False)

    def test_formal_guard_runs_before_assets_or_models(self):
        with patch.object(engine, "check_config", side_effect=AssertionError("read assets before permission")):
            with self.assertRaises(Exception) as caught:
                engine.run("missing.json", machine=None)
            self.assertNotIsInstance(caught.exception, AssertionError)

    def test_cli_help_and_blocked_preparation(self):
        from aic_robust_clip.round2.__main__ import main, parser
        for command in ("prepare", "check", "audit", "cache", "init-head", "checks", "profile", "choose", "admit", "run", "replay-dev", "startup-check", "concurrent"):
            with self.assertRaises(SystemExit) as caught:
                parser().parse_args([command, "--help"])
            self.assertEqual(caught.exception.code, 0)
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(main(["prepare", "--output", str(Path(directory) / "second_round/configs")]), 0)

    def test_real_loader_epoch10_pause_resume_and_student_dev_replay(self):
        from transformers import CLIPImageProcessor
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "second_round"
            root.mkdir()
            path, identity = self.fixtures(root)
            with patch.object(config, "inspect_weights", return_value=identity):
                a = config.load_assets(path)
            processor = CLIPImageProcessor()
            initial = admission.tiny_student("full_visual", image_size=224)
            initial_state = copy.deepcopy(initial.state_dict())
            def student_for(*args):
                student = admission.tiny_student("full_visual", image_size=224)
                student.load_state_dict(initial_state)
                return student, processor, {}
            cfg = {"digest": "synthetic", "adaptation": "full_visual", "method": "ce", "engineering": {"microbatch": 8, "workers": 0},
                   "output": str(root / "run"), "recipe": config.RECIPE}
            original_trainer = engine.Trainer
            def cpu_trainer(*args, **kwargs):
                kwargs.update(device="cpu", precision="fp32")
                return original_trainer(*args, **kwargs)
            with patch.object(engine, "require_machine"), patch.object(engine, "check_config", return_value=(cfg, a)), \
                 patch.object(engine, "student_for", side_effect=student_for), patch.object(engine, "Trainer", side_effect=cpu_trainer), \
                 patch.object(torch.cuda, "max_memory_reserved", return_value=0), patch.object(torch.cuda, "is_available", return_value=False):
                result = engine.run("synthetic", machine="synthetic")
                self.assertEqual((result["status"], result["completed_epochs"], result["full_epochs"]), ("paused_at_epoch10", 10, 30))
                saved = torch.load(root / "run/last.pt", weights_only=False)
                self.assertEqual(len(saved["history"]), 10)
                self.assertGreater(saved["optimizer"]["param_groups"][0]["lr"], 0)
                self.assertTrue((root / "run/dev-epoch-10.json").is_file())
                self.assertTrue((root / "run/epoch-10.json").is_file())
                with self.assertRaisesRegex(ValueError, "epoch10"):
                    engine.run("synthetic", machine="synthetic", resume=True)
                # Exact sequential-vs-resume path: interrupt before epoch 4,
                # resume from complete epoch 3 including loader epoch and RNG.
                cfg["output"] = str(root / "staged")
                train_epoch = original_trainer.train_epoch
                def interrupt(trainer, loader, **kwargs):
                    if kwargs["epoch"] == 3:
                        raise RuntimeError("synthetic interruption")
                    return train_epoch(trainer, loader, **kwargs)
                with patch.object(original_trainer, "train_epoch", interrupt):
                    with self.assertRaisesRegex(RuntimeError, "interruption"):
                        engine.run("synthetic", machine="synthetic")
                engine.run("synthetic", machine="synthetic", resume=True)
                resumed = torch.load(root / "staged/last.pt", weights_only=False)
                for k, value in saved["student"].items():
                    torch.testing.assert_close(value, resumed["student"][k], rtol=0, atol=0)
                # Explicit student-only export/reload on labelled dev.
                trained = student_for()[0]
                trained.load_state_dict(saved["student"])
                exported = {k: v for k, v in trained.state_dict().items() if "text" not in k and "logit_scale" not in k}
                fresh = student_for()[0]
                fresh.load_state_dict({**fresh.state_dict(), **exported})
                dev = a.dataset("dev", processor)
                with engine.loader_for(dev, batch=64) as loader:
                    evaluated = engine.evaluate(fresh, loader, device="cpu", classes=3, training_counts={0: 10, 1: 10, 2: 10})
                reference = torch.load(root / "run/dev-epoch-10.pt", weights_only=False)
                torch.testing.assert_close(evaluated["logits"], reference["logits"], rtol=0, atol=0)
                self.assertEqual(evaluated["predictions"], reference["predictions"])
                dev.close()


class AdmissionTests(unittest.TestCase):
    def test_complete_new_receipt_and_evidence_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = {"host": "fixture-host", "source": "fixture-source", "dependencies": {"torch": "fixture"},
                       "gpu": {"uuid": "c99db39b-3ee0-d3f4-2639-1141aa73f05d", "name": "RTX 4060", "total_bytes": 100}}
            check_dir = root / "checks"
            check_dir.mkdir()
            for i in range(4):
                (check_dir / f"check-{i}.log").write_text("synthetic metadata fixture\n")
            check = config.sealed({"version": config.VERSION, "kind": "checks", "group": "4060-a", "device": "cuda",
                "runtime": runtime, "suite": {"tests": 1, "errors": 0, "failures": 0, "skips": 0}, "returncodes": [0] * 4,
                "startup": [{"method": m, "precision": p, "status": "passed", "updates": 2}
                            for m in ("ce", "turn") for p in ("fp32", "fp16")], "previous_failure": None,
                "machine_history": {"host": "fixture-host", "gpu_uuid": "GPU-c99db39b-3ee0-d3f4-2639-1141aa73f05d", "original_b04_failure": False,
                                    "reviewer": "synthetic test", "basis": "synthetic test"},
                "logs": {f"check-{i}.log": admission.file_sha256(check_dir / f"check-{i}.log") for i in range(4)}})
            write_json(check_dir / "checks.json", check)
            def make_profiles(name, final):
                rows = []
                for m in ("ce", "turn"):
                    for b, w in ([(4, 2)] if final else admission.GRID):
                        row = config.sealed({"version": config.VERSION, "kind": "profile", "status": "passed",
                            "runtime": runtime, "asset_digest": "fixture-assets", "head_sha256": "fixture-head", "method": m,
                            "adaptation": "lora", "microbatch": b, "workers": w, "peak_occupied_bytes": 80, "total_bytes": 100,
                            "projected_epoch_seconds": 100., "eval_seconds": [1.] * (3 if final else 1),
                            "dev_student_replay": True, "workers_closed": True})
                        path = root / name / f"{m}-{b}-{w}.json"
                        write_json(path, row)
                        rows.append({"path": str(path), "sha256": admission.file_sha256(path)})
                index = root / name / "profiles.json"
                write_json(index, config.sealed({"version": config.VERSION, "kind": "profile_grid", "rows": rows}))
                return index
            profiles, final = make_profiles("grid", False), make_profiles("final", True)
            fixture_assets = SimpleNamespace(descriptor={"digest": "fixture-assets"})
            with patch.object(admission, "load_assets", return_value=fixture_assets), \
                 patch.object(admission, "head_descriptor", return_value={"sha256": "fixture-head"}), \
                 patch.object(admission, "current_code_revision", return_value="fixture-source"), \
                 patch.object(admission, "runtime_identity", return_value=runtime):
                kwargs = dict(group="4060-a", owner="synthetic owner", check_paths=[check_dir / "checks.json"],
                              profile_paths=[profiles], final_paths=[final])
                receipt = admission.admit("fixture", output=root / "receipt.json", **kwargs)
                admission.validate_receipt(receipt, fixture_assets, live=True)
                self.assertEqual(receipt["assignments"], {"ce": runtime["gpu"]["uuid"], "turn": runtime["gpu"]["uuid"]})
                with self.assertRaisesRegex(ValueError, "original B04"):
                    admission.admit("fixture", output=root / "bad.json", previous_failure_required=True, **kwargs)
                with patch.object(admission, "runtime_identity", return_value={**runtime, "host": "other-host"}):
                    with self.assertRaisesRegex(ValueError, "machine/GPU"):
                        admission.validate_receipt(receipt, fixture_assets, live=True)
                (check_dir / "check-0.log").write_text("changed")
                with self.assertRaisesRegex(ValueError, "evidence changed"):
                    admission.validate_receipt(receipt, fixture_assets, live=False)

    def test_candidate_requires_completed_matching_local_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = {"group": "4060-a", "method": "turn", "output": str(root / "4060-A-TURN"),
                         "asset_digest": "a", "head_sha256": "h", "receipt_digest": "r", "engineering": {},
                         "recipe": config.RECIPE, "adaptation": "lora"}
            control = root / "4060-A-CE"
            with self.assertRaises(FileNotFoundError):
                engine.require_matching_control(candidate)
            write_json(control / "resolved.json", candidate)
            digest = engine.atomic_save({"fixture": True}, control / "last.pt")
            write_json(control / "result.json", {"status": "paused_at_epoch10", "completed_epochs": 9, "checkpoint_sha256": digest})
            with self.assertRaisesRegex(ValueError, "finish epoch10"):
                engine.require_matching_control(candidate)
            write_json(control / "result.json", {"status": "paused_at_epoch10", "completed_epochs": 10, "checkpoint_sha256": digest})
            engine.require_matching_control(candidate)
            write_json(control / "resolved.json", {**candidate, "head_sha256": "HEAD3"})
            with self.assertRaisesRegex(ValueError, "identity differs"):
                engine.require_matching_control(candidate)


if __name__ == "__main__":
    unittest.main()
