"""Small generated-vector correctness checks, never competition training."""
import copy
import json
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from aic_robust_clip.contracts import RunConfig, sha256_json
from aic_robust_clip.training.baseline import (
    TrainConfig, TrainingError, evaluate_loader, score_training_loader, train_baseline, validate_pause,
)
from aic_robust_clip.training.checkpoint import publish_best
from test_performance import fixture_stream
from test_regressions import metadata


class Adapted(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 4)
        self.classifier = torch.nn.Linear(4, 2)

    def forward_with_features(self, image):
        feature = self.encoder(image)
        return self.classifier(feature), feature

    def forward(self, image):
        return self.forward_with_features(image)[0]


def config(**options):
    run = RunConfig(stage="preliminary", execution_mode="formal", batch_size=2,
                    manifest_digest="a" * 64, split_digest="b" * 64,
                    class_map_digest="c" * 64, official_weight_revision="d" * 40)
    return TrainConfig(run, epochs=10, accumulation_steps=2, scheduler="warmup_cosine", **options)


class StageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_complete_epoch_resume_ce_gce_w_p(self):
        labels = {f"s{i}": i % 2 for i in range(10)}
        for options in ({}, {"objective": "gce"}, {"weighting": True}, {"lambda_preserve": .1}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory, \
                    patch("aic_robust_clip.training.baseline.resolve_run_config"):
                cfg, root = config(**options), Path(directory)
                initial, reference = Adapted(), torch.nn.Linear(4, 4)
                full, staged = copy.deepcopy(initial), copy.deepcopy(initial)

                def run(model, name, stop, resume=None):
                    with fixture_stream(count=10) as train, fixture_stream(count=10) as dev, \
                            fixture_stream(count=10) as scoring:
                        scoring.shuffle = False
                        scoring.reset(0)
                        result = train_baseline(model, train, config=cfg, class_count=2, device="cpu",
                            dev_loader=dev, training_labels=labels, scoring_loader=scoring,
                            reference_encoder=copy.deepcopy(reference), checkpoint_dir=root / name,
                            checkpoint_metadata=metadata(cfg), stop_after_epochs=stop, resume_from=resume)
                    return result

                whole = run(full, "full", 10)
                first = run(staged, "staged", 3)
                self.assertEqual((first.completed_epochs, first.paused, len(first.epoch_logs)), (3, True, 3))
                self.assertTrue(all(row["complete"] for row in first.epoch_logs))
                checkpoint = root / "staged/last.pt"
                third = torch.load(checkpoint, weights_only=False)
                self.assertEqual(third["sampler_state"]["position"], 0)
                self.assertEqual(third["sampler_state"]["epoch"], 3)
                if cfg.weighting:
                    state = third["module_state"]["reliability"]
                    self.assertEqual(state["completed_epochs"], 3)
                    self.assertTrue(any(w < 1 for w in state["weights"].values()))
                with self.assertRaisesRegex(TrainingError, "already completed"):
                    run(staged, "staged", 3, checkpoint)
                middle = run(staged, "staged", 6, checkpoint)
                last = run(staged, "staged", 10, checkpoint)
                self.assertEqual((middle.completed_epochs, middle.paused), (6, True))
                self.assertEqual((last.completed_epochs, last.paused), (10, False))
                self.assertEqual((last.updates, last.samples), (30, 100))
                self.assertEqual(last.scoring_samples, 100 if cfg.weighting else 0)
                self.assertEqual(last.reference_samples, 100 if cfg.lambda_preserve else 0)
                # The two-class fixture has an empty tail group (NaN); compare
                # serialized metrics so NaN != NaN doesn't mask real parity.
                self.assertEqual(json.dumps(whole.epoch_logs, sort_keys=True), json.dumps(last.epoch_logs, sort_keys=True))
                for key, tensor in full.state_dict().items():
                    self.assertTrue(torch.equal(tensor, staged.state_dict()[key]), key)
                a, b = (torch.load(root / name / "last.pt", weights_only=False) for name in ("full", "staged"))
                self.assertEqual(a["scheduler_state"], b["scheduler_state"])
                self.assertEqual(a["sampler_state"], b["sampler_state"])
                self.assertEqual(a["module_state"]["reliability"], b["module_state"]["reliability"])
                self.assertEqual(len(list((root / "staged").glob("timing-epoch-*.json"))), 10)
                for timing in last.phase_timings:
                    self.assertGreater(timing["train_seconds"], 0)
                    self.assertGreater(timing["dev_seconds"], 0)
                    self.assertGreater(timing["checkpoint_seconds"], 0)

    def test_pause_validation_and_fp32_digest_compatibility(self):
        cfg = config()
        for updates, epochs in ((1, 3), (None, 0), (None, 11), (None, True), (0, None), (True, None)):
            with self.assertRaises(TrainingError):
                validate_pause(cfg, updates, epochs)
        with self.assertRaisesRegex(TrainingError, "formal-only"):
            validate_pause(TrainConfig(RunConfig(stage="preliminary")), None, 1)
        old = asdict(cfg)
        old.pop("precision")
        self.assertEqual(cfg.digest, sha256_json(old))
        self.assertNotEqual(cfg.digest, replace(cfg, precision="fp16").digest)
        with self.assertRaises(ValueError):
            replace(cfg, precision="bf16")

    def test_scoring_matches_raw_ce_and_restores_mode(self):
        model = Adapted().train()
        with fixture_stream(count=10) as loader:
            loader.shuffle = False
            loader.reset(0)
            scores = score_training_loader(model, loader, device="cpu", max_batches=2)
            self.assertEqual(set(scores), {"s0", "s1", "s2", "s3"})
            self.assertEqual(loader.position, 4)
        self.assertTrue(model.training)
        with fixture_stream(count=10) as loader:
            loader.shuffle = False
            loader.reset(0)
            for _ in range(2):
                batch = next(loader)
                values = torch.nn.functional.cross_entropy(model(batch["image"]), batch["label_index"], reduction="none")
                for sample, value in zip(batch["sample_id"], values.tolist()):
                    self.assertAlmostEqual(scores[sample], value)

    def test_nonfinite_eval_and_scoring_fail_closed(self):
        class Bad(Adapted):
            def forward(self, x):
                return super().forward(x) * float("nan")
        for phase in ("eval", "scoring"):
            with fixture_stream() as loader, self.assertRaisesRegex(TrainingError, "nonfinite"):
                if phase == "eval":
                    evaluate_loader(Bad(), loader, device="cpu", total_classes=2)
                else:
                    score_training_loader(Bad(), loader, device="cpu")

    def test_best_snapshot_is_not_changed_when_last_is_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save({"x": 1}, root / "last.pt")
            publish_best(root)
            torch.save({"x": 2}, root / "next.pt")
            (root / "next.pt").replace(root / "last.pt")
            self.assertEqual(torch.load(root / "best.pt", weights_only=True)["x"], 1)
            with patch("aic_robust_clip.training.checkpoint.os.link", side_effect=OSError("no hardlinks")):
                publish_best(root)
            self.assertEqual(torch.load(root / "best.pt", weights_only=True)["x"], 2)

    def test_amp_scaler_resume_control_flow_on_cpu_fixture(self):
        # Test accumulation/scaler persistence using a real CPU scaler, while
        # mocking CUDA dispatch. This is NOT CUDA numerical/performance evidence.
        import aic_robust_clip.training.baseline as baseline
        original_scaler = torch.amp.GradScaler
        cfg = config(precision="fp16")
        initial = Adapted()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(baseline, "resolve_run_config"), \
                patch.object(baseline, "_phase_clock", side_effect=lambda _: time.monotonic()), \
                patch.object(baseline, "_prepare_images", side_effect=lambda x, _: x), \
                patch.object(baseline, "_prepare_labels", side_effect=lambda x, _: x), \
                patch.object(torch, "autocast", side_effect=lambda *a, **k: nullcontext()), \
                patch.object(torch.amp, "GradScaler", side_effect=lambda *a, **k: original_scaler("cpu", **k)), \
                patch.object(Adapted, "to", side_effect=lambda *a, **k: None):
            root = Path(directory)
            full, staged = copy.deepcopy(initial), copy.deepcopy(initial)
            def run(model, name, stop, resume=None):
                with fixture_stream(count=10) as loader:
                    return train_baseline(model, loader, config=cfg, class_count=2, device="cuda",
                        checkpoint_dir=root / name, checkpoint_metadata=metadata(cfg),
                        stop_after_epochs=stop, resume_from=resume)
            run(full, "full", 10)
            run(staged, "staged", 3)
            run(staged, "staged", 10, root / "staged/last.pt")
            for name, tensor in full.state_dict().items():
                self.assertTrue(torch.equal(tensor, staged.state_dict()[name]))
            a, b = (torch.load(root / name / "last.pt", weights_only=False) for name in ("full", "staged"))
            self.assertEqual(a["module_state"]["grad_scaler"], b["module_state"]["grad_scaler"])


if __name__ == "__main__":
    unittest.main()
