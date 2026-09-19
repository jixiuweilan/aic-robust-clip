"""Exercise robust-recipe benchmark wiring without touching CUDA or archives."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from aic_robust_clip import benchmark, workflow
from aic_robust_clip.benchmark import StepTimer
from aic_robust_clip.configuration import RECIPES
from aic_robust_clip.contracts import RunConfig
from aic_robust_clip.training.baseline import TrainConfig
from test_performance import AddressedFixture, fixture_stream


class ResearchBenchmarkTests(unittest.TestCase):
    def test_method_training_paths_and_scoring_are_bounded(self):
        for recipe, phase in (("R01", "train"), ("F100", "train"), ("F010", "train"), ("F100", "scoring")):
            with self.subTest(recipe=recipe, phase=phase), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run = RunConfig(stage="preliminary", execution_mode="formal", batch_size=1)
                cfg = {"execution_mode": "formal", "device": "cuda", "recipe": recipe, "effective_batch_size": 2}
                train_cfg = TrainConfig(run, accumulation_steps=2, **RECIPES[recipe])
                ctx = SimpleNamespace(config=cfg, run=run, train=train_cfg, policy=SimpleNamespace(allow_formal=True),
                    class_map=SimpleNamespace(id_to_index={"0000": 0, "0001": 1}),
                    performance=SimpleNamespace(cache_batch_size=2, eval_batch_size=2, scoring_batch_size=2),
                    summary=lambda: {"config": cfg})
                tiny = AddressedFixture(10)
                tiny.records = [SimpleNamespace(sample_id=f"s{i}", class_id=f"{i % 2:04}") for i in range(10)]
                tiny.class_to_index = ctx.class_map.id_to_index
                tiny.close = lambda: None
                reference, model = Mock(), Mock()
                observed = {}
                def train(_model, loader, **kwargs):
                    self.assertEqual(loader.diagnostics()["dispatch_stop"], 6)
                    self.assertNotIn("dev_loader", kwargs)
                    if recipe == "F100":
                        self.assertEqual(kwargs["scoring_loader"].position, 0)
                        self.assertEqual(len(kwargs["training_labels"]), 10)
                        self.assertGreater(len(set(kwargs["reliability"].weights.values())), 1)
                    if recipe == "F010":
                        self.assertIs(kwargs["reference_encoder"], reference)
                    timer = kwargs["observer"]
                    for _ in range(3):
                        timer.begin_step(); timer.data_ready(); timer.end_step(2)
                    observed["train"] = kwargs["stop_after_updates"]
                    return SimpleNamespace(updates=3)
                def scoring(_model, loader, **kwargs):
                    self.assertEqual(loader.diagnostics()["dispatch_stop"], 3)
                    timer = kwargs["observer"]
                    for _ in range(kwargs["max_batches"]):
                        timer.begin_step(); timer.data_ready(); timer.end_step(1)
                    observed["scoring"] = kwargs["max_batches"]
                    return {"fake": 1.}
                with patch.object(benchmark, "load_config", return_value=cfg), \
                        patch.object(benchmark, "prepare", return_value=ctx), \
                        patch.object(benchmark, "StepTimer", side_effect=lambda w, m: StepTimer(w, m, cuda=False)), \
                        patch.object(benchmark, "environment_report", return_value={}), \
                        patch.object(torch.cuda, "is_available", return_value=True), \
                        patch.object(torch.cuda, "reset_peak_memory_stats"), \
                        patch.object(torch.cuda, "max_memory_allocated", return_value=0), \
                        patch.object(torch.cuda, "max_memory_reserved", return_value=0), \
                        patch.object(workflow, "training_components", return_value=(model, tiny, tiny, tiny, reference, "head")), \
                        patch.object(workflow, "stream", side_effect=lambda *a, **k: fixture_stream(batch=1, count=10)), \
                        patch.object(workflow, "metadata_for", return_value=SimpleNamespace(model_family="BENCHMARK")), \
                        patch.object(benchmark, "train_baseline", side_effect=train), \
                        patch.object(benchmark, "score_training_loader", side_effect=scoring):
                    report = benchmark.benchmark_command("unused", phase, root / "result", warmup_steps=1, measure_steps=2)
                self.assertEqual(observed[phase], 3)
                self.assertFalse(report["selection_eligible"])
                self.assertEqual(report["reference_forward_included"], recipe == "F010")
                if recipe == "F100" and phase == "train":
                    self.assertEqual(report["weighting_state"], "synthetic_nonuniform_benchmark_only")


if __name__ == "__main__":
    unittest.main()
