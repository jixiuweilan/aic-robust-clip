"""Synthetic correctness tests only; no competition assets or GPU profiling."""
from __future__ import annotations

import copy
import io
import os
import pickle
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from aic_robust_clip.benchmark import StepTimer, benchmark_command, validate_request
from aic_robust_clip.configuration import prepare
from aic_robust_clip.contracts import RunConfig, read_json, write_json
from aic_robust_clip.data.audit import audit_archive
from aic_robust_clip.data.cache import CachedDataset, generate_cache
from aic_robust_clip.data.dataset import DatasetError, ManifestDataset, SampleItem
from aic_robust_clip.data.loading import StatefulBatchLoader
from aic_robust_clip.data.transforms import ClipTransform
from aic_robust_clip.performance import PerformanceConfig, transfer_tensor
from aic_robust_clip.performance_configs import prepare_candidates
from aic_robust_clip.runtime import RuntimeLimitError
from aic_robust_clip.training.baseline import FrozenFeatureBaseline, TrainConfig, evaluate_loader, train_baseline
from aic_robust_clip.training.checkpoint import load_checkpoint
from test_regressions import metadata
import test_workflow as workflow_fixtures


class AddressedFixture:
    def __init__(self, count=9, fail=None):
        self.count, self.fail, self.epoch = count, fail, 0

    def __len__(self):
        return self.count

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        if index == self.fail:
            raise ValueError("synthetic worker failure")
        return SampleItem(torch.tensor([index / 10, self.epoch / 10, 1., -1.]), f"s{index}", str(index % 2), index % 2)


def fixture_stream(*, workers=0, batch=2, count=9, fail=None):
    return StatefulBatchLoader(AddressedFixture(count, fail), batch_size=batch, seed=17,
        sample_ids=[f"s{i}" for i in range(count)], num_workers=workers)


def generated_archive(root, count=7):
    from PIL import Image
    source = root / "fixture.zip"
    with zipfile.ZipFile(source, "w") as archive:
        for index in range(count):
            raw = io.BytesIO()
            Image.new("RGB", (12 + index, 14), (index * 30, 50, 70)).save(raw, format="PNG")
            archive.writestr(f"{index % 2:04}/{index}.png", raw.getvalue())
    records = audit_archive(source, stage="preliminary", role="train").records
    return ManifestDataset(records, stage="preliminary", role="train", partition="train",
        purpose="train", class_to_index={"0000": 0, "0001": 1})


class MeanEncoder(torch.nn.Module):
    def forward(self, images):
        return images.mean(dim=(-1, -2))


class PerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_legacy_defaults_and_explicit_settings(self):
        original = {"execution_mode": "formal", "batch_size": 16}
        snapshot = copy.deepcopy(original)
        perf = PerformanceConfig.from_config(original)
        self.assertEqual((perf.cache_batch_size, perf.eval_batch_size, perf.head_batch_size), (16, 16, 16))
        self.assertEqual((perf.num_workers, perf.pin_memory), (0, False))
        self.assertEqual(original, snapshot)
        perf = PerformanceConfig.from_config({**original, "performance": {
            "cache_batch_size": 64, "eval_batch_size": 64, "head_batch_size": 128,
            "num_workers": 4, "pin_memory": True}})
        self.assertEqual(perf.head_batch_size, 128)

    def test_invalid_settings_and_smoke_reject_before_artifacts(self):
        cases = [{"unknown": 1}, {"num_workers": -1}, {"num_workers": True}, {"num_workers": 17},
                 {"prefetch_factor": 0}, {"prefetch_factor": 5}, {"pin_memory": 1},
                 {"head_batch_size": 3}, {"cache_batch_size": 1.5}, {"eval_batch_size": False}]
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                PerformanceConfig.from_config({"execution_mode": "formal", "performance": value})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for perf in ({"num_workers": 1}, {"cache_batch_size": 2}, {"eval_batch_size": 2},
                         {"head_batch_size": 2}, {"pin_memory": True}):
                write_json(path, {"schema_version": 2, "recipe": "B03", "stage": "preliminary", "performance": perf})
                with patch("aic_robust_clip.configuration.load_manifest", side_effect=AssertionError("artifact read")):
                    with self.assertRaisesRegex(ValueError, "smoke"):
                        prepare(path)

    def test_smoke_stream_cannot_prefetch(self):
        with self.assertRaisesRegex(DatasetError, "smoke"):
            StatefulBatchLoader(AddressedFixture(2), sample_ids=["s0", "s1"], max_samples=2, num_workers=1)

    def test_accumulation_override_cannot_change_effective_batch(self):
        with self.assertRaisesRegex(ValueError, "accumulation_steps"):
            PerformanceConfig.from_config({"execution_mode": "formal", "batch_size": 16,
                "parameters": {"accumulation_steps": 128}})

    def test_transfer_only_nonblocking_for_pinned_cuda(self):
        from unittest.mock import Mock
        tensor = Mock()
        for pinned, device, expected in ((True, "cuda", True), (False, "cuda", False), (True, "cpu", False)):
            tensor.is_pinned.return_value = pinned
            transfer_tensor(tensor, device)
            self.assertEqual(tensor.to.call_args.kwargs["non_blocking"], expected)

    def test_spawn_order_epoch_tail_and_parent_rng(self):
        with fixture_stream() as serial, fixture_stream(workers=2) as parallel:
            for epoch in (0, 1):
                serial.reset(epoch)
                parallel.reset(epoch)
                rng = torch.get_rng_state().clone()
                expected = list(serial)
                actual = list(parallel)
                self.assertTrue(torch.equal(torch.get_rng_state(), rng))
                self.assertEqual([x["sample_id"] for x in expected], [x["sample_id"] for x in actual])
                self.assertEqual(len(actual[-1]["sample_id"]), 1)
                for left, right in zip(expected, actual):
                    self.assertTrue(torch.equal(left["image"], right["image"]))
                    self.assertTrue(torch.equal(left["label_index"], right["label_index"]))
            self.assertEqual(parallel.state_dict(), serial.state_dict())
            workers = list(parallel._iterator._workers)
        self.assertTrue(all(not worker.is_alive() for worker in workers))

    def test_prefetch_pause_restore_uses_delivered_cursor(self):
        with fixture_stream(workers=2) as paused:
            first = next(paused)
            state = paused.state_dict()
            self.assertEqual(state["position"], 2)
            paused.load_state_dict(state)  # partial reset shuts down stale queue
            second = next(paused)
        with fixture_stream() as serial, fixture_stream(workers=1) as resumed:
            reference = list(serial)
            resumed.load_state_dict(state)
            remaining = list(resumed)
        self.assertEqual(first["sample_id"], reference[0]["sample_id"])
        self.assertEqual(second["sample_id"], reference[1]["sample_id"])
        self.assertEqual([b["sample_id"] for b in remaining], [b["sample_id"] for b in reference[1:]])

    def test_worker_exception_keeps_cursor_and_closes(self):
        with fixture_stream(workers=1, fail=0) as loader:
            loader.shuffle = False
            loader.reset(0)
            with self.assertRaisesRegex(ValueError, "synthetic worker failure"):
                next(loader)
            self.assertEqual(loader.position, 0)
            self.assertIsNone(loader._parallel)
            self.assertIsNone(loader._iterator)

    def test_cleanup_failure_does_not_mask_model_failure(self):
        loader = fixture_stream()
        with self.assertRaisesRegex(ValueError, "model failed"), \
                patch.object(loader, "close", side_effect=RuntimeError("cleanup failed")):
            with loader:
                raise ValueError("model failed")

    def test_real_zip_spawn_and_sample_addressed_augmentation(self):
        from transformers import CLIPImageProcessor
        with tempfile.TemporaryDirectory() as directory:
            dataset = generated_archive(Path(directory))
            pickle.loads(pickle.dumps(dataset))  # default transform is spawn safe
            dataset.transform = ClipTransform(CLIPImageProcessor(), online=True, seed=17)
            # Deliberately open a parent ZipFile; it must not be serialized.
            dataset[0]
            with StatefulBatchLoader(dataset, batch_size=2) as serial, \
                    StatefulBatchLoader(dataset, batch_size=2, num_workers=1) as parallel:
                for epoch in (0, 1):
                    serial.reset(epoch)
                    parallel.reset(epoch)
                    for left, right in zip(serial, parallel):
                        self.assertTrue(torch.equal(left["image"], right["image"]))
            dataset.close()

    def test_cache_coverage_identity_and_mmap_pickle(self):
        from transformers import CLIPImageProcessor
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = generated_archive(root)
            dataset.transform = ClipTransform(CLIPImageProcessor())
            key = {"execution_mode": "formal", "partition": "train"}  # generated seven-image fixture only
            a = generate_cache(dataset, MeanEncoder(), root / "a", key=key, device="cpu", batch_size=1, shard_rows=3)
            b = generate_cache(dataset, MeanEncoder(), root / "b", key=key, device="cpu", batch_size=4,
                num_workers=1, shard_rows=3)
            self.assertEqual(a["rows"], b["rows"])
            self.assertEqual(a["records_digest"], b["records_digest"])
            left, right = (CachedDataset(dataset, root / name, expected_key=key) for name in ("a", "b"))
            for i in range(len(dataset)):
                torch.testing.assert_close(left[i].image, right[i].image)
            self.assertTrue(left._mapped)
            self.assertFalse(pickle.loads(pickle.dumps(left))._mapped)
            dataset.close()

    def test_spawn_worker_revalidates_changed_archive_mapping(self):
        from transformers import CLIPImageProcessor
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = generated_archive(root)
            dataset.transform = ClipTransform(CLIPImageProcessor())
            record = dataset.records[0]
            mapping = {"schema_version": 1, "reason": "synthetic spawn test", "archives": {
                record.archive_path: {"path": record.archive_path, "sha256": record.archive_identity}}}
            path = root / "locations.json"
            write_json(path, mapping)
            with patch.dict(os.environ, {"AIC_ARCHIVE_LOCATIONS": str(path)}), \
                    StatefulBatchLoader(dataset, batch_size=1, num_workers=1, prefetch_factor=1) as loader:
                next(loader)
                mapping["archives"][record.archive_path]["sha256"] = "0" * 64
                write_json(path, mapping)
                with self.assertRaisesRegex(RuntimeError, "hash mismatch|changed while being read"):
                    list(loader)
                self.assertLess(loader.position, len(dataset))
                self.assertIsNone(loader._parallel)
            dataset.close()

    def test_prefetched_checkpoint_resume_matches_two_synthetic_updates(self):
        # Mock only policy validation for this eight-vector, two-update test.
        # No production host binding is changed and no full epoch is executed.
        run = RunConfig(stage="preliminary", execution_mode="formal", batch_size=2,
            manifest_digest="a" * 64, split_digest="b" * 64, class_map_digest="c" * 64,
            official_weight_revision="d" * 40)
        config = TrainConfig(run, scheduler="warmup_cosine")
        initial = FrozenFeatureBaseline(4, 2).state_dict()
        with tempfile.TemporaryDirectory() as directory, \
                patch("aic_robust_clip.training.baseline.resolve_run_config"):
            root = Path(directory)
            continuous, resumed = FrozenFeatureBaseline(4, 2), FrozenFeatureBaseline(4, 2)
            continuous.load_state_dict(initial)
            resumed.load_state_dict(initial)
            with fixture_stream(count=8) as loader:
                train_baseline(continuous, loader, config=config, class_count=2, device="cpu",
                    checkpoint_dir=root / "full", checkpoint_metadata=metadata(config), stop_after_updates=2)
            with fixture_stream(workers=1, count=8) as loader:
                train_baseline(resumed, loader, config=config, class_count=2, device="cpu",
                    checkpoint_dir=root / "resume", checkpoint_metadata=metadata(config), stop_after_updates=1)
            with fixture_stream(workers=1, count=8) as loader:
                result = train_baseline(resumed, loader, config=config, class_count=2, device="cpu",
                    checkpoint_dir=root / "resume", checkpoint_metadata=metadata(config), stop_after_updates=2,
                    resume_from=root / "resume/last.pt")
            self.assertEqual((result.updates, result.samples), (2, 4))
            for name, tensor in continuous.state_dict().items():
                self.assertTrue(torch.equal(tensor, resumed.state_dict()[name]))
            left, right = (torch.load(root / name / "last.pt", weights_only=False) for name in ("full", "resume"))
            self.assertEqual(left["scheduler_state"], right["scheduler_state"])
            self.assertEqual(left["sampler_state"], right["sampler_state"])
            for key, value in left["optimizer_state"]["state"].items():
                for name, tensor in value.items():
                    self.assertTrue(torch.equal(tensor, right["optimizer_state"]["state"][key][name]))

    def test_workflow_independent_batches_and_head_accumulation(self):
        from aic_robust_clip import workflow
        helpers = workflow_fixtures.WorkflowTests()
        identity = {"model_id": "openai/clip-vit-base-patch32", "revision": "a" * 40,
                    "digest": "b" * 64, "preprocessing_digest": "c" * 64}
        with tempfile.TemporaryDirectory() as directory, \
                patch("aic_robust_clip.configuration.inspect_weights", return_value=identity):
            root = Path(directory)
            config = helpers.fixtures(root)
            write_json(root / "config.json", config)
            ctx = prepare(root / "config.json")
            self.assertNotIn("performance", ctx.summary()["config"])
            ctx.run = replace(ctx.run, execution_mode="formal", batch_size=16)
            ctx.config.update(execution_mode="formal", batch_size=16, performance={
                "cache_batch_size": 64, "eval_batch_size": 32, "head_batch_size": 128, "num_workers": 4})
            for partition, batch in (("train", 16), ("dev", 32)):
                with workflow.stream(ctx, workflow.dataset_for(ctx, partition)) as loader:
                    self.assertEqual(loader.batch_size, batch)
            dataset = SimpleNamespace(feature_dim=4, purpose="train")
            observed = {}
            def fake_train(model, loader, **kwargs):
                observed.update(batch=loader.batch_size, config=kwargs["config"])
                return SimpleNamespace(to_dict=lambda: {})
            with patch.object(workflow, "prepare", return_value=ctx), \
                    patch.object(workflow, "cache_for", return_value=dataset), \
                    patch.object(workflow, "stream", return_value=fixture_stream(batch=128)), \
                    patch.object(workflow, "train_baseline", side_effect=fake_train):
                workflow.init_head_command(root / "config.json")
            self.assertEqual((observed["batch"], observed["config"].accumulation_steps), (128, 1))
            self.assertEqual(observed["config"].run.batch_size, 128)

    def test_candidate_generator_preserves_inputs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = {"schema_version": 2, "recipe": "B03", "stage": "preliminary",
                "execution_mode": "formal", "device": "cuda", "batch_size": 1,
                "manifest": "manifest.json", "head": "head", "train_cache": "cache",
                "output": "original-run", "parameters": {"formal_epochs": 10, "accumulation_steps": 128}}
            path = root / "source.json"
            write_json(path, source)
            paths = prepare_candidates(path, root / "candidates")
            self.assertEqual(len(paths), 14)
            fast = read_json(root / "candidates/B03-m16-w4.json")
            pair = read_json(root / "candidates/B04-m16-w4.json")
            self.assertEqual(fast["head"], str(root / "head"))
            self.assertEqual(fast["manifest"], str(root / "manifest.json"))
            self.assertEqual(fast["performance"], pair["performance"])
            self.assertNotIn("accumulation_steps", fast["parameters"])
            self.assertEqual(read_json(path), source)
            self.assertFalse((root / "original-run").exists())
            with self.assertRaises(FileExistsError):
                prepare_candidates(path, root / "candidates")
            source["parameters"]["formal_epochs"] = 3
            write_json(path, source)
            with self.assertRaisesRegex(ValueError, "ten-epoch"):
                prepare_candidates(path, root / "short")
            self.assertFalse((root / "short").exists())


class BenchmarkTests(unittest.TestCase):
    def test_request_bounds_and_local_refusal(self):
        for phase, warmup, measured in (("confirm", 2, 10), ("test", 2, 10), ("train", 6, 10),
                                         ("train", 2, 21), ("train", 2, 0), ("train", True, 1)):
            with self.assertRaises(ValueError):
                validate_request(phase, warmup, measured)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            write_json(config, {"schema_version": 2, "recipe": "B03", "stage": "preliminary"})
            with patch("aic_robust_clip.benchmark.prepare", side_effect=AssertionError("artifact read")):
                with self.assertRaises(RuntimeLimitError):
                    benchmark_command(config, "train", root / "result")
            self.assertFalse((root / "result").exists())
            write_json(config, {"schema_version": 2, "recipe": "B03", "stage": "preliminary",
                               "execution_mode": "formal", "device": "cuda"})
            with self.assertRaises(RuntimeLimitError):
                benchmark_command(config, "train", root / "result")
            self.assertFalse((root / "result").exists())

    def test_timer_finite_window_and_no_automatic_continuation(self):
        timer = StepTimer(1, 2, cuda=False)
        with self.assertRaisesRegex(ValueError, "insufficient"):
            timer.report()
        for _ in range(3):
            timer.begin_step()
            timer.data_ready()
            timer.end_step(4)
        report = timer.report()
        self.assertEqual(report["measured_samples"], 8)
        self.assertEqual(report["measured_steps"], 2)
        with self.assertRaisesRegex(RuntimeError, "step limit"):
            timer.begin_step()

    def test_production_trainer_observer_and_checkpoint_quarantine(self):
        run = RunConfig(stage="preliminary", max_updates=2, official_weight_revision="a" * 40,
                        manifest_digest="b" * 64, split_digest="c" * 64, class_map_digest="d" * 64)
        config = TrainConfig(run)
        timer = StepTimer(0, 2, cuda=False)
        model = FrozenFeatureBaseline(4, 2)
        with tempfile.TemporaryDirectory() as directory, \
                StatefulBatchLoader(AddressedFixture(4), sample_ids=[f"s{i}" for i in range(4)], max_samples=4) as loader:
            meta = replace(metadata(config), model_family="BENCHMARK")
            result = train_baseline(model, loader, config=config, class_count=2, device="cpu",
                checkpoint_dir=directory, checkpoint_metadata=meta, observer=timer)
            self.assertEqual((result.updates, timer.steps), (2, 2))
            self.assertEqual(timer.report()["measured_samples"], 2)
            self.assertGreater(timer.checkpoint_seconds, 0)
            with self.assertRaisesRegex(ValueError, "benchmark checkpoints"):
                load_checkpoint(Path(directory) / "last.pt", model=model, requested={})
            from aic_robust_clip.workflow import lock_selection_command
            with patch("aic_robust_clip.workflow.prepare", return_value=object()):
                with self.assertRaisesRegex(ValueError, "cannot be selected"):
                    lock_selection_command("unused", Path(directory) / "last.pt", Path(directory) / "selection.json")
            self.assertFalse((Path(directory) / "selection.json").exists())

    def test_driver_each_phase_is_bounded_and_writes_only_benchmark_reports(self):
        from unittest.mock import Mock
        import aic_robust_clip.benchmark as benchmark
        from aic_robust_clip import workflow
        helpers = workflow_fixtures.WorkflowTests()
        identity = {"model_id": "openai/clip-vit-base-patch32", "revision": "a" * 40,
                    "digest": "b" * 64, "preprocessing_digest": "c" * 64}
        for phase in ("cache", "train", "eval"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory, \
                    patch("aic_robust_clip.configuration.inspect_weights", return_value=identity):
                root = Path(directory)
                value = helpers.fixtures(root)
                write_json(root / "config.json", value)
                ctx = prepare(root / "config.json")
                ctx.config.update(execution_mode="formal", device="cuda", effective_batch_size=1)
                ctx.run = replace(ctx.run, execution_mode="formal")
                ctx.train = TrainConfig(ctx.run)
                ctx.policy = SimpleNamespace(allow_formal=True)
                write_json(root / "config.json", {**value, "execution_mode": "formal", "device": "cuda", "effective_batch_size": 1})
                tiny = AddressedFixture(8)
                tiny.close = lambda: None
                model = Mock()
                calls = []
                def fake_train(_model, _loader, **kwargs):
                    calls.append(kwargs["stop_after_updates"])
                    self.assertNotIn("dev_loader", kwargs)
                    self.assertEqual(kwargs["checkpoint_metadata"].model_family, "BENCHMARK")
                    observer = kwargs["observer"]
                    for _ in range(kwargs["stop_after_updates"]):
                        observer.begin_step()
                        observer.data_ready()
                        observer.end_step(1)
                    return SimpleNamespace(updates=kwargs["stop_after_updates"])
                def fake_features(_loader, _encoder, **kwargs):
                    observer = kwargs["observer"]
                    for index in range(100):
                        observer.begin_step()
                        observer.data_ready()
                        observer.end_step(1)
                        calls.append(index)
                        yield {}, None
                def fake_eval(_model, _loader, **kwargs):
                    for index in range(kwargs["max_batches"]):
                        observer = kwargs["observer"]
                        observer.begin_step()
                        observer.data_ready()
                        observer.end_step(1)
                        calls.append(index)
                with patch.object(benchmark, "prepare", return_value=ctx), \
                        patch.object(benchmark, "StepTimer", side_effect=lambda w, m: StepTimer(w, m, cuda=False)), \
                        patch.object(benchmark, "environment_report", return_value={"synthetic": True}), \
                        patch.object(torch.cuda, "is_available", return_value=True), \
                        patch.object(torch.cuda, "reset_peak_memory_stats"), \
                        patch.object(torch.cuda, "max_memory_allocated", return_value=0), \
                        patch.object(torch.cuda, "max_memory_reserved", return_value=0), \
                        patch.object(workflow, "stream", side_effect=lambda *_args, **_kwargs: fixture_stream(batch=1)), \
                        patch.object(workflow, "training_components", return_value=(model, tiny, tiny, None, None, "a" * 64)), \
                        patch.object(workflow, "load_bundle", return_value=SimpleNamespace(encoder=model, processor=None)), \
                        patch.object(workflow, "dataset_for", return_value=tiny), \
                        patch.object(benchmark, "train_baseline", side_effect=fake_train), \
                        patch.object(benchmark, "feature_batches", side_effect=fake_features), \
                        patch.object(benchmark, "evaluate_loader", side_effect=fake_eval):
                    report = benchmark_command(root / "config.json", phase, root / "result", warmup_steps=1, measure_steps=2)
                    with self.assertRaises(FileExistsError):
                        benchmark_command(root / "config.json", phase, root / "result")
                self.assertEqual(calls, [3] if phase == "train" else [0, 1, 2])
                self.assertEqual(report["measured_samples"], 2)
                self.assertFalse(report["selection_eligible"])
                self.assertTrue((root / "result/benchmark-only.json").is_file())
                self.assertFalse((root / "result/index.json").exists())
                self.assertFalse((root / "result/head.json").exists())

    def test_evaluation_generator_and_observer_bounds(self):
        model = FrozenFeatureBaseline(4, 2)
        with fixture_stream() as loader:
            timer = StepTimer(0, 2, cuda=False)
            _, predictions, labels = evaluate_loader(model, (batch for batch in loader), total_classes=2,
                max_batches=2, device="cpu", observer=timer)
            self.assertEqual((len(predictions), len(labels), loader.position), (4, 4, 4))
            self.assertEqual(timer.report()["measured_samples"], 4)


if __name__ == "__main__":
    unittest.main()
