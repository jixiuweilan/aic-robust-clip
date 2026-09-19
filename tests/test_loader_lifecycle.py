"""Bounded CPU fixtures: queue shutdown and train/eval worker isolation."""
import multiprocessing
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from aic_robust_clip.data.cache import CachedDataset
from aic_robust_clip.data.dataset import DatasetError, SampleItem
from aic_robust_clip.data.loading import StatefulBatchLoader
from aic_robust_clip.performance import PerformanceConfig
from aic_robust_clip.workflow import stream
from aic_robust_clip.training.baseline import evaluate_loader
from test_performance import fixture_stream


class CountedImages:
    def __init__(self, count=96, shape=(3, 64, 64)):
        self.count = count
        self.shape = shape
        self.reads = multiprocessing.get_context("spawn").Value("i", 0)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        with self.reads.get_lock():
            self.reads.value += 1
        return SampleItem(torch.full(self.shape, float(index)), f"s{index}", "0", 0)


class LoaderLifecycleTests(unittest.TestCase):
    def test_bounded_eval_uses_real_batch64_shape_and_leaves_no_pending_reads(self):
        # Two forward-only CPU fixture batches, never real images or a full epoch.
        dataset = CountedImages(count=512, shape=(3, 224, 224))
        model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 1))
        with StatefulBatchLoader(dataset, sample_ids=[f"s{i}" for i in range(len(dataset))],
                batch_size=64, num_workers=2, prefetch_factor=2, shuffle=False) as loader:
            loader.limit_dispatch(2)
            self.assertEqual(len(loader), 8)  # preserve full-epoch scheduler budget
            _, predictions, _ = evaluate_loader(model, loader, total_classes=1,
                max_batches=2, device="cpu")
            self.assertEqual(len(predictions), 128)
            self.assertEqual(dataset.reads.value, 128)
            with self.assertRaises(StopIteration):
                next(loader)
            state = loader.state_dict()
        self.assertEqual(dataset.reads.value, 128)
        self.assertEqual(state, loader.state_dict())
        self.assertNotIn("dispatch_stop", state)  # persisted format unchanged
        self.assertEqual(len(loader.diagnostics()["worker_exits"]), 2)
        self.assertTrue(all(w["exitcode"] == 0 for w in loader.diagnostics()["worker_exits"]))

    def test_dispatch_limit_validation_and_serial_parallel_parity(self):
        for value in (0, -1, True, 1.5):
            with fixture_stream() as loader, self.assertRaises(DatasetError):
                loader.limit_dispatch(value)
        with fixture_stream() as serial, fixture_stream(workers=1) as parallel:
            serial.limit_dispatch(2)
            parallel.limit_dispatch(2)
            self.assertEqual([b["sample_id"] for b in serial], [b["sample_id"] for b in parallel])
            with self.assertRaisesRegex(DatasetError, "before reading"):
                parallel.limit_dispatch(3)

    def test_late_aborted_worker_exit_cannot_be_reported_as_success(self):
        loader = fixture_stream(workers=1)
        worker = Mock(pid=321, exitcode=-6)
        worker.is_alive.return_value = False
        iterator = Mock()
        iterator._workers = [worker]
        loader._iterator = iterator
        with patch("aic_robust_clip.data.loading.next", side_effect=StopIteration, create=True):
            with self.assertRaisesRegex(DatasetError, "abnormal.*-6"):
                loader.close()
        self.assertEqual(loader.diagnostics()["worker_exits"][0]["exitcode"], -6)
        with self.assertRaisesRegex(DatasetError, "no retry"):
            next(loader)
        loader.close()  # repeat cleanup is harmless, not a retry

    def test_worker_read_error_is_terminal_even_if_caller_requests_next_again(self):
        with fixture_stream(workers=1, fail=0) as loader:
            loader.shuffle = False
            loader.reset(0)
            with self.assertRaisesRegex(ValueError, "synthetic worker failure"):
                next(loader)
            with self.assertRaisesRegex(DatasetError, "no retry"):
                next(loader)

    def test_partial_close_drains_only_dispatched_images_without_advancing_cursor(self):
        for _ in range(2):
            dataset = CountedImages()
            loader = StatefulBatchLoader(dataset, sample_ids=[f"s{i}" for i in range(len(dataset))],
                batch_size=8, num_workers=2, prefetch_factor=2, shuffle=False)
            try:
                next(loader)
                state = loader.state_dict()
                workers = list(loader._iterator._workers)
                loader.close()
                self.assertEqual(loader.state_dict(), state)
                self.assertEqual(dataset.reads.value, 8 + 2 * 2 * 8)
                self.assertTrue(all(not w.is_alive() and w.exitcode == 0 for w in workers))
                loader.close()  # idempotent
            finally:
                loader.close()

    def test_pending_read_error_is_not_hidden_by_close(self):
        loader = fixture_stream(workers=1, batch=1, count=20, fail=1)
        loader.shuffle = False
        loader.reset(0)
        next(loader)
        workers = list(loader._iterator._workers)
        with self.assertRaisesRegex(ValueError, "synthetic worker failure"):
            loader.close()
        self.assertEqual(loader.position, 1)
        self.assertIsNone(loader._parallel)
        self.assertTrue(all(not w.is_alive() for w in workers))

    def test_model_error_survives_pending_read_error(self):
        loader = fixture_stream(workers=1, batch=1, count=20, fail=1)
        loader.shuffle = False
        loader.reset(0)
        with self.assertRaisesRegex(RuntimeError, "model failed") as caught:
            with loader:
                next(loader)
                raise RuntimeError("model failed")
        self.assertIn("synthetic worker failure", " ".join(caught.exception.__notes__))
        self.assertIsNone(loader._parallel)

    def test_broken_iterator_is_not_drained(self):
        loader = fixture_stream(workers=1)
        iterator = Mock()
        iterator._workers = []
        loader._iterator = iterator
        loader._worker_failed = True
        loader.close()
        iterator._shutdown_workers.assert_called_once()

    def test_pending_drain_has_hard_batch_bound(self):
        loader = fixture_stream(workers=1)
        iterator = Mock()
        iterator._workers = []
        loader._iterator = iterator
        with patch("aic_robust_clip.data.loading.next", return_value={}, create=True) as advance:
            with self.assertRaisesRegex(DatasetError, "pending-batch bound"):
                loader.close()
        self.assertEqual(advance.call_count, 3)
        iterator._shutdown_workers.assert_called_once()

    def test_workflow_routes_train_eval_scoring_and_cached_workers(self):
        perf = PerformanceConfig.from_config({"execution_mode": "formal", "batch_size": 16,
            "performance": {"num_workers": 4, "eval_num_workers": 0, "eval_batch_size": 64}})
        ctx = SimpleNamespace(performance=perf,
            run=SimpleNamespace(batch_size=16, seed=17, execution_mode="formal"))
        with patch("aic_robust_clip.workflow.StatefulBatchLoader") as factory:
            for purpose, batch, workers in (("train", 16, 4), ("dev", 64, 0),
                    ("confirm", 64, 0), ("scoring", 64, 0)):
                stream(ctx, SimpleNamespace(purpose=purpose))
                self.assertEqual(factory.call_args.kwargs["num_workers"], workers)
                self.assertEqual(factory.call_args.kwargs["batch_size"], batch)
            stream(ctx, SimpleNamespace(purpose="train"), batch_size=64)  # cache benchmark
            self.assertEqual(factory.call_args.kwargs["num_workers"], 4)
            cached = Mock(spec=CachedDataset)
            cached.purpose = "train"
            stream(ctx, cached)
            self.assertEqual(factory.call_args.kwargs["num_workers"], 0)


if __name__ == "__main__":
    unittest.main()
