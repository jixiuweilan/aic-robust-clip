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
from test_performance import fixture_stream


class CountedImages:
    def __init__(self, count=96):
        self.count = count
        self.reads = multiprocessing.get_context("spawn").Value("i", 0)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        with self.reads.get_lock():
            self.reads.value += 1
        return SampleItem(torch.full((3, 64, 64), float(index)), f"s{index}", "0", 0)


class LoaderLifecycleTests(unittest.TestCase):
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
