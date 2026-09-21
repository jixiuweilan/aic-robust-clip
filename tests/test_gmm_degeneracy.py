"""Deterministic score fixtures only; no images, training or real losses."""
import copy
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from aic_robust_clip.contracts import read_json
from aic_robust_clip.round2 import admission, engine, config
from aic_robust_clip.round2.methods import MethodError, MethodState, gmm, select


class GMMTests(unittest.TestCase):
    def test_repeated_scores_break_only_tied_quantile_initialization(self):
        x = np.r_[np.zeros(20), np.ones(2)]
        original = x.copy()
        details = {}
        p = gmm(x, diagnostics=details)
        np.testing.assert_array_equal(x, original)
        np.testing.assert_array_equal(p, np.r_[np.ones(20), np.zeros(2)])
        self.assertEqual(details["initialization"], "endpoints_for_tied_quartiles")
        self.assertEqual(details["unique_count"], 2)
        repeated_details = {}
        np.testing.assert_array_equal(p, gmm(x, diagnostics=repeated_details))
        self.assertEqual(details, repeated_details)
        # Repeated observations retain their statistical weight.
        np.testing.assert_allclose(details["weights"], [20 / 22, 2 / 22])
        normal = {}
        gmm(np.r_[np.zeros(12), np.ones(12)], diagnostics=normal)
        self.assertEqual(normal["initialization"], "quartiles")

    def test_near_constant_is_not_reported_as_confident_or_unfiltered(self):
        x = 1 + np.linspace(0, 1e-8, 16)
        for method in ("turn", "snscl"):
            with self.assertRaises(MethodError) as caught:
                select(np.zeros(16, dtype=int), x, method=method)
            details = caught.exception.diagnostics
            self.assertEqual(details["reason"], "unordered_components")
            self.assertTrue(details["variance_below_floor"])
            self.assertEqual(details["status"], "failed")
            self.assertEqual(details["method"], method)

    def test_short_classes_and_constant_keep_existing_explicit_policy(self):
        keep, p, report = select([0] * 7 + [1] * 8, np.r_[np.arange(7), np.ones(8)], method="turn")
        self.assertTrue(keep.all())
        self.assertTrue(np.isnan(p).all())
        self.assertEqual(report["0"]["reason"], "fewer_than_8")
        self.assertEqual(report["1"]["reason"], "constant")
        with self.assertRaises(MethodError):
            gmm(np.arange(7))
        with self.assertRaises(MethodError):
            select([0] * 8, np.ones(8), method="snscl")

    def test_last_update_is_evaluated_and_short_budget_is_honest(self):
        x = np.r_[np.linspace(.1, .2, 12), np.linspace(2, 3, 12)]
        details = {}
        p = gmm(x, diagnostics=details)
        iterations = details["iterations"]
        np.testing.assert_array_equal(p, gmm(x, max_iter=iterations))
        with self.assertRaisesRegex(MethodError, f"{iterations - 1} iterations") as caught:
            gmm(x, max_iter=iterations - 1)
        self.assertEqual(caught.exception.diagnostics["iterations"], iterations - 1)
        self.assertEqual(caught.exception.diagnostics["reason"], "nonconvergence")
        with self.assertRaisesRegex(MethodError, "1 iterations"):
            gmm(x, max_iter=1)

    def test_normal_bimodal_posteriors_threshold_and_determinism(self):
        x = np.r_[np.linspace(.1, .2, 12), np.linspace(2, 3, 12)]
        a, b = {}, {}
        p = gmm(x, diagnostics=a)
        np.testing.assert_array_equal(p, gmm(x, diagnostics=b))
        self.assertEqual(a, b)
        np.testing.assert_allclose(p + gmm(x, high=True), np.ones(24))
        self.assertTrue((p[:12] > .99).all())
        self.assertTrue((p[12:] < .01).all())
        keep, probability, _ = select(np.zeros(24, dtype=int), x, method="turn")
        np.testing.assert_array_equal(keep, probability >= .6)
        order = np.random.default_rng(17).permutation(24)
        np.testing.assert_allclose(gmm(x[order]), p[order], atol=1e-12, rtol=1e-12)

    def test_regular_scores_still_fail_after_full_budget_without_fallback(self):
        x = np.random.default_rng(0).normal(2, .4, 64)
        for _ in range(2):
            with self.assertRaisesRegex(MethodError, "100 iterations") as caught:
                gmm(x, max_iter=100)
            details = caught.exception.diagnostics
            self.assertEqual((details["reason"], details["iterations"]), ("nonconvergence", 100))
            self.assertEqual(details["initialization"], "quartiles")
            self.assertGreater(abs(details["likelihood_delta"]), 1e-6)

    def test_slow_convergence_170_samples_167_unique_single_fit(self):
        # Synthetic shape-matched fixture, NOT the unavailable server scores.
        # Seed17 and three exact duplicates give 170 rows / 167 unique scores.
        rng = np.random.default_rng(17)
        x = np.r_[rng.normal(4.117420683496456, np.sqrt(.05493894527176942), 130),
                  rng.normal(4.560348192428134, np.sqrt(.1636713450248558), 40)]
        x[-3:] = x[:3]
        x = (x - x.mean()) * np.sqrt(.11615313786018694 / x.var()) + 4.22
        self.assertEqual((len(x), len(np.unique(x))), (170, 167))
        self.assertAlmostEqual(x.var(), .11615313786018694)
        with self.assertRaisesRegex(MethodError, "100 iterations") as caught:
            gmm(x, max_iter=100)
        previous = caught.exception.diagnostics
        self.assertGreater(previous["likelihood_delta"], 1e-6)
        self.assertLess(previous["likelihood_delta"], 1.6e-6)
        details = {}
        p = gmm(x, diagnostics=details)
        self.assertEqual(details["max_iter"], 1000)
        self.assertGreater(details["iterations"], 100)
        self.assertLessEqual(details["iterations"], 1000)
        self.assertLessEqual(abs(details["likelihood_delta"]), 1e-6)
        self.assertEqual(details["initial_means"], previous["initial_means"])
        self.assertEqual(details["tolerance"], previous["tolerance"])
        self.assertEqual(details["variance_floor"], previous["variance_floor"])
        again = {}
        np.testing.assert_array_equal(p, gmm(x, diagnostics=again))
        self.assertEqual(details, again)
        # TURN performs one fit with the declared budget, no retry loop.
        with patch("aic_robust_clip.round2.methods.gmm", wraps=gmm) as fit:
            keep, probability, report = select(np.full(170, 10), x, method="turn")
        fit.assert_called_once()
        np.testing.assert_array_equal(keep, p >= .6)
        np.testing.assert_array_equal(probability, p)
        self.assertEqual(report["10"]["gmm"]["iterations"], details["iterations"])

    def test_iteration_budget_is_bound_to_plan_and_pilot_recipe(self):
        from aic_robust_clip.preliminary_pilot import pilot_config
        self.assertEqual(config.GMM_MAX_ITERATIONS, 1000)
        self.assertEqual(config.RECIPE["gmm"], {"iterations": 1000, "tolerance": 1e-6, "variance_floor": 1e-6, "threshold": .6})
        plan = read_json(Path(__file__).resolve().parents[1] / "configs/second_round/plan.json")
        self.assertEqual(plan["recipe"], config.RECIPE)
        assets = SimpleNamespace(descriptor={"digest": "synthetic"}, head_sha256="synthetic")
        pilot = pilot_config(Path("/tmp/preliminary/synthetic"), assets, {}, {"digest": "checks"}, {"digest": "profile"})
        self.assertEqual(pilot["recipe"]["gmm"], config.RECIPE["gmm"])
        self.assertEqual((pilot["recipe"]["epochs"], pilot["recipe"]["pause_after_epoch"]), (30, 10))
        for invalid in (0, -1, 1001, True, 1000.0):
            with self.assertRaisesRegex(MethodError, "iteration|integer"):
                gmm(np.arange(8), max_iter=invalid)

    def test_failure_statistics_are_finite_json_and_do_not_contain_arrays(self):
        for x in (np.full(8, np.nan), np.array([1e308, -1e308] * 4), 1 + np.linspace(0, 1e-8, 16)):
            with self.assertRaises(MethodError) as caught:
                gmm(x)
            encoded = json.dumps(caught.exception.diagnostics, allow_nan=False)
            self.assertNotIn('"scores"', encoded)
            self.assertNotIn('"sample_ids"', encoded)
            self.assertNotIn('"posterior"', encoded)
            self.assertNotIn('"losses"', encoded)

    def test_profile_preserves_real_method_failure_and_closes_loaders(self):
        # Execute the actual profile->MethodState->select->gmm failure path;
        # only CUDA/model/loader access is replaced by inert fixture objects.
        from unittest.mock import Mock
        x = np.random.default_rng(0).normal(2, .4, 64)
        records = [SimpleNamespace(sample_id=f"fixture-{i}", class_id="0000") for i in range(64)]
        datasets = [SimpleNamespace(records=records, class_to_index={"0000": 0}, close=Mock()) for _ in range(3)]
        assets = SimpleNamespace(dataset=Mock(side_effect=datasets), class_map=SimpleNamespace(id_to_index={"0000": 0}))
        scored = ([r.sample_id for r in records], x, None, None)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(engine, "require_machine"), patch.object(admission, "runtime_identity", return_value={"fixture": True}), \
             patch.object(engine, "Trainer"), patch.object(engine, "clock", return_value=0), \
             patch.object(engine.torch.cuda, "reset_peak_memory_stats"), \
             patch.object(engine, "loader_for", return_value=nullcontext(object())), \
             patch.object(engine, "scoring", return_value=scored), \
             patch("aic_robust_clip.round2.methods.gmm", side_effect=lambda x, **kw: gmm(x, max_iter=100, **kw)):
            # Deliberately short test budget exercises the real failure path;
            # production uses one fit with the fixed 1000-update budget.
            root = Path(directory) / "preliminary/profile"
            with self.assertRaises(MethodError):
                admission._profile_one(None, method="turn", adaptation="lora", microbatch=32, workers=2,
                    machine=None, output=root, _assets=assets, _student_factory=lambda *args: (None, None, {}), _stage="preliminary")
            failure = read_json(root / "failure.json")
            self.assertTrue(failure["method_failure"])
            self.assertFalse(failure["auto_retry"])
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(failure["method_diagnostics"]["class_index"], 0)
            self.assertEqual(failure["method_diagnostics"]["count"], 64)
            self.assertEqual(failure["method_diagnostics"]["reason"], "nonconvergence")
            self.assertEqual(failure["method_diagnostics"]["iterations"], 100)
            self.assertFalse((root / "profile.json").exists())
            for dataset in datasets:
                dataset.close.assert_called_once()

    def test_rescore_failure_preserves_selection_and_failure_receipts(self):
        state = MethodState("turn", [f"fixture-{i}" for i in range(16)], [0] * 16, 1)
        before = copy.deepcopy(state.state_dict())
        with self.assertRaises(MethodError) as caught:
            state.rescore(state.ids, 1 + np.linspace(0, 1e-8, 16), None, None, completed_epochs=1)
        self.assertEqual(state.completed_epochs, 0)
        self.assertTrue(state.selected.equal(before["selected"]))
        self.assertEqual(state.report, before["report"])
        with tempfile.TemporaryDirectory() as directory:
            engine.record_failure(directory, caught.exception)
            engine.record_failure(directory, MethodError("another failure"))
            first = read_json(Path(directory) / "failure.json")
            self.assertEqual(first, read_json(Path(directory) / "failure-001.json"))
            self.assertEqual(first["method_diagnostics"]["class_index"], 0)
            self.assertTrue((Path(directory) / "failure-002.json").is_file())


if __name__ == "__main__":
    unittest.main()
