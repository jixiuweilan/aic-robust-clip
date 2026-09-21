"""Explicit abstention semantics; synthetic inputs and aggregate fit parameters."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from aic_robust_clip.round2 import engine, admission
from aic_robust_clip.round2.methods import MethodError, MethodState, gmm, select, posterior_range_bound


# Aggregate parameters from the reported class325 fit; no original scores/IDs.
FIT = {"means": [4.552528428415668, 4.8417315475888305],
       "variances": [0.0016850378359884349, 0.0688156595763851],
       "weights": [0.09468820900351727, 0.9053117909964825],
       "quantiles": [4.20210599899292, 4.6082823276519775, 4.8134870529174805, 4.994119167327881, 5.44567346572876],
       "status": "converged", "iterations": 378, "max_iter": 1000, "likelihood_delta": 8.921538495326731e-7}
SCORES = np.linspace(FIT["quantiles"][0], FIT["quantiles"][-1], 163)


def fitted_probabilities(values, *, high=False, diagnostics=None):
    """Evaluate the fixed fitted model; this does not pretend to rerun its EM."""
    if len(values) != 163:
        return gmm(values, high=high, diagnostics=diagnostics)
    if diagnostics is not None:
        diagnostics.update(copy.deepcopy(FIT))
    m, v, w = (np.asarray(FIT[key]) for key in ("means", "variances", "weights"))
    logp = np.log(w) - .5 * np.log(2 * np.pi * v) - .5 * (np.asarray(values)[:, None] - m) ** 2 / v
    p = np.exp(logp[:, 0] - np.logaddexp(logp[:, 0], logp[:, 1]))
    return 1 - p if high else p


class AbstentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_analytic_bound_matches_fit_and_handles_endpoint_maxima(self):
        result = posterior_range_bound(FIT)
        self.assertAlmostEqual(result["maximum"], .5548001637962442, places=12)
        self.assertAlmostEqual(result["score_at_maximum"], 4.545269174956387, places=12)
        self.assertFalse(result["threshold_reachable"])
        for high in (False, True):
            bound = posterior_range_bound(FIT, high=high)
            m, v, w = (np.asarray(FIT[key]) for key in ("means", "variances", "weights"))
            x = np.linspace(FIT["quantiles"][0], FIT["quantiles"][-1], 100001)
            logp = np.log(w) - .5 * np.log(2 * np.pi * v) - .5 * (x[:, None] - m) ** 2 / v
            actual = np.exp(logp[:, int(high)] - np.logaddexp(logp[:, 0], logp[:, 1])).max()
            self.assertAlmostEqual(bound["maximum"], actual, places=8)
        equal = {"means": [0., 1.], "variances": [1., 1.], "weights": [.5, .5], "quantiles": [-1., 2.]}
        self.assertEqual(posterior_range_bound(equal)["score_at_maximum"], -1.)
        with self.assertRaises(MethodError):
            posterior_range_bound({**equal, "variances": [0., 1.]})

    def test_strict_default_still_stops_but_explicit_variant_retains_observed(self):
        labels = np.zeros(163, dtype=int)
        original = SCORES.copy()
        with patch("aic_robust_clip.round2.methods.gmm", side_effect=fitted_probabilities):
            with self.assertRaisesRegex(MethodError, "zero selected") as caught:
                select(labels, SCORES, method="turn")
            self.assertFalse(caught.exception.diagnostics["score_range_bound"]["threshold_reachable"])
            keep, probability, report = select(labels, SCORES, method="turn", zero_selection_policy="retain_observed")
            second = select(labels, SCORES, method="turn", zero_selection_policy="retain_observed")
        self.assertTrue(keep.all())
        self.assertTrue((probability < .6).all())
        np.testing.assert_array_equal(probability, fitted_probabilities(SCORES))
        np.testing.assert_array_equal(SCORES, original)
        np.testing.assert_array_equal(keep, second[0])
        self.assertEqual(report, second[2])
        self.assertEqual(report["0"]["reason"], "no_confident_samples")
        self.assertEqual(report["0"]["gmm"]["status"], "converged")
        self.assertEqual((report["0"]["confident_selected"], report["0"]["retained_observed"]), (0, 163))
        self.assertEqual(report["summary"]["abstained_sample_fraction"], 1.)

    def test_mixed_classes_keep_confident_and_retained_counts_separate(self):
        x = np.r_[SCORES, np.linspace(.1, .2, 12), np.linspace(2., 3., 12), [1.] * 4]
        labels = np.r_[[0] * 163, [1] * 24, [2] * 4]
        with patch("aic_robust_clip.round2.methods.gmm", side_effect=fitted_probabilities):
            keep, p, report = select(labels, x, method="turn", zero_selection_policy="retain_observed")
        self.assertTrue(keep[:163].all())
        np.testing.assert_array_equal(keep[163:187], p[163:187] >= .6)
        self.assertTrue(np.isnan(p[187:]).all())
        summary = report["summary"]
        self.assertEqual((summary["confident_selected"], summary["retained_observed"]), (12, 167))
        self.assertEqual((summary["abstained_classes"], summary["abstained_samples"], summary["unfitted_classes"]), (1, 163, 1))
        self.assertEqual(int(keep.sum()), summary["confident_selected"] + summary["retained_observed"])

    def test_numerical_failures_and_unsupported_recipes_never_abstain(self):
        with self.assertRaises(MethodError):
            select([0] * 16, 1 + np.linspace(0, 1e-8, 16), method="turn", zero_selection_policy="retain_observed")
        with patch("aic_robust_clip.round2.methods.gmm", side_effect=MethodError("nonconvergence")):
            with self.assertRaisesRegex(MethodError, "nonconvergence"):
                select([0] * 163, SCORES, method="turn", zero_selection_policy="retain_observed")
        with self.assertRaises(MethodError) as caught:
            select([0] * 8, [np.nan] * 8, method="turn", zero_selection_policy="retain_observed")
        self.assertEqual(caught.exception.diagnostics["nonfinite_count"], 8)
        for method in ("fine", "snscl", "ce"):
            with self.assertRaises(MethodError):
                MethodState(method, ["fixture"], [0], 1, zero_selection_policy="retain_observed")
        with self.assertRaisesRegex(ValueError, "preliminary"):
            engine._run_prepared({"recipe": {"zero_selection_policy": "retain_observed"}}, None)

    def test_policy_state_and_rng_resume_preserve_next_update(self):
        ids = [f"synthetic-{i}" for i in range(163)]
        state = MethodState("turn", ids, [0] * 163, 3, zero_selection_policy="retain_observed")
        with patch("aic_robust_clip.round2.methods.gmm", side_effect=fitted_probabilities):
            state.rescore(ids, SCORES, None, None, completed_epochs=0)
        value = state.state_dict()
        strict = MethodState("turn", ids, [0] * 163, 3)
        with self.assertRaises(MethodError):
            strict.load_state_dict(value)
        with self.assertRaises(MethodError):
            state.load_state_dict(strict.state_dict())
        corrupt = copy.deepcopy(value)
        corrupt["zero_selection_policy"] = "error"
        with self.assertRaises(MethodError):
            state.load_state_dict(corrupt)
        trainer = engine.Trainer(admission.tiny_student("lora"), state, identity={"purpose": "synthetic"}, effective_batch=2)
        batch = {"sample_id": ids[:2], "label_index": torch.zeros(2, dtype=torch.long), "image": torch.randn(2, 3, 32, 32)}
        trainer.update_window([batch], epoch=0, fraction=.5)
        saved = copy.deepcopy(trainer.checkpoint())
        trainer.update_window([batch], epoch=0, fraction=1.)
        restored = engine.Trainer(admission.tiny_student("lora"),
            MethodState("turn", ids, [0] * 163, 3, zero_selection_policy="retain_observed"),
            identity={"purpose": "synthetic"}, effective_batch=2)
        restored.restore(saved)
        restored.update_window([batch], epoch=0, fraction=1.)
        for name, tensor in trainer.student.state_dict().items():
            torch.testing.assert_close(tensor, restored.student.state_dict()[name], atol=0, rtol=0)
        self.assertEqual(restored.state.report, state.report)
        self.assertTrue(restored.state.soft.equal(restored.state.observed))


if __name__ == "__main__":
    unittest.main()
