"""Real-format CUDA/nvidia-smi boundary and complete admission-chain regression."""
import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from uuid import UUID

from aic_robust_clip import preliminary_experiments as exp
from aic_robust_clip.gpu_identity import normalize_gpu_uuid, cuda_gpu_uuids
from aic_robust_clip.contracts import read_json, write_json
from aic_robust_clip.round2 import admission, engine
from aic_robust_clip.round2.config import sealed
import test_preliminary_experiments as fixture_module
from test_preliminary_experiments import RUNTIME, UUID_A, UUID_B, profile, checks

UUIDS = [UUID_A, UUID_B, "d57241f7-49cb-5ea6-d17a-f00457b64b76", "ffaa0f02-4491-b744-1616-40fe0361d058"]


class GPUIdentityTests(unittest.TestCase):
    def test_full_uuid_representations_and_rejections(self):
        for value in (UUID_A, "GPU-" + UUID_A, UUID_A.upper(), UUID(UUID_A)):
            self.assertEqual(normalize_gpu_uuid(value), UUID_A)
        for value in (None, 0, "0", "", "GPU-0", "GPU-c99db39b", "MIG-" + UUID_A,
                      "{" + UUID_A + "}", UUID_A.replace("-", ""), " " + UUID_A, UUID_A + "," + UUID_B):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_gpu_uuid(value)
        for values in ([UUID_A], ["0"], [], ["GPU-" + UUID_A, "GPU-" + UUID_A.upper()]):
            with self.assertRaises(ValueError):
                cuda_gpu_uuids(values)
        self.assertEqual(cuda_gpu_uuids(["GPU-" + UUID_A.upper()]), ["GPU-" + UUID_A])

    def test_real_torch_uuid_object_runtime_and_4060_history(self):
        device = SimpleNamespace(uuid=UUID(UUID_A), name="RTX 4060", total_memory=8000000000)
        with patch.object(admission.torch.cuda, "is_available", return_value=True), \
             patch.object(admission.torch.cuda, "get_device_properties", return_value=device):
            runtime = admission.runtime_identity()
        self.assertEqual(runtime["gpu"]["uuid"], UUID_A)
        history = {"host": runtime["host"], "gpu_uuid": "GPU-" + UUID_A, "original_b04_failure": True,
                   "reviewer": "fixture", "basis": "fixture"}
        prior = {"host": runtime["host"], "gpu_uuid": "GPU-" + UUID_A,
                 "test": "test_verification_cached_only_for_unchanged_file_and_process", "original_evidence_sha256": "a" * 64}
        original = copy.deepcopy((history, prior, runtime))
        exp.check_history(history, prior, runtime)
        self.assertEqual(original, (history, prior, runtime))
        for changed in ("GPU-" + UUID_B, None, "0"):
            with self.assertRaises(ValueError):
                exp.check_history({**history, "gpu_uuid": changed}, prior, runtime)
            with self.assertRaises(ValueError):
                exp.check_history(history, {**prior, "gpu_uuid": changed}, runtime)
        with patch.object(admission.torch.cuda, "is_available", return_value=True), \
             patch.object(admission.torch.cuda, "get_device_properties", return_value=SimpleNamespace(uuid=None)):
            with self.assertRaisesRegex(ValueError, "UUID unavailable"):
                admission.runtime_identity()

    def test_preliminary_admission_to_run_with_bare_runtime_uuid(self):
        # Exercise the real receipt creation and run validation in sequence;
        # hardware and numerical work are mocked, identity handling is not.
        fixture = fixture_module.PreliminaryExperimentsTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        for mismatch in (None, "gpu", "source", "asset", "configuration"):
            root = fixture.root / str(mismatch)
            target = root / "admit"
            runtime = copy.deepcopy(RUNTIME)
            if mismatch == "gpu": runtime["gpu"]["uuid"] = UUID_B
            def child(command, **kwargs):
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "GPU-" + UUID_A)
                output = Path(command[command.index("--output") + 1])
                row = profile("ce", fixture.assets, runtime)
                if mismatch == "asset":
                    row = sealed({k: ("wrong" if k == "asset_digest" else v) for k, v in row.items() if k != "digest"})
                write_json(output / "profile/profile.json", row)
                write_json(output / "checks.json", checks("ce", runtime))
                info = {"assets": fixture.assets.descriptor, "runtime": copy.deepcopy(runtime),
                        "source_sha256": exp.file_sha256(fixture.source)}
                if mismatch == "source": info["runtime"]["source"] = "other-code"
                if mismatch == "configuration": info["source_sha256"] = "wrong"
                write_json(output / "provenance.json", sealed(info))
                write_json(target / "barrier/ce.ready.json", {"uuid": runtime["gpu"]["uuid"]})
                return SimpleNamespace(pid=123, returncode=0, poll=lambda: 0)
            with patch.object(engine, "require_machine"), patch.object(exp.subprocess, "Popen", side_effect=child):
                if mismatch:
                    with self.assertRaisesRegex(ValueError, "identity mismatch"):
                        exp.admit_t4(fixture.source, target, methods=["ce"], gpu_uuids=["GPU-" + UUID_A], owner="fixture")
                    self.assertFalse((target / "admission.json").exists())
                    continue
                receipt = exp.admit_t4(fixture.source, target, methods=["ce"], gpu_uuids=["GPU-" + UUID_A], owner="fixture")
            self.assertEqual(receipt["gpu_uuids"], ["GPU-" + UUID_A])
            with patch.object(exp, "source_context", return_value=({}, fixture.assets, runtime)), \
                 patch.object(engine, "_run_prepared", return_value={"best_metrics": {"macro_recall": .5}, "last": {"macro_recall": .5}}) as numerical, \
                 patch.object(engine, "_replay_prepared", return_value={"passed": True}):
                exp.run(target / "admission.json", root / "run", method="ce")
                self.assertEqual(numerical.call_count, 1)
                self.assertEqual(numerical.call_args.args[0]["recipe"]["epochs"], 30)

    def test_invalid_assignment_rejected_before_process_or_machine_access(self):
        with patch.object(exp.subprocess, "Popen", side_effect=AssertionError("spawned")), \
             patch.object(engine, "require_machine", side_effect=AssertionError("machine")):
            for values in ([UUID_A], ["0"], ["GPU-short"], [None]):
                with self.assertRaises(ValueError):
                    exp.admit_t4("missing", "missing", methods=["ce"], gpu_uuids=values, owner="fixture")

    def test_round2_concurrent_bare_uuid_and_swapped_card_rejection(self):
        for swapped in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "second_round"
                choice = root / "choice.json"
                write_json(choice, sealed({"group": "t4", "engineering": {"microbatch": 4, "workers": 2}}))
                output = root / "concurrent"
                def child(command, **kwargs):
                    method = command[command.index("--method") + 1]
                    index = admission.methods_for("t4").index(method)
                    uuid = UUIDS[(1 - index) if swapped and index < 2 else index]
                    target = Path(command[command.index("--output") + 1])
                    write_json(target / "profile.json", sealed({"method": method, "runtime": {"gpu": {"uuid": uuid}},
                        "train_window_started": 1., "train_window_ended": 2.}))
                    write_json(output / "barrier" / f"{method}.ready.json", {"uuid": uuid})
                    return SimpleNamespace(poll=lambda: 0, wait=lambda: 0)
                with patch.object(engine, "require_machine"), patch.object(admission.subprocess, "Popen", side_effect=child), \
                     patch.object(admission, "current_code_revision", return_value="fixture-source"), \
                     patch.object(admission, "load_assets", return_value=SimpleNamespace(descriptor={"digest": "assets"})):
                    arguments = dict(machine="fixture", choice_path=choice, gpu_uuids=["GPU-" + u for u in UUIDS], output=output)
                    if swapped:
                        with self.assertRaisesRegex(ValueError, "assignment mismatch"):
                            admission.concurrent("fixture", **arguments)
                    else:
                        self.assertEqual(admission.concurrent("fixture", **arguments)["status"], "passed")


if __name__ == "__main__":
    unittest.main()
