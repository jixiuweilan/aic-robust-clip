"""Generated images/tiny random CLIP only; no downloads or real training."""
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import CLIPImageProcessor
from aic_robust_clip.contracts import read_json, write_json
from aic_robust_clip.round2 import admission, assets, config, delivery, engine
from aic_robust_clip.data import audit as audit_module
from aic_robust_clip.models.provision import file_sha256
from aic_robust_clip.runtime import current_code_revision
import test_round2 as fixtures

UUID = "c99db39b-3ee0-d3f4-2639-1141aa73f05d"


def evidence(root, a, runtime, *, group="4060-a", methods=("ce", "turn")):
    methods = list(methods)
    adaptation = config.adaptation_for(group)
    check = root / "checks"
    check.mkdir()
    for index in range(4):
        (check / f"check-{index}.log").write_text("synthetic evidence fixture\n")
    write_json(check / "checks.json", config.sealed({"version": config.VERSION, "kind": "checks", "group": group,
        "methods": methods,
        "device": "cuda", "runtime": runtime, "suite": {"tests": 1, "errors": 0, "failures": 0, "skips": 0},
        "returncodes": [0] * 4, "startup": [{"method": m, "precision": p, "status": "passed", "updates": 2}
            for m in methods for p in ("fp32", "fp16")], "previous_failure": None,
        "machine_history": ({"host": runtime["host"], "gpu_uuid": "GPU-" + UUID, "original_b04_failure": False,
                            "reviewer": "synthetic", "basis": "generated fixture"} if group == "4060-a" else None),
        "logs": {f"check-{i}.log": file_sha256(check / f"check-{i}.log") for i in range(4)}}))
    identity = {"version": config.VERSION, "status": "passed", "runtime": runtime,
                "asset_digest": a.descriptor["digest"], "head_sha256": config.head_descriptor(a)["sha256"], "adaptation": adaptation}
    paths = []
    for name, grid in (("grid", admission.GRID), ("final", [(4, 2)])):
        entries = []
        for method in methods:
            for b, w in grid:
                path = root / name / f"{method}-{b}-{w}.json"
                write_json(path, config.sealed({**identity, "kind": "profile", "method": method, "microbatch": b, "workers": w,
                    "peak_occupied_bytes": 80, "total_bytes": 100, "projected_epoch_seconds": 10.,
                    "eval_seconds": [1.] * (3 if name == "final" else 1), "dev_student_replay": True, "workers_closed": True, "train_window_started": 2.}))
                entries.append({"path": str(path), "sha256": file_sha256(path)})
        path = root / name / "profiles.json"
        write_json(path, config.sealed({"version": config.VERSION, "kind": "profile_grid", "rows": entries}))
        paths.append(path)
    feasible = []
    for method in methods:
        path = root / f"feasibility-{method}.json"
        write_json(path, config.sealed({**identity, "kind": "feasibility", "method": method, "recipe": config.RECIPE,
            "scoring_samples": 30, "samples": 256, "updates": 2, "checkpoint_created": False, "completed_at": 1.}))
        feasible.append(path)
    return dict(group=group, owner="synthetic", check_paths=[check / "checks.json"], methods=methods,
                profile_paths=[paths[0]], final_paths=[paths[1]], feasibility_paths=feasible)


class Round2DeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_interrupted_audit_and_wrong_registered_hash_cannot_publish_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "second_round"
            root.mkdir()
            path, identity = fixtures.AssetConfigTests().fixtures(root)
            original = (path.parent / "manifest.json").read_bytes()
            kwargs = dict(source_url="https://organizer.invalid/synthetic", retrieved_at="synthetic",
                          organizer_version="synthetic", weights="unused", weight_revision="unused")
            with patch.object(assets, "inspect_weights", return_value=identity):
                with patch.object(audit_module, "_decode_pixels", side_effect=KeyboardInterrupt("synthetic interruption")):
                    with self.assertRaises(KeyboardInterrupt):
                        assets.audit(root / "train.zip", root / "interrupted", **kwargs)
                with patch.object(audit_module, "_decode_pixels", side_effect=AssertionError("decoded before SHA check")):
                    with self.assertRaisesRegex(ValueError, "registered SHA256"):
                        assets.audit(root / "train.zip", root / "wrong-hash", expected_sha256="0" * 64, **kwargs)
            for name in ("interrupted", "wrong-hash"):
                status = read_json(root / name / "status.json")
                self.assertEqual((status["status"], status["exit_code"]), ("failed", 1))
                self.assertFalse((root / name / "assets.json").exists())
                self.assertTrue((root / name / "events.jsonl").is_file())
            self.assertEqual(original, (path.parent / "manifest.json").read_bytes())
            write_json(path.parent / "status.json", {"status": "running", "exit_code": None})
            with self.assertRaisesRegex(ValueError, "interrupted"):
                config.load_assets(path)

    def test_detached_failure_has_durable_exit_code_and_never_reports_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "second_round"
            root.mkdir()
            cfg = root / "bad-config.json"
            write_json(cfg, {"run_id": "4060-A-CE", "method": "ce"})
            job = {"schema": delivery.SCHEMA, "stage": "second_round", "action": "train", "run_id": "4060-A-CE",
                   "config": str(cfg), "output": str(root / "job"), "gpu_uuid": "GPU-" + UUID,
                   "machine": str(root / "missing-machine.json")}
            write_json(root / "request.json", job)
            result = delivery.launch(root / "request.json")
            self.assertEqual(result["status"], "launched_not_completed")
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                value = delivery.status(root / "job")
                if value["status"] == "failed":
                    break
                time.sleep(.1)
            self.assertEqual(value["status"], "failed")
            for child in list(delivery._children):
                child.wait(timeout=10)
            delivery.status(root / "job")
            self.assertFalse(value["completion_verified"])
            self.assertEqual(read_json(root / "job/01-epoch-1.json")["exit_code"], 1)
            self.assertTrue((root / "job/01-epoch-1.log").is_file())
            with self.assertRaises(FileExistsError):
                delivery.launch(root / "request.json")
            with self.assertRaises(FileExistsError):
                delivery.execute(root / "job/job.json")
            with self.assertRaises(ValueError):
                delivery.validate_job({**job, "stage": "preliminary"})
            with self.assertRaises(ValueError):
                delivery.validate_job({**job, "schema": "preliminary-pilot"})

    def test_real_admission_relocation_head_load_config_first_epoch_resume_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "second_round"
            root.mkdir()
            path, identity = fixtures.AssetConfigTests().fixtures(root)
            target = root / "machine-b"
            target.mkdir()
            shutil.copytree(path.parent, target / "assets")
            moved = target / "train.zip"
            (root / "train.zip").rename(moved)
            mapping = root / "locations.json"
            write_json(mapping, {"schema_version": 1, "reason": "synthetic cross machine test", "archives": {
                str(root / "train.zip"): {"path": str(moved), "sha256": file_sha256(moved)}}})
            path = target / "assets/assets.json"
            manifest_bytes = (path.parent / "manifest.json").read_bytes()
            runtime = {"host": "synthetic-host", "source": current_code_revision(), "dependencies": {"torch": "synthetic"},
                       "gpu": {"uuid": UUID, "name": "NVIDIA GeForce RTX 4090", "total_bytes": 100}}
            with patch.dict(os.environ, {"AIC_ARCHIVE_LOCATIONS": str(mapping)}), \
                 patch.object(config, "inspect_weights", return_value=identity), \
                 patch.object(admission, "runtime_identity", return_value=runtime):
                a = config.load_assets(path, verify_archives=True)
                initial = admission.tiny_student("full_visual", image_size=224)
                state = copy.deepcopy(initial.encoder.clip_model.state_dict())
                digest = engine.atomic_save(initial.classifier.state_dict(), a.root / "HEAD20-GCE/head.pt")
                write_json(a.root / "HEAD20-GCE/head.json", {"identity": a.head_identity(), "sha256": digest})
                kwargs = evidence(root, a, runtime, group="cloud4090-full")
                with self.assertRaisesRegex(ValueError, "feasibility"):
                    admission.admit(path, output=root / "bad-admission.json", **{**kwargs, "feasibility_paths": []})
                receipt = admission.admit(path, output=root / "admission.json", **kwargs)
                # Physical assignment in nvidia-smi form; runtime is bare UUID.
                receipt = config.sealed({**config.verify_seal(receipt), "assignments": {"ce": "GPU-" + UUID.upper(), "turn": UUID}})
                write_json(root / "admission.json", receipt)
                config.prepare_configs(root / "configs", path, root / "admission.json")
                cfg_path = root / "configs/C4090-FULL-CE.json"
                cfg, _ = config.check_config(cfg_path)
                def bundle(_):
                    fresh = admission.tiny_student("full_visual", image_size=224)
                    fresh.encoder.clip_model.load_state_dict(state)
                    return SimpleNamespace(encoder=fresh.encoder, processor=CLIPImageProcessor())
                actual_trainer = engine.Trainer
                def cpu_trainer(*args, **kwargs):
                    kwargs.update(device="cpu", precision="fp32")
                    return actual_trainer(*args, **kwargs)
                with patch.object(engine, "require_machine"), patch.object(engine, "bundle_for", side_effect=bundle), \
                     patch.object(engine, "Trainer", side_effect=cpu_trainer), \
                     patch.object(torch.cuda, "is_available", return_value=False), \
                     patch.object(torch.cuda, "max_memory_reserved", return_value=0):
                    # No patch to admit/check_config/student_for/head loading or archive loader.
                    result = engine.run(cfg_path, machine="synthetic", stop_after_epoch=1)
                    self.assertEqual(result["status"], "paused_at_observation")
                    delivery.verify_observation(cfg, 1)
                    checkpoint = torch.load(Path(cfg["output"]) / "last.pt", weights_only=False)
                    self.assertEqual(checkpoint["scheduler"], {"total_epochs": 30, "completed_epochs": 1})
                    engine.run(cfg_path, machine="synthetic", resume=True)
                    delivery.verify_observation(cfg, 10)
                    replay = engine._replay_prepared(cfg, a, output=root / "replay", device="cpu")
                    self.assertTrue(replay["passed"])
                    with self.assertRaisesRegex(ValueError, "epoch10"):
                        engine.run(cfg_path, machine="synthetic", resume=True)
                self.assertEqual(manifest_bytes, (path.parent / "manifest.json").read_bytes())
                with patch.object(admission, "runtime_identity", return_value={**runtime, "host": "other-machine"}):
                    with self.assertRaisesRegex(ValueError, "machine/GPU"):
                        config.check_config(cfg_path)

    def test_cloud4090_ce_only_admission_does_not_release_turn(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "second_round"
            root.mkdir()
            path, identity = fixtures.AssetConfigTests().fixtures(root)
            runtime = {"host": "synthetic-host", "source": current_code_revision(), "dependencies": {"torch": "synthetic"},
                       "gpu": {"uuid": UUID, "name": "NVIDIA GeForce RTX 4090", "total_bytes": 100}}
            with patch.object(config, "inspect_weights", return_value=identity), \
                 patch.object(admission, "runtime_identity", return_value=runtime):
                a = config.load_assets(path, verify_archives=True)
                (a.root / "HEAD20-GCE").mkdir()
                digest = engine.atomic_save({"head": torch.ones(1)}, a.root / "HEAD20-GCE/head.pt")
                write_json(a.root / "HEAD20-GCE/head.json", {"identity": a.head_identity(), "sha256": digest})
                kwargs = evidence(root, a, runtime, group="cloud4090-lora", methods=("ce",))
                receipt = admission.admit(path, output=root / "admission.json", **kwargs)
                self.assertEqual(receipt["assignments"], {"ce": UUID})
                config.prepare_configs(root / "configs", path, root / "admission.json")
                ce, _ = config.check_config(root / "configs/C4090-LORA-CE.json")
                self.assertEqual(ce["status"], "ready")
                turn, _ = config.check_config(root / "configs/C4090-LORA-TURN.json", ready=False)
                self.assertEqual(turn["status"], "blocked_on_machine_admission")
                with self.assertRaisesRegex(ValueError, "blocked_on_machine_admission"):
                    config.check_config(root / "configs/C4090-LORA-TURN.json")
            job = {"schema": delivery.SCHEMA, "stage": "second_round", "action": "group",
                   "group": "cloud4090-lora", "methods": ["ce"], "owner": "synthetic",
                   "gpu_uuids": ["GPU-" + UUID], "machine": str(root / "machine.json"),
                   "assets": str(path), "output": str(root / "job")}
            delivery.validate_job(job)
            with self.assertRaisesRegex(ValueError, "CE alone"):
                delivery.validate_job({**job, "methods": ["turn"]})
            with self.assertRaisesRegex(ValueError, "GPU"):
                delivery.validate_job({**job, "gpu_uuids": ["GPU-" + UUID, "GPU-" + UUID]})

    def test_real_feasibility_snscl_active_updates_never_save_weights(self):
        from aic_robust_clip.round2.methods import MethodState
        from aic_robust_clip.data.dataset import SampleItem
        class Record(SimpleNamespace):
            def to_dict(self):
                return vars(self)
        records = [Record(sample_id=f"synthetic-{i}", class_id=str(i % 3)) for i in range(300)]
        generator = torch.Generator().manual_seed(17)
        images = torch.randn(300, 3, 32, 32, generator=generator)
        class Dataset:
            stage, role, partition = "second_round", "train", "train"
            class_to_index = {str(i): i for i in range(3)}
            def __init__(self, purpose):
                self.records, self.purpose = records, purpose
            def __len__(self):
                return len(self.records)
            def __getitem__(self, index):
                row = self.records[index]
                return SampleItem(image=images[int(row.sample_id.split("-")[-1])], sample_id=row.sample_id,
                                  class_id=row.class_id, label_index=int(row.class_id))
            def close(self):
                pass
        a = SimpleNamespace(descriptor={"digest": "synthetic-assets"},
                            class_map=SimpleNamespace(id_to_index={str(i): i for i in range(3)}),
                            dataset=lambda partition, processor=None, online=False, purpose=None: Dataset(purpose or "train"))
        actual_trainer, actual_scoring, actual_loader = engine.Trainer, engine.scoring, engine.loader_for
        def trainer(*args, **kwargs):
            kwargs.update(device="cpu", precision="fp32")
            return actual_trainer(*args, **kwargs)
        def scoring(*args, **kwargs):
            kwargs["device"] = "cpu"
            return actual_scoring(*args, **kwargs)
        def loader(*args, **kwargs):
            kwargs["workers"] = 0
            return actual_loader(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(engine, "require_machine"), patch.object(admission, "load_assets", return_value=a), \
             patch.object(engine, "student_for", return_value=(admission.tiny_student("full_visual"), None, {"sha256": "synthetic-head"})), \
             patch.object(engine, "Trainer", side_effect=trainer), patch.object(engine, "scoring", side_effect=scoring), \
             patch.object(engine, "loader_for", side_effect=loader):
            root = Path(directory) / "second_round/feasibility"
            value = admission.feasibility("synthetic", method="snscl", adaptation="full_visual", machine="synthetic", output=root)
            self.assertEqual((value["scoring_samples"], value["samples"], value["updates"]), (300, 256, 2))
            self.assertGreater(value["queue_count"], 0)
            self.assertFalse(value["checkpoint_created"])
            self.assertEqual(list(root.glob("*.pt")), [])

    def test_method_failure_stops_group_even_if_another_grid_point_passed(self):
        with self.assertRaisesRegex(ValueError, "entire admission group"):
            admission.choose_common([{"method_failure": True}], ["turn"])


if __name__ == "__main__":
    unittest.main()
