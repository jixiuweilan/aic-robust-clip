"""Offline migration regressions: synthetic archives and mocked GPU reports only."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from aic_robust_clip.contracts import ContractError, RunConfig, read_json, sha256_json, write_json
from aic_robust_clip.data import relocation
from aic_robust_clip.data.audit import audit_archive
from aic_robust_clip.data.dataset import DatasetError, ManifestDataset, _read_record_bytes
from aic_robust_clip.data.splits import make_grouped_split
from aic_robust_clip.environment import DEVELOPMENT_HOST, bind_training, machine_policy
from aic_robust_clip.runtime import RuntimeLimitError, RuntimePolicy


class EnrollmentTests(unittest.TestCase):
    def report(self, name="Tesla T4", *, host="a" * 64, cuda=True):
        return {"fingerprint": host, "cuda_available": cuda, "gpu": {"name": name}}

    def test_approved_gpu_enrollment_is_host_bound(self):
        for name in ("Tesla T4", "NVIDIA T4", "NVIDIA GeForce RTX 4060", "NVIDIA GeForce RTX 4060 Laptop GPU"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "machine.json"
                report = self.report(name)
                with patch("aic_robust_clip.environment.environment_report", return_value=report):
                    self.assertEqual(bind_training(path), report)
                    original = path.read_bytes()
                    with self.assertRaises(FileExistsError):
                        bind_training(path)
                    self.assertEqual(path.read_bytes(), original)
                self.assertEqual(read_json(path)["preflight"], report)
                with patch("aic_robust_clip.environment.machine_fingerprint", return_value=report["fingerprint"]):
                    policy = machine_policy(path)
                    self.assertTrue(policy.allow_formal)
                    self.assertEqual(policy.machine, "bound-approved-experiment")
                    policy.validate(RunConfig(stage="preliminary", execution_mode="formal"))
                with patch("aic_robust_clip.environment.machine_fingerprint", return_value="b" * 64):
                    with self.assertRaises(RuntimeLimitError):
                        machine_policy(path)

    def test_development_host_remains_forbidden_with_approved_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "machine.json"
            with patch("aic_robust_clip.environment.environment_report", return_value=self.report(host=DEVELOPMENT_HOST)):
                with self.assertRaisesRegex(RuntimeLimitError, "development host"):
                    bind_training(path)
                self.assertFalse(path.exists())
            with patch("aic_robust_clip.environment.machine_fingerprint", return_value=DEVELOPMENT_HOST):
                with self.assertRaisesRegex(RuntimeLimitError, "forbidden"):
                    RuntimePolicy(allow_formal=True).validate(RunConfig(stage="preliminary", execution_mode="formal"))

    def test_unsupported_or_unavailable_gpu_cannot_enroll(self):
        cases = [self.report("NVIDIA A100"), self.report("Tesla T40"), self.report("RTX 40600"),
                 self.report("unrelated 4060"), self.report(cuda=False)]
        for report in cases:
            with self.subTest(report=report), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "machine.json"
                with patch("aic_robust_clip.environment.environment_report", return_value=report):
                    with self.assertRaisesRegex(RuntimeLimitError, "approved CUDA"):
                        bind_training(path)
                self.assertFalse(path.exists())

    def test_enrolled_host_does_not_relax_smoke_limits(self):
        policy = RuntimePolicy(machine="bound-approved-experiment", allow_formal=True)
        for limits in ({"max_updates": 4}, {"max_samples": 9}, {"max_eval_batches": 3}):
            with self.subTest(limits=limits), self.assertRaises((ContractError, RuntimeLimitError)):
                policy.validate(RunConfig(stage="preliminary", execution_mode="smoke", **limits))


class RelocationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.original = self.root / "original.zip"
        with zipfile.ZipFile(self.original, "w") as archive:
            for i in range(20):
                archive.writestr(f"{i % 2:04}/{i}.png", f"synthetic-bytes-{i}".encode())
        self.report = audit_archive(self.original, stage="preliminary", role="train", decode=False)
        self.report.records.sort(key=lambda record: record.sample_id)
        self.record = self.report.records[0]
        self.target = self.root / "moved.zip"
        shutil.copyfile(self.original, self.target)
        self.mapping = self.root / "locations.json"
        self.value = {"schema_version": 1, "reason": "synthetic migration regression", "archives": {
            str(self.original): {"path": str(self.target), "sha256": self.record.archive_identity}}}
        write_json(self.mapping, self.value)
        self.environment = patch.dict(os.environ, {"AIC_ARCHIVE_LOCATIONS": str(self.mapping)})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.dataset = ManifestDataset(self.report.records, stage="preliminary", role="train",
            partition="train", purpose="train", class_to_index={"0000": 0, "0001": 1})
        self.addCleanup(self.dataset.close)

    def test_both_readers_preserve_records_and_split_identity(self):
        before = [r.to_dict() for r in self.report.records]
        split = make_grouped_split(self.report.records)
        digest = split.digest
        with zipfile.ZipFile(self.original) as archive:
            expected = archive.read(self.record.member_path)
        self.original.unlink()
        self.assertEqual(_read_record_bytes(self.record), expected)
        self.assertEqual(self.dataset[0].image, expected)
        selected = ManifestDataset.from_split(self.report.records, split.records, stage="preliminary",
            partition="train", purpose="train", class_to_index={"0000": 0, "0001": 1})
        try:
            self.assertIsInstance(selected[0].image, bytes)
        finally:
            selected.close()
        self.assertEqual(before, [r.to_dict() for r in self.report.records])
        self.assertEqual(sha256_json(before), self.report.manifest_digest)
        self.assertEqual(split.digest, digest)

    def test_opt_in_and_unmapped_paths_keep_original_behavior(self):
        with patch.dict(os.environ, {"AIC_ARCHIVE_LOCATIONS": ""}), \
                patch.object(relocation, "_sha256_file", side_effect=AssertionError("unexpected hash")):
            self.assertIsInstance(self.dataset[0].image, bytes)
            self.original.unlink()
            with self.assertRaises(DatasetError):
                _read_record_bytes(self.record)
        self.value["archives"] = {str(self.root / "unrelated.zip"): self.value["archives"][str(self.original)]}
        write_json(self.mapping, self.value)
        self.assertEqual(relocation.resolve_archive(self.original, archive_identity=self.record.archive_identity), self.original)

    def test_wrong_target_rejected_before_zip_open_without_fallback(self):
        self.target.write_bytes(b"not the audited archive")
        with patch("aic_robust_clip.data.dataset.zipfile.ZipFile", side_effect=AssertionError("opened unverified file")):
            for read in (lambda: self.dataset[0], lambda: _read_record_bytes(self.record)):
                with self.assertRaisesRegex(relocation.RelocationError, "hash mismatch"):
                    read()

    def test_self_consistent_wrong_mapping_cannot_replace_audited_identity(self):
        self.target.write_bytes(b"another archive with its own valid hash")
        self.value["archives"][str(self.original)]["sha256"] = hashlib.sha256(self.target.read_bytes()).hexdigest()
        write_json(self.mapping, self.value)
        with patch.object(relocation, "_sha256_file", side_effect=AssertionError("wrong identity was accepted")):
            with self.assertRaisesRegex(relocation.RelocationError, "audited archive_identity"):
                self.dataset[0]

    def test_missing_target_and_directory_are_rejected(self):
        for path in (self.root / "absent.zip", self.root):
            self.value["archives"][str(self.original)]["path"] = str(path)
            write_json(self.mapping, self.value)
            with self.subTest(path=path), self.assertRaises(relocation.RelocationError):
                self.dataset[0]

    def test_malformed_mapping_is_not_silently_ignored(self):
        invalid = [[], {}, {**self.value, "schema_version": True}, {**self.value, "archives": {}},
                   {**self.value, "reason": ""}, {**self.value, "typo": "unexpected"}]
        for change in ({"path": "relative.zip"}, {"sha256": "A" * 64}, {"path": None},
                       {"path": "/bad\x00path"}, {"extra": 1}):
            value = copy.deepcopy(self.value)
            value["archives"][str(self.original)].update(change)
            invalid.append(value)
        invalid.append({**self.value, "archives": {"relative.zip": self.value["archives"][str(self.original)]}})
        for value in invalid:
            write_json(self.mapping, value)
            with self.subTest(value=value), self.assertRaises(relocation.RelocationError):
                self.dataset[0]
        for raw in ('{', '{"schema_version":1,"schema_version":1}', '\ufffd'):
            self.mapping.write_text(raw)
            with self.subTest(raw=raw), self.assertRaises(relocation.RelocationError):
                self.dataset[0]
        self.mapping.unlink()
        with self.assertRaises(relocation.RelocationError):
            self.dataset[0]

    def test_verification_cached_only_for_unchanged_file_and_process(self):
        with patch.object(relocation, "_sha256_file", wraps=relocation._sha256_file) as hashed:
            self.dataset[0]
            _read_record_bytes(self.record)
            self.assertEqual(hashed.call_count, 1)
            pid = os.getpid()
            with patch.object(relocation.os, "getpid", return_value=pid + 1):
                _read_record_bytes(self.record)
            self.assertEqual(hashed.call_count, 2)
            previous = self.target.stat()
            raw = bytearray(self.target.read_bytes())
            raw[-1] ^= 1
            self.target.write_bytes(raw)
            os.utime(self.target, ns=(previous.st_atime_ns, previous.st_mtime_ns))
            with self.assertRaisesRegex(relocation.RelocationError, "hash mismatch"):
                self.dataset[0]  # must reject even with a cached ZipFile handle
            self.assertEqual(hashed.call_count, 3)

    def test_replacement_and_removal_invalidate_cached_verification(self):
        self.dataset[0]
        replacement = self.root / "replacement.zip"
        replacement.write_bytes(b"wrong replacement")
        replacement.replace(self.target)
        with self.assertRaisesRegex(relocation.RelocationError, "hash mismatch"):
            self.dataset[0]
        self.target.unlink()
        with self.assertRaises(relocation.RelocationError):
            self.dataset[0]

    def test_changed_mapping_is_revalidated_and_other_mapping_is_not_cached(self):
        self.dataset[0]
        self.value["archives"][str(self.original)]["sha256"] = "0" * 64
        write_json(self.mapping, self.value)
        with self.assertRaisesRegex(relocation.RelocationError, "hash mismatch"):
            self.dataset[0]
        other = self.root / "other.json"
        other.write_text("not json")
        with patch.dict(os.environ, {"AIC_ARCHIVE_LOCATIONS": str(other)}):
            with self.assertRaises(relocation.RelocationError):
                _read_record_bytes(self.record)

    def test_change_during_hash_is_rejected(self):
        original_hash = relocation._sha256_file
        def unstable_hash(path):
            digest = original_hash(path)
            path.write_bytes(b"changed after hash")
            return digest
        with patch.object(relocation, "_sha256_file", side_effect=unstable_hash):
            with self.assertRaisesRegex(relocation.RelocationError, "changed during verification"):
                self.dataset[0]


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ("torch", "PIL", "numpy", "transformers")),
                     "CPU torch/transformers/Pillow/numpy required")
class RelocationWorkflowTests(unittest.TestCase):
    def test_same_frozen_manifest_produces_identical_cache_and_smoke_head_after_move(self):
        # Reuse the existing explicit tiny-model fixture, not official images or
        # weights. Move the physical file, NEVER rewrite the frozen manifest.
        import test_workflow as fixtures
        from aic_robust_clip.configuration import prepare
        from aic_robust_clip.workflow import cache_command, init_head_command
        fixtures.torch.set_num_threads(1)
        identity = {"model_id": "openai/clip-vit-base-patch32", "revision": "a" * 40,
                    "digest": "b" * 64, "preprocessing_digest": "c" * 64}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"AIC_ARCHIVE_LOCATIONS": ""}), \
                patch("aic_robust_clip.configuration.inspect_weights", return_value=identity), \
                patch("aic_robust_clip.workflow.load_bundle", side_effect=lambda ctx: fixtures.WorkflowTests.bundle(self, ctx)):
            root = Path(directory)
            config = fixtures.WorkflowTests.fixtures(self, root)
            path = root / "config.json"
            write_json(path, config)
            frozen = {name: (root / name).read_bytes() for name in ("manifest.json", "split.json", "class-map.json")}
            ctx = prepare(path)
            direct_cache = cache_command(path, "train")
            direct_head = init_head_command(path)
            moved = root / "moved.zip"
            (root / "train.zip").rename(moved)
            mapping = root / "mapping.json"
            write_json(mapping, {"schema_version": 1, "reason": "fixture relocation", "archives": {
                str(root / "train.zip"): {"path": str(moved), "sha256": ctx.records[0].archive_identity}}})
            config.update(train_cache="moved-cache", head="moved-head")
            write_json(path, config)
            with patch.dict(os.environ, {"AIC_ARCHIVE_LOCATIONS": str(mapping)}):
                relocated_cache = cache_command(path, "train")
                relocated_head = init_head_command(path)
            self.assertEqual(direct_cache, relocated_cache)
            self.assertEqual(direct_head["identity"], relocated_head["identity"])
            self.assertEqual(direct_head["sha256"], relocated_head["sha256"])
            for head in (direct_head, relocated_head):
                self.assertEqual(head["identity"]["profile"], "HEAD-SMOKE")
                self.assertEqual((head["result"]["updates"], head["result"]["samples"]), (2, 2))
            for name, original in frozen.items():
                self.assertEqual((root / name).read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
