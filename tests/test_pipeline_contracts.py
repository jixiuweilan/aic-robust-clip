from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from aic_robust_clip.contracts import (
    CheckpointMetadata,
    ClassMap,
    ContractError,
    DatasetRegistration,
    RunConfig,
    SampleRecord,
)
from aic_robust_clip.data.audit import audit_archive, load_manifest, write_manifest
from aic_robust_clip.data.dataset import DatasetError, ManifestDataset
from aic_robust_clip.data.splits import SplitError, assert_no_group_crossing, make_grouped_split
from aic_robust_clip.metrics import evaluate_classification
from aic_robust_clip.models.clip import WeightIdentity, WeightIdentityError, validate_weight_identity
from aic_robust_clip.runtime import RuntimeLimitError, RuntimePolicy, resolve_run_config
from aic_robust_clip.training.reliability import ReliabilityError, ReliabilityState, observed_prior


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def sample(sample_id: str, class_id: str, *, byte_digest: str = DIGEST_A) -> SampleRecord:
    return SampleRecord(
        stage="preliminary",
        role="train",
        archive_identity=DIGEST_B,
        archive_path="fixture.zip",
        member_path=f"{class_id}/{sample_id}.jpg",
        sample_id=sample_id,
        byte_size=4,
        crc32=0,
        byte_sha256=byte_digest,
        pixel_sha256=byte_digest,
        decode_status="decoded",
        class_id=class_id,
    )


class ContractTests(unittest.TestCase):
    def test_contracts_reject_test_labels_and_noncanonical_weight(self) -> None:
        with self.assertRaises(ContractError):
            SampleRecord(
                stage="preliminary",
                role="test",
                archive_identity=DIGEST_A,
                archive_path="test.zip",
                member_path="x.jpg",
                sample_id=DIGEST_B,
                byte_size=1,
                crc32=0,
                class_id="0001",
            )
        with self.assertRaises(WeightIdentityError):
            validate_weight_identity("openai/clip-vit-base-patch32", "main")

    def test_class_map_preserves_exact_ids(self) -> None:
        class_map = ClassMap.from_ids("preliminary", ["0010", "0002"])
        self.assertEqual(class_map.index_for("0002"), 0)
        self.assertEqual(class_map.id_for(1), "0010")
        self.assertEqual(len(class_map.digest), 64)

    def test_runtime_rejects_formal_mode_locally(self) -> None:
        config = RunConfig(stage="preliminary", execution_mode="formal", official_weight_revision="immutable")
        with self.assertRaises(RuntimeLimitError):
            resolve_run_config(config)
        self.assertIsInstance(resolve_run_config(RunConfig(stage="preliminary")), RunConfig)


class AuditAndSplitTests(unittest.TestCase):
    def test_zip_audit_is_read_only_and_reports_unsafe_duplicate_members(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "train.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("0001/a.jpg", b"one")
                archive.writestr("0001/a.jpg", b"two")
                archive.writestr("../escape.jpg", b"bad")
                archive.writestr("0001/notes.txt", b"not an image")
            before = archive_path.read_bytes()
            report = audit_archive(
                archive_path,
                stage="preliminary",
                role="train",
                decode=False,
            )
            self.assertTrue(report.complete)
            self.assertEqual(report.total_members, 4)
            self.assertEqual(len(report.records), 1)
            self.assertTrue(any(failure.reason == "duplicate_member_path" for failure in report.failures))
            self.assertTrue(any(failure.reason == "unsafe_member_path" for failure in report.failures))
            self.assertEqual(before, archive_path.read_bytes())

    def test_split_is_order_independent_and_groups_exact_duplicates(self) -> None:
        records = [sample(f"s{index}", "0001" if index < 6 else "0002", byte_digest=DIGEST_A if index < 2 else f"{index:064x}") for index in range(10)]
        first = make_grouped_split(records, seed=17)
        second = make_grouped_split(list(reversed(records)), seed=17)
        self.assertEqual(first.digest, second.digest)
        assert_no_group_crossing(first)
        self.assertEqual(sum(first.counts().values()), 10)
        with self.assertRaises(SplitError):
            changed = records[1].to_dict()
            changed.pop("schema_version", None)
            changed["stage"] = "semifinal"
            make_grouped_split([records[0], SampleRecord(**changed)])

    def test_manifest_round_trip_requires_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            records = [sample("one", "0001", byte_digest=DIGEST_A)]
            write_manifest(path, records, complete=True)
            self.assertEqual(load_manifest(path, require_complete=True)[0].sample_id, records[0].sample_id)

    def test_loader_rejects_dev_as_training_loader(self) -> None:
        with self.assertRaises(DatasetError):
            ManifestDataset(
                [sample("one", "0001")],
                stage="preliminary",
                role="train",
                partition="dev",
                purpose="train",
                class_to_index={"0001": 0},
            )


class MetricTests(unittest.TestCase):
    def test_hand_computed_metrics(self) -> None:
        metrics = evaluate_classification([0, 0, 1, 1], [0, 1, 1, 1], total_classes=2, training_counts={0: 10, 1: 2})
        self.assertAlmostEqual(metrics.micro_top1, 0.75)
        self.assertAlmostEqual(metrics.macro_recall, 0.75)
        self.assertEqual(metrics.present_classes, 2)


class ReliabilityTests(unittest.TestCase):
    def test_prior_is_laplace_smoothed(self) -> None:
        self.assertEqual(observed_prior([0, 0, 1], 3), [0.5, 1 / 3, 1 / 6])

    def test_weight_history_is_training_only_and_warmup_bounded(self) -> None:
        ids = tuple(f"s{index}" for index in range(5))
        state = ReliabilityState(ids, {sample_id: "0001" for sample_id in ids})
        with self.assertRaises(ReliabilityError):
            state.update_losses({**{sample_id: 1.0 for sample_id in ids}, "dev-id": 2.0})
        losses = {sample_id: float(index) for index, sample_id in enumerate(ids)}
        self.assertEqual(state.finish_epoch(losses), {sample_id: 1.0 for sample_id in ids})
        state.finish_epoch(losses)
        weights = state.next_epoch_weights()
        self.assertAlmostEqual(weights["s0"], 1.0)
        self.assertAlmostEqual(weights["s4"], 0.2)


if __name__ == "__main__":
    unittest.main()
