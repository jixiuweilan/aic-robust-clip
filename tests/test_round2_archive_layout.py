"""Official train/ wrapper, with original archive/member bytes preserved."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image
from aic_robust_clip.data.audit import audit_archive, AuditError
from aic_robust_clip.round2 import assets
from aic_robust_clip.contracts import read_json


class ArchiveLayoutTests(unittest.TestCase):
    def test_explicit_wrapper_keeps_names_hashes_and_strict_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "second_round"
            root.mkdir()
            archive = root / "train.zip"
            with zipfile.ZipFile(archive, "w") as z:
                for index in range(20):
                    data = io.BytesIO()
                    Image.new("RGB", (index + 2, 4), (index, 10, 20)).save(data, format="PNG")
                    z.writestr(f"train/{index % 2:04d}/{index}.png", data.getvalue())
            before = assets.file_sha256(archive)
            old = audit_archive(archive, stage="second_round", role="train")
            self.assertFalse(old.usable)
            new = audit_archive(archive, stage="second_round", role="train", member_prefix="train")
            self.assertTrue(new.usable)
            self.assertEqual({r.class_id for r in new.records}, {"0000", "0001"})
            self.assertTrue(all(r.member_path.startswith("train/") for r in new.records))
            self.assertEqual(before, new.archive_identity)
            with patch.object(assets, "inspect_weights", return_value={"digest": "synthetic"}):
                result = assets.audit(archive, root / "audit", source_url="https://organizer.invalid/synthetic",
                    retrieved_at="synthetic", organizer_version="synthetic", weights="unused", weight_revision="unused",
                    member_prefix="train")
            self.assertEqual(result["train_member_prefix"], "train")
            self.assertEqual(len(read_json(root / "audit/manifest.json")["records"]), 20)
            self.assertEqual(assets.file_sha256(archive), before)

    def test_prefix_never_accepts_test_or_outside_training_members(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.zip"
            data = io.BytesIO()
            Image.new("RGB", (3, 3)).save(data, format="PNG")
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("test/0001/x.png", data.getvalue())
            report = audit_archive(path, stage="second_round", role="train", member_prefix="train")
            self.assertFalse(report.usable)
            self.assertEqual(report.records, [])
            for prefix in ("", "../train", "/train", "train/", "train\\"):
                with self.assertRaises(AuditError):
                    audit_archive(path, stage="second_round", role="train", member_prefix=prefix)
            with self.assertRaises(AuditError):
                audit_archive(path, stage="second_round", role="test", member_prefix="train")


if __name__ == "__main__":
    unittest.main()
