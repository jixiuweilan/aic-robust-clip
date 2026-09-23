"""Official train/ wrapper, with original archive/member bytes preserved."""
import io
import hashlib
import importlib
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
    def test_authorized_single_member_exclusion_is_bound_to_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "second_round"
            root.mkdir()
            archive = root / "train.zip"
            bad_member = "train/0000/5.png"
            with zipfile.ZipFile(archive, "w") as z:
                for index in range(20):
                    data = io.BytesIO()
                    Image.new("RGB", (3, 3), (index, 10, 20)).save(data, format="PNG")
                    z.writestr(f"train/0000/{index}.png", data.getvalue())
                    if index == 5:
                        bad_bytes = data.getvalue()
            original_sha = assets.file_sha256(archive)
            byte_sha = hashlib.sha256(bad_bytes).hexdigest()
            audit_module = importlib.import_module("aic_robust_clip.data.audit")
            decode = audit_module._decode_pixels

            def decode_one(raw, **kwargs):
                if raw == bad_bytes:
                    raise SyntaxError("not a TIFF file")
                return decode(raw, **kwargs)

            with patch.object(audit_module, "_decode_pixels", side_effect=decode_one):
                with patch.object(assets, "inspect_weights", return_value={"digest": "synthetic"}):
                    result = assets.audit(archive, root / "audit", source_url="https://organizer.invalid/synthetic",
                        retrieved_at="synthetic", organizer_version="synthetic", weights="unused",
                        weight_revision="unused", member_prefix="train", expected_sha256=original_sha,
                        exclude_member=bad_member, exclude_byte_sha256=byte_sha)
                with self.assertRaisesRegex(AuditError, "exclusion was not verified"):
                    audit_archive(archive, stage="second_round", role="train", member_prefix="train",
                        exclude_member=bad_member, exclude_byte_sha256="0" * 64)
            report = read_json(root / "audit/manifest.json")
            self.assertEqual(len(report["records"]), 19)
            self.assertEqual(report["failures"], [])
            self.assertEqual(result["exclusions"], read_json(root / "audit/exclusions.json"))
            self.assertEqual(result["exclusions"][0]["member_path"], bad_member)
            self.assertEqual(result["exclusions"][0]["byte_sha256"], byte_sha)
            self.assertEqual(assets.file_sha256(archive), original_sha)

    def test_pillow_exif_syntax_error_is_recorded_without_stopping_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "train.zip"
            data = io.BytesIO()
            Image.new("RGB", (2, 2)).save(data, format="PNG")
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("train/0000/bad.png", data.getvalue())
                z.writestr("train/0000/good.png", data.getvalue())
            with patch("aic_robust_clip.data.audit._decode_pixels", side_effect=[
                SyntaxError("not a TIFF file"), ("0" * 64, 2, 2, 3)]):
                report = audit_archive(archive, stage="second_round", role="train", member_prefix="train")
            self.assertTrue(report.complete)
            self.assertFalse(report.usable)
            self.assertEqual(len(report.records), 2)
            self.assertEqual(report.records[0].decode_status, "failed")
            self.assertEqual(report.records[1].decode_status, "decoded")
            self.assertEqual(report.failures[0].member_path, "train/0000/bad.png")

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
