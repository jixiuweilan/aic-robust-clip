from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from aic_robust_clip.submission import SubmissionError, validate_submission


VALID_CONTENT = "image_a.jpg,0001\nimage_b.JPG,0123\n"


class ValidateSubmissionTests(unittest.TestCase):
    def test_valid_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pred_results.csv"
            path.write_text(VALID_CONTENT, encoding="utf-8")
            self.assertEqual(validate_submission(path), 2)

    def test_rejects_wrong_csv_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            path.write_text(VALID_CONTENT, encoding="utf-8")
            with self.assertRaisesRegex(SubmissionError, "must be named"):
                validate_submission(path)

    def test_rejects_bad_label_and_header(self) -> None:
        cases = {
            "unpadded": "image.jpg,12\n",
            "header": "filename,label\n",
        }
        for name, content in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "pred_results.csv"
                path.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(SubmissionError, "four digits"):
                    validate_submission(path)

    def test_rejects_duplicate_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pred_results.csv"
            path.write_text("image.jpg,0001\nimage.jpg,0002\n", encoding="utf-8")
            with self.assertRaisesRegex(SubmissionError, "duplicate"):
                validate_submission(path)

    def test_valid_zip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("pred_results.csv", VALID_CONTENT)
            self.assertEqual(validate_submission(path), 2)

    def test_rejects_extra_zip_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("pred_results.csv", VALID_CONTENT)
                archive.writestr("notes.txt", "unexpected")
            with self.assertRaisesRegex(SubmissionError, "contain only"):
                validate_submission(path)

    def test_expected_file_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submission = root / "pred_results.csv"
            expected = root / "expected.txt"
            submission.write_text(VALID_CONTENT, encoding="utf-8")
            expected.write_text("image_a.jpg\nimage_c.jpg\n", encoding="utf-8")
            with self.assertRaisesRegex(SubmissionError, "missing 1 file"):
                validate_submission(submission, expected)


if __name__ == "__main__":
    unittest.main()
