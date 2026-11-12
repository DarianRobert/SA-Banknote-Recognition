import contextlib
import importlib.util
import io
import tempfile
import unittest
import warnings
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("banknotes", ROOT / "main.py")
banknotes = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(banknotes)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.system = banknotes.SequentialBankNoteOptimizer(self.root / "snapshots")
        self.blank = np.zeros((300, 600, 3), dtype=np.uint8)

    def test_preprocessing_keeps_image_dimensions(self):
        for method in (
            self.system.prep_clahe,
            self.system.prep_bilateral,
            self.system.prep_gaussian,
            self.system.prep_equalize,
        ):
            with self.subTest(method=method.__name__):
                result = method(self.blank)
                self.assertEqual(result.shape, (300, 600))
                self.assertEqual(result.dtype, np.uint8)

    def test_no_contours_produces_no_segmentation(self):
        gray = self.system.prep_clahe(self.blank)
        for method in (
            self.system.seg_warp,
            self.system.seg_exact_mask,
            self.system.seg_padded_bbox,
            self.system.seg_adapt_thresh,
        ):
            with self.subTest(method=method.__name__):
                self.assertIsNone(method(gray))
        self.assertEqual(
            self.system.apply_segmentation(self.blank, None, "Contour + Exact Mask"),
            (None, None),
        )

    def test_degenerate_warp_is_rejected(self):
        points = np.zeros((4, 1, 2), dtype=np.int32)
        self.assertEqual(
            self.system.apply_segmentation(
                self.blank, points, "Contour + Perspective Warp"
            ),
            (None, None),
        )

    def test_featureless_notes_are_unknown(self):
        for method in (
            self.system.classify_sift,
            self.system.classify_orb,
            self.system.classify_akaze,
        ):
            with self.subTest(method=method.__name__):
                self.assertEqual(method(self.blank, None, {"R10": [None]}), "Unknown")

    def test_dataset_is_sorted_and_skips_non_images(self):
        for denomination in ("R20", "R10"):
            folder = self.root / denomination
            folder.mkdir()
            for name in ("b.JPG", "a.png"):
                self.assertTrue(cv2.imwrite(str(folder / name), self.blank))
            (folder / "notes.txt").write_text("ignored", encoding="utf-8")
        dataset = banknotes.load_dataset(self.root)
        self.assertEqual(
            [(label, name) for label, name, _ in dataset],
            [(label, name) for label in ("R10", "R20") for name in ("a.png", "b.JPG")],
        )

    def test_missing_empty_and_unreadable_datasets_have_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "directory not found"):
            banknotes.load_dataset(self.root / "missing")
        with self.assertRaisesRegex(ValueError, "No readable"):
            banknotes.load_dataset(self.root)
        folder = self.root / "R10"
        folder.mkdir()
        (folder / "broken.jpg").write_text("not an image", encoding="utf-8")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaisesRegex(ValueError, "No readable"):
                banknotes.load_dataset(self.root)
        self.assertTrue(
            any("Skipping unreadable image" in str(item.message) for item in caught)
        )

    def test_cli_explains_a_missing_dataset(self):
        error = io.StringIO()
        with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as raised:
            banknotes.main(["--templates", str(self.root / "missing")])
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("Dataset directory not found", error.getvalue())


if __name__ == "__main__":
    unittest.main()
