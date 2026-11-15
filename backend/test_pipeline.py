"""Fast tests for the group-aware split and audit artifacts (no TensorFlow)."""

import csv
import json
import unittest
from collections import Counter
from pathlib import Path

from backend.data_pipeline import (
    DEFAULT_AUDIT_SUMMARY_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_ORIGINAL_VAL_PATH,
    REPO_DIR,
    load_manifest,
    validate_manifest,
)


class SplitManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_manifest(DEFAULT_MANIFEST_PATH)
        cls.validation = validate_manifest(cls.rows)

    def test_split_sizes_and_fractions(self):
        self.assertEqual(self.validation["image_count"], 20638)
        self.assertEqual(self.validation["split_counts"]["train"], 14444)
        self.assertEqual(self.validation["split_counts"]["val"], 3097)
        self.assertEqual(self.validation["split_counts"]["test"], 3097)
        for split in ("train", "val", "test"):
            fraction = self.validation["split_fractions"][split]
            expected = {"train": 0.70, "val": 0.15, "test": 0.15}[split]
            self.assertAlmostEqual(fraction, expected, delta=0.005)

    def test_every_class_is_in_every_split(self):
        self.assertEqual(self.validation["class_count"], 15)
        for label, counts in self.validation["class_split_counts"].items():
            for split in ("train", "val", "test"):
                self.assertGreater(counts[split], 0, f"{label} missing from {split}")

    def test_duplicate_groups_do_not_cross_splits(self):
        groups = {}
        for row in self.rows:
            groups.setdefault(row["group_id"], set()).add(row["split"])
        crossing = {key: value for key, value in groups.items() if len(value) > 1}
        self.assertEqual(crossing, {})
        self.assertEqual(self.validation["duplicate_groups"], len(groups))

    def test_no_paths_are_shared_between_splits(self):
        paths = [row["path"] for row in self.rows]
        self.assertEqual(len(paths), len(set(paths)))
        test_paths = {row["path"] for row in self.rows if row["split"] == "test"}
        train_paths = {row["path"] for row in self.rows if row["split"] == "train"}
        val_paths = {row["path"] for row in self.rows if row["split"] == "val"}
        self.assertEqual(test_paths & train_paths, set())
        self.assertEqual(test_paths & val_paths, set())
        self.assertEqual(train_paths & val_paths, set())

    def test_class_mapping_matches_active_saved_model(self):
        saved = json.loads(
            (REPO_DIR / "backend" / "saved_model" / "class_names.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(saved, self.validation["class_names"])


class AuditSummaryTest(unittest.TestCase):
    def test_audit_summary_is_consistent(self):
        if not DEFAULT_AUDIT_SUMMARY_PATH.is_file():
            self.skipTest("dataset_audit_summary.json not generated")
        summary = json.loads(DEFAULT_AUDIT_SUMMARY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(summary["image_count"], 20638)
        self.assertEqual(summary["class_count"], 15)
        self.assertEqual(summary["corrupt_images"], [])
        self.assertEqual(summary["cross_class_exact_duplicate_groups"], [])
        self.assertEqual(summary["split"]["split_counts"]["train"], 14444)
        self.assertEqual(summary["split"]["split_counts"]["val"], 3097)
        self.assertEqual(summary["split"]["split_counts"]["test"], 3097)
        self.assertEqual(
            summary["original_split"]["validation_images_with_duplicate_in_training"],
            17,
        )


class OriginalValidationSplitTest(unittest.TestCase):
    def test_original_split_matches_keras_int_20_percent_rule(self):
        with DEFAULT_ORIGINAL_VAL_PATH.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4122)
        by_class = Counter(row["label"] for row in rows)
        dataset_counts = Counter()
        for class_dir in (REPO_DIR / "dataset" / "PlantVillage").iterdir():
            if class_dir.is_dir():
                dataset_counts[class_dir.name] = len(
                    [
                        path
                        for path in class_dir.iterdir()
                        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
                    ]
                )
        self.assertEqual(set(by_class), set(dataset_counts))
        for label, count in dataset_counts.items():
            self.assertEqual(by_class[label], int(0.2 * count), label)


if __name__ == "__main__":
    unittest.main()
