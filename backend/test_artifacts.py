"""Integration tests for validated artifact promotion (small models only)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from backend.train import promote_artifacts


def build_tiny_model():
    inputs = layers.Input(shape=(8, 8, 3), name="input_layer")
    pooled = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(inputs)
    outputs = layers.Dense(2, activation="softmax", name="predictions")(pooled)
    return Model(inputs, outputs)


class PromoteArtifactsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image_path = self.root / "sample.png"
        Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(self.image_path)
        self.model = build_tiny_model()
        self.class_names = ["alpha", "beta"]
        self.metrics = {
            "class_count": 2,
            "class_names": self.class_names,
            "best_val_accuracy": 0.9,
            "test_set_evaluated": False,
        }
        self.model_dir = self.root / "saved_model"
        self.sample_rows = [{"path": str(self.image_path)}]

    def tearDown(self):
        self.temporary.cleanup()

    def promote(self, metrics=None):
        return promote_artifacts(
            self.model,
            self.class_names,
            metrics or self.metrics,
            model_dir=self.model_dir,
            image_size=8,
            sample_rows=self.sample_rows,
        )

    def test_promotes_valid_artifacts_and_keeps_backups(self):
        self.model_dir.mkdir()
        for name in (
            "plant_disease_model.h5",
            "class_names.json",
            "training_metrics.json",
        ):
            (self.model_dir / name).write_text("previous", encoding="utf-8")

        promoted = self.promote()

        self.assertTrue((self.model_dir / "plant_disease_model.h5").is_file())
        self.assertEqual(
            json.loads((self.model_dir / "class_names.json").read_text(encoding="utf-8")),
            self.class_names,
        )
        self.assertEqual(
            json.loads(
                (self.model_dir / "training_metrics.json").read_text(encoding="utf-8")
            ),
            self.metrics,
        )
        backup_names = {path.name for path in promoted}
        self.assertIn("plant_disease_model_backup.h5", backup_names)
        self.assertEqual(list(self.model_dir.glob("*.tmp.*")), [])

    def test_inconsistent_metrics_do_not_replace_active_artifacts(self):
        self.model_dir.mkdir()
        active = self.model_dir / "plant_disease_model.h5"
        active.write_bytes(b"original-model-placeholder")
        bad_metrics = dict(self.metrics)
        bad_metrics["class_names"] = ["alpha", "gamma"]

        with self.assertRaises(ValueError):
            self.promote(bad_metrics)

        self.assertEqual(active.read_bytes(), b"original-model-placeholder")
        self.assertEqual(list(self.model_dir.glob("*.tmp.*")), [])


if __name__ == "__main__":
    unittest.main()
