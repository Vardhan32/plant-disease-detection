"""Backend tests for the plant disease detection Flask app.

Run from the repository root with the project virtual environment::

    .\\.venv\\Scripts\\python.exe -m unittest discover -s backend -p "test_*.py" -t .

The tests load the real tracked model via ``backend.app`` (no retraining), so
TensorFlow and the other ``requirements.txt`` dependencies must be installed.
Model loading takes ~30 s on CPU; that cost is paid once per run.
"""

import io
import json
import os
import sys
import unittest

import numpy as np
import tensorflow as tf
from PIL import Image

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BACKEND_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from backend.app import (  # noqa: E402
    IMG_SIZE,
    MAX_UPLOAD_BYTES,
    app,
    class_names,
    model,
)
from backend.data_pipeline import DEFAULT_MANIFEST_PATH, REPO_DIR as PIPELINE_REPO_DIR  # noqa: E402
from backend.preprocessing import preprocess_bytes, preprocess_image  # noqa: E402


def make_image_bytes(fmt="JPEG", size=(128, 128), color=(34, 139, 34)):
    """Build a small in-memory image for upload tests."""
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def first_validation_image():
    import csv

    with open(DEFAULT_MANIFEST_PATH, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "val":
                return PIPELINE_REPO_DIR / row["path"]
    raise unittest.SkipTest("No validation image available in split manifest")


class HealthTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_returns_ok_and_class_count(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["classes"], len(class_names))
        self.assertGreater(len(class_names), 1)
        self.assertEqual(payload["input_size"], list(IMG_SIZE)[::-1])


class PredictTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def post_image(self, data, filename):
        return self.client.post(
            "/predict",
            data={"file": (io.BytesIO(data), filename)},
            content_type="multipart/form-data",
        )

    def test_valid_image_returns_label_and_confidence(self):
        response = self.post_image(make_image_bytes(), "leaf.jpg")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn(payload["prediction"], class_names)
        self.assertIsInstance(payload["confidence"], float)
        self.assertGreaterEqual(payload["confidence"], 0.0)
        self.assertLessEqual(payload["confidence"], 1.0)

    def test_real_validation_image_matches_direct_model_prediction(self):
        path = first_validation_image()
        raw = path.read_bytes()
        response = self.post_image(raw, path.name)
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()

        tensor = preprocess_bytes(raw, IMG_SIZE)
        probabilities = model.predict(
            np.expand_dims(tensor, axis=0), verbose=0
        )[0]
        expected_index = int(np.argmax(probabilities))
        self.assertEqual(payload["prediction"], class_names[expected_index])
        self.assertAlmostEqual(
            payload["confidence"], float(probabilities[expected_index]), places=5
        )

    def test_missing_file_part_returns_400(self):
        response = self.client.post("/predict", data={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_empty_filename_returns_400(self):
        response = self.client.post(
            "/predict",
            data={"file": (io.BytesIO(make_image_bytes()), "")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_empty_file_returns_400(self):
        response = self.post_image(b"", "empty.jpg")
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_file_over_configured_limit_returns_413(self):
        # Multipart overhead is allowed beyond the configured file-size limit.
        oversized_data = b"x" * (MAX_UPLOAD_BYTES + 1)
        response = self.post_image(oversized_data, "too-large.jpg")
        self.assertEqual(response.status_code, 413)
        self.assertIn("Maximum file size", response.get_json()["error"])

    def test_invalid_image_returns_400(self):
        response = self.post_image(b"this is not image data", "leaf.jpg")
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())


class MappingConsistencyTest(unittest.TestCase):
    def test_class_mapping_matches_model_outputs(self):
        mapping_path = os.path.join(BACKEND_DIR, "saved_model", "class_names.json")
        with open(mapping_path, "r", encoding="utf-8") as file:
            mapping = json.load(file)
        self.assertIsInstance(mapping, list)
        self.assertEqual(mapping, class_names)
        self.assertEqual(len(mapping), int(model.output_shape[-1]))

    def test_metrics_agree_with_mapping_and_model_input(self):
        metrics_path = os.path.join(
            BACKEND_DIR, "saved_model", "training_metrics.json"
        )
        if not os.path.isfile(metrics_path):
            self.skipTest("training_metrics.json not present")
        with open(metrics_path, "r", encoding="utf-8") as file:
            metrics = json.load(file)
        self.assertEqual(metrics["class_count"], len(class_names))
        self.assertEqual(metrics["class_names"], class_names)
        height, width = model.input_shape[1], model.input_shape[2]
        self.assertEqual(metrics["input_size"], [height, width])
        self.assertEqual(list(IMG_SIZE), [width, height])
        # Older tracked metrics predate the explicit test-set flag.  Newly
        # trained artifacts must carry it and must keep the test set unused.
        self.assertFalse(metrics.get("test_set_evaluated", False))

    def test_model_accepts_fixed_rgb_input_and_returns_15_classes(self):
        self.assertEqual(len(model.input_shape), 4)
        self.assertEqual(model.input_shape[-1], 3)
        self.assertIsNotNone(model.input_shape[1])
        self.assertIsNotNone(model.input_shape[2])
        self.assertEqual(tuple(model.output_shape), (None, len(class_names)))

    def test_preprocessing_matches_training_resize(self):
        path = first_validation_image()
        expected = preprocess_image(path, IMG_SIZE)
        loaded = tf.keras.utils.load_img(
            str(path),
            target_size=(IMG_SIZE[1], IMG_SIZE[0]),
            interpolation="bilinear",
        )
        actual = np.asarray(loaded, dtype=np.float32)
        self.assertEqual(actual.shape, expected.shape)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)

    def test_preprocessing_preserves_0_255_range(self):
        path = first_validation_image()
        tensor = preprocess_image(path, IMG_SIZE)
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(tensor.shape, (IMG_SIZE[1], IMG_SIZE[0], 3))
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 255.0)
        self.assertGreater(float(tensor.max()), 1.0)

    def test_preprocessing_is_deterministic(self):
        raw = make_image_bytes()
        first = preprocess_bytes(raw, IMG_SIZE)
        second = preprocess_bytes(raw, IMG_SIZE)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
