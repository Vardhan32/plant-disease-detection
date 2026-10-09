"""Backend tests for the plant disease detection Flask app.

Run from the repository root with the project virtual environment::

    .\\.venv\\Scripts\\python.exe -m unittest discover -s backend -p "test_*.py" -t .

The tests load the real tracked model via ``backend.app`` (no retraining),
so TensorFlow and the other ``requirements.txt`` dependencies must be
installed. Model loading takes ~30 s on CPU; that cost is paid once per run.
"""

import io
import json
import os
import sys
import unittest

from PIL import Image

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BACKEND_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from backend.app import MAX_UPLOAD_BYTES, app, class_names, model  # noqa: E402


def make_image_bytes(fmt="JPEG", size=(128, 128), color=(34, 139, 34)):
    """Build a small in-memory leaf-like image for upload tests."""
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


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
        mapping_path = os.path.join(
            BACKEND_DIR, "saved_model", "class_names.json"
        )
        with open(mapping_path, "r", encoding="utf-8") as file:
            mapping = json.load(file)
        self.assertIsInstance(mapping, list)
        self.assertEqual(mapping, class_names)
        self.assertEqual(len(mapping), int(model.output_shape[-1]))

    def test_metrics_agree_with_mapping(self):
        metrics_path = os.path.join(
            BACKEND_DIR, "saved_model", "training_metrics.json"
        )
        if not os.path.isfile(metrics_path):
            self.skipTest("training_metrics.json not present")
        with open(metrics_path, "r", encoding="utf-8") as file:
            metrics = json.load(file)
        self.assertEqual(metrics["class_count"], len(class_names))
        self.assertEqual(metrics["class_names"], class_names)


if __name__ == "__main__":
    unittest.main()
