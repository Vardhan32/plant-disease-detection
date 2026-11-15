"""Unit tests for the shared preprocessing contract (no model loading)."""

import io
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from backend.preprocessing import (
    MODEL_INPUT_RANGE,
    model_input_size,
    normalize_image_size,
    preprocess_bytes,
    preprocess_image,
    preprocess_pil_image,
)


class ImageSizeTest(unittest.TestCase):
    def test_accepts_int_and_tuple(self):
        self.assertEqual(normalize_image_size(128), (128, 128))
        self.assertEqual(normalize_image_size((224, 192)), (224, 192))

    def test_rejects_invalid_sizes(self):
        with self.assertRaises(ValueError):
            normalize_image_size((0, 128))
        with self.assertRaises(ValueError):
            normalize_image_size((128, 128, 3))

    def test_model_input_size_uses_height_width_order(self):
        class DummyModel:
            input_shape = (None, 224, 192, 3)

        self.assertEqual(model_input_size(DummyModel()), (192, 224))


class PreprocessingTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (300, 200), (20, 120, 40))
        # Add structure so a wrong resize/interpolation is detectable.
        for x in range(300):
            for y in range(0, 200, 7):
                self.image.putpixel((x, y), (200, y, 255 - x % 256))
        self.buffer = io.BytesIO()
        self.image.save(self.buffer, format="PNG")
        self.raw = self.buffer.getvalue()

    def test_preprocess_bytes_matches_path_loading(self):
        from_bytes = preprocess_bytes(self.raw, (128, 96))
        from_image = preprocess_pil_image(self.image, (128, 96))
        np.testing.assert_array_equal(from_bytes, from_image)

    def test_shape_dtype_and_range(self):
        tensor = preprocess_bytes(self.raw, (96, 128))
        self.assertEqual(tensor.shape, (128, 96, 3))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(MODEL_INPUT_RANGE, (0.0, 255.0))
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 255.0)
        self.assertGreater(float(tensor.max()), 1.0)

    def test_matches_pil_bilinear_resize(self):
        expected = np.asarray(
            self.image.resize((128, 96), Image.Resampling.BILINEAR), dtype=np.float32
        )
        actual = preprocess_pil_image(self.image, (128, 96))
        np.testing.assert_array_equal(actual, expected)

    def test_empty_bytes_are_rejected(self):
        with self.assertRaises(ValueError):
            preprocess_bytes(b"", (128, 128))

    def test_invalid_bytes_are_rejected(self):
        with self.assertRaises(OSError):
            preprocess_bytes(b"not an image", (128, 128))


class PathPreprocessingTest(unittest.TestCase):
    def test_preprocess_image_from_path(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "leaf.png"
            Image.new("RGB", (64, 64), (30, 140, 60)).save(path)
            tensor = preprocess_image(path, 128)
            self.assertEqual(tensor.shape, (128, 128, 3))
            self.assertEqual(tensor.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
