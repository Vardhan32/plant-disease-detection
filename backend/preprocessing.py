"""Image preprocessing shared by training, evaluation, and the Flask API.

EfficientNetB0 in this project is created with ``weights="imagenet"`` and no
``preprocessing_function``.  The model therefore contains its own rescaling
layer and expects RGB pixel values in the [0, 255] range.  Never divide by 255
before passing an image to the model.

Pillow's bilinear resize is used everywhere.  Training previously used Keras'
default nearest-neighbour resize while inference used Pillow's default bicubic
resize; that train/serve mismatch is fixed by keeping this module as the single
source of truth.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Tuple

import numpy as np
from PIL import Image

# PIL uses (width, height) while TensorFlow/Keras use (height, width).
ImageSize = Tuple[int, int]
RESIZE_INTERPOLATION = Image.Resampling.BILINEAR
MODEL_INPUT_RANGE = (0.0, 255.0)


def normalize_image_size(image_size: int | Tuple[int, int]) -> ImageSize:
    """Return a validated ``(width, height)`` tuple."""
    if isinstance(image_size, int):
        width = height = image_size
    else:
        if len(image_size) != 2:
            raise ValueError(f"image_size must have two values, got {image_size!r}")
        width, height = image_size
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size must be positive, got {image_size!r}")
    return width, height


def load_rgb_image(source: str | Path | BinaryIO) -> Image.Image:
    """Open an image and convert it to RGB without retaining the file handle."""
    with Image.open(source) as image:
        return image.convert("RGB")


def resize_rgb_image(image: Image.Image, image_size: int | Tuple[int, int]) -> Image.Image:
    """Resize an RGB image with the project-wide interpolation method."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image.resize(normalize_image_size(image_size), RESIZE_INTERPOLATION)


def preprocess_pil_image(
    image: Image.Image, image_size: int | Tuple[int, int]
) -> np.ndarray:
    """Convert an RGB PIL image to a float32 ``(H, W, 3)`` [0, 255] array."""
    resized = resize_rgb_image(image, image_size)
    return np.asarray(resized, dtype=np.float32)


def preprocess_image(
    source: str | Path | BinaryIO, image_size: int | Tuple[int, int]
) -> np.ndarray:
    """Load and preprocess an image from a path or binary file object."""
    return preprocess_pil_image(load_rgb_image(source), image_size)


def preprocess_bytes(raw: bytes, image_size: int | Tuple[int, int]) -> np.ndarray:
    """Preprocess encoded image bytes exactly as the Flask API does."""
    if not raw:
        raise ValueError("Uploaded image is empty")
    with Image.open(io.BytesIO(raw)) as image:
        rgb = image.convert("RGB")
        return preprocess_pil_image(rgb, image_size)


def model_input_size(model) -> ImageSize:
    """Infer ``(width, height)`` from a Keras model's input shape."""
    shape = model.input_shape
    if not isinstance(shape, (tuple, list)) or len(shape) != 4:
        raise ValueError(f"Expected a 4D model input, got {shape!r}")
    height, width = shape[1], shape[2]
    if not height or not width:
        raise ValueError(f"Model input spatial dimensions are not fixed: {shape!r}")
    return int(width), int(height)
