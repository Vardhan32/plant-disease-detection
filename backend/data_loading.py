"""Shared tf.data pipelines for training and evaluation.

Every image is decoded and resized by :mod:`backend.preprocessing` (Pillow,
bilinear, RGB, [0, 255]).  The Flask API uses the same function, so training,
validation, test evaluation, and serving cannot silently diverge.

Augmentation is implemented with Keras preprocessing layers and is applied to
the training batches only.  Validation and test pipelines never augment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf

from backend.data_pipeline import REPO_DIR
from backend.preprocessing import ImageSize, preprocess_image

AUTOTUNE = tf.data.AUTOTUNE
DEFAULT_SHUFFLE_BUFFER = 10000


def _resolve_paths(rows: Iterable[dict[str, str]]) -> list[str]:
    paths = []
    for row in rows:
        path = Path(row["path"])
        if not path.is_absolute():
            path = REPO_DIR / path
        paths.append(str(path))
    return paths


def _augmentation_layer(seed: int) -> tf.keras.Sequential:
    """Moderate, label-preserving augmentation for leaf images.

    Horizontal flips are safe for leaves.  Rotations, zoom, translation, and
    brightness are kept small so disease spots and leaf texture stay visible.
    Vertical flips and strong colour shifts are deliberately excluded.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(
                factor=15.0 / 360.0,
                fill_mode="nearest",
                seed=seed,
            ),
            tf.keras.layers.RandomZoom(
                height_factor=0.10,
                width_factor=0.10,
                fill_mode="nearest",
                seed=seed,
            ),
            tf.keras.layers.RandomTranslation(
                height_factor=0.05,
                width_factor=0.05,
                fill_mode="nearest",
                seed=seed,
            ),
            tf.keras.layers.RandomBrightness(
                factor=0.10,
                value_range=(0.0, 255.0),
                seed=seed,
            ),
        ],
        name="moderate_augmentation",
    )


def build_image_dataset(
    rows: list[dict[str, str]],
    class_names: list[str],
    image_size: ImageSize,
    *,
    batch_size: int,
    training: bool,
    augmentation: str = "none",
    seed: int = 42,
    class_weights: dict[int, float] | None = None,
    shuffle_buffer: int | None = None,
) -> tf.data.Dataset:
    """Build a batched ``tf.data`` dataset from manifest rows.

    ``class_weights`` are emitted as per-sample weights, which works with
    ``model.fit`` and keeps the weighted loss reproducible.
    """
    if not rows:
        raise ValueError("Cannot build a dataset from zero rows")
    class_index = {name: index for index, name in enumerate(class_names)}
    missing = sorted({row["label"] for row in rows} - set(class_index))
    if missing:
        raise ValueError(f"Rows contain labels not present in class_names: {missing}")

    width, height = int(image_size[0]), int(image_size[1])
    paths = _resolve_paths(rows)
    labels = np.asarray([class_index[row["label"]] for row in rows], dtype=np.int32)

    tensors: tuple[np.ndarray, ...] = (np.asarray(paths), labels)
    if class_weights is not None:
        sample_weights = np.asarray(
            [float(class_weights.get(int(label), 1.0)) for label in labels],
            dtype=np.float32,
        )
        tensors = (np.asarray(paths), labels, sample_weights)

    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    if training:
        buffer_size = shuffle_buffer or min(len(paths), DEFAULT_SHUFFLE_BUFFER)
        dataset = dataset.shuffle(
            buffer_size=max(1, buffer_size),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def load(path: tf.Tensor, label: tf.Tensor, *extra: tf.Tensor):
        image = tf.numpy_function(
            lambda raw: preprocess_image(raw.decode("utf-8"), (width, height)),
            [path],
            tf.float32,
        )
        image.set_shape((height, width, 3))
        one_hot = tf.one_hot(label, len(class_names))
        return (image, one_hot, *extra)

    dataset = dataset.map(load, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    if training and augmentation == "moderate":
        augmenter = _augmentation_layer(seed)
        dataset = dataset.map(
            lambda images, targets, *extra: (
                augmenter(images, training=True),
                targets,
                *extra,
            ),
            num_parallel_calls=AUTOTUNE,
        )
    elif training and augmentation != "none":
        raise ValueError(f"Unknown augmentation mode: {augmentation}")

    return dataset.prefetch(AUTOTUNE)


def cpu_count() -> int:
    """Small helper used for run metadata."""
    return os.cpu_count() or 1
