"""Model construction and fine-tuning helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

BASE_MODEL_NAME = "efficientnetb0"


def build_model(
    num_classes: int,
    image_size: int | tuple[int, int],
    *,
    weights: str | None = "imagenet",
    dropout: float = 0.3,
    dense_units: int = 128,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build the transfer-learning model used by the API.

    The architecture intentionally stays close to the original project:
    frozen ImageNet EfficientNetB0 -> global average pooling -> 128-unit ReLU
    -> dropout -> softmax.  Improvements come from resolution, augmentation,
    fine-tuning, and training controls, not from replacing the model family.
    """
    if isinstance(image_size, int):
        height = width = image_size
    else:
        height, width = image_size
    base_model = EfficientNetB0(
        weights=weights,
        include_top=False,
        input_shape=(int(height), int(width), 3),
        name=BASE_MODEL_NAME,
    )
    base_model.trainable = False

    x = GlobalAveragePooling2D(name="global_average_pooling2d")(base_model.output)
    x = Dense(int(dense_units), activation="relu", name="head_dense")(x)
    x = Dropout(float(dropout), name="head_dropout")(x)
    predictions = Dense(int(num_classes), activation="softmax", name="predictions")(x)
    model = Model(
        inputs=base_model.input,
        outputs=predictions,
        name="plant_disease_efficientnetb0",
    )
    return model, base_model


def configure_fine_tuning(
    model: tf.keras.Model,
    *,
    unfreeze_last_n: int = 40,
) -> dict[str, Any]:
    """Unfreeze only the top ``unfreeze_last_n`` EfficientNet layers.

    BatchNormalization layers stay in inference mode even when they fall inside
    the unfrozen range.  Updating their moving statistics on a small leaf
    dataset tends to destabilise transfer learning; this is the standard
    conservative approach.
    """
    # A model saved to HDF5 is reloaded as a flat graph: the nested
    # EfficientNetB0 layer boundary is not preserved.  Support both the freshly
    # built graph and the reloaded/flattened graph so fine-tuning can start from
    # any checkpoint produced by this project.
    try:
        base_model = model.get_layer(BASE_MODEL_NAME)
    except ValueError:
        base_model = None

    if base_model is not None:
        base_model.trainable = True
        layers = [
            layer
            for layer in base_model.layers
            if not isinstance(layer, tf.keras.layers.InputLayer)
        ]
    else:
        head_marker = "global_average_pooling2d"
        marker_indexes = [
            index
            for index, layer in enumerate(model.layers)
            if layer.name == head_marker
        ]
        if not marker_indexes:
            raise ValueError(
                "Could not locate the classification head. Expected a layer "
                f"named {head_marker!r}."
            )
        layers = [
            layer
            for layer in model.layers[: marker_indexes[0]]
            if not isinstance(layer, tf.keras.layers.InputLayer)
        ]

    if unfreeze_last_n <= 0:
        raise ValueError("unfreeze_last_n must be positive")
    if unfreeze_last_n >= len(layers):
        raise ValueError(
            f"unfreeze_last_n={unfreeze_last_n} is too large for "
            f"{len(layers)} base-model layers"
        )

    boundary = len(layers) - unfreeze_last_n
    trainable_names: list[str] = []
    frozen_batchnorm: list[str] = []
    for index, layer in enumerate(layers):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            frozen_batchnorm.append(layer.name)
        elif index >= boundary:
            layer.trainable = True
            trainable_names.append(layer.name)
        else:
            layer.trainable = False

    info = {
        "base_model_was_nested": base_model is not None,
        "base_layers": len(layers),
        "unfreeze_last_n": unfreeze_last_n,
        "first_unfrozen_layer_index": boundary,
        "trainable_base_layers": len(trainable_names),
        "trainable_base_layer_names": trainable_names,
        "frozen_batchnorm_layers": len(frozen_batchnorm),
        "trainable_parameters": int(
            sum(np.prod(weight.shape) for weight in model.trainable_weights)
        ),
        "total_parameters": int(model.count_params()),
    }
    return info


def load_trained_model(path: str, *, compile: bool = False) -> tf.keras.Model:
    """Load a saved model and validate the project contract.

    The API expects a 4D RGB input and a 15-class softmax output.  The class
    count is checked against ``class_names.json`` by the caller.
    """
    model = tf.keras.models.load_model(str(path), compile=compile)
    if len(model.input_shape) != 4:
        raise ValueError(f"Model input must be 4D, got {model.input_shape}")
    if model.input_shape[-1] != 3:
        raise ValueError(f"Model must accept 3-channel RGB input, got {model.input_shape}")
    if len(model.output_shape) != 2:
        raise ValueError(f"Model output must be 2D, got {model.output_shape}")
    return model
