"""Tests for model construction, training controls, and class weights."""

import math
import unittest

from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    Input,
)
from tensorflow.keras.models import Model

from backend.modeling import configure_fine_tuning
from backend.train import class_weight_dict


def build_flattened_model():
    """A small graph shaped like a reloaded H5 model: no nested base layer."""
    inputs = Input(shape=(32, 32, 3), name="input_layer")
    x = Conv2D(4, 3, padding="same", name="base_conv1")(inputs)
    x = BatchNormalization(name="base_bn1")(x)
    x = Conv2D(4, 3, padding="same", name="base_conv2")(x)
    x = BatchNormalization(name="base_bn2")(x)
    x = GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = Dense(8, activation="relu", name="head_dense")(x)
    outputs = Dense(3, activation="softmax", name="predictions")(x)
    return Model(inputs, outputs)


class FineTuningTest(unittest.TestCase):
    def test_flattened_model_unfreezes_only_top_layers(self):
        model = build_flattened_model()
        info = configure_fine_tuning(model, unfreeze_last_n=2)

        self.assertFalse(info["base_model_was_nested"])
        # Last two base layers: conv2 (trainable) and bn2 (frozen).
        self.assertTrue(model.get_layer("base_conv2").trainable)
        self.assertFalse(model.get_layer("base_bn2").trainable)
        self.assertFalse(model.get_layer("base_conv1").trainable)
        self.assertFalse(model.get_layer("base_bn1").trainable)
        # The classification head must stay trainable.
        self.assertTrue(model.get_layer("head_dense").trainable)
        self.assertTrue(model.get_layer("predictions").trainable)
        self.assertGreater(info["trainable_parameters"], 0)

    def test_too_many_unfrozen_layers_is_rejected(self):
        model = build_flattened_model()
        with self.assertRaises(ValueError):
            configure_fine_tuning(model, unfreeze_last_n=99)

    def test_zero_layers_is_rejected(self):
        model = build_flattened_model()
        with self.assertRaises(ValueError):
            configure_fine_tuning(model, unfreeze_last_n=0)


class ClassWeightTest(unittest.TestCase):
    rows = [{"label": "alpha"}] * 4 + [{"label": "beta"}]

    def test_none_mode_returns_no_weights(self):
        self.assertIsNone(class_weight_dict(self.rows, ["alpha", "beta"], "none"))

    def test_balanced_weights_are_inverse_frequency(self):
        weights = class_weight_dict(self.rows, ["alpha", "beta"], "balanced")
        self.assertAlmostEqual(weights[0], 5.0 / (2 * 4))
        self.assertAlmostEqual(weights[1], 5.0 / (2 * 1))

    def test_sqrt_weights_damp_the_imbalance(self):
        weights = class_weight_dict(self.rows, ["alpha", "beta"], "sqrt")
        self.assertAlmostEqual(weights[0], math.sqrt(5.0 / (2 * 4)))
        self.assertAlmostEqual(weights[1], math.sqrt(5.0 / (2 * 1)))
        self.assertGreater(weights[1], weights[0])

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            class_weight_dict(self.rows, ["alpha", "beta"], "unknown")


if __name__ == "__main__":
    unittest.main()
