"""Unit tests for evaluation metrics (no dataset or model needed)."""

import unittest

import numpy as np

from backend.evaluation import compute_metrics, expected_calibration_error

CLASS_NAMES = ["alpha", "beta", "gamma"]


def one_hot_predictions(indices, classes=3, confidence=0.9):
    probabilities = np.full((len(indices), classes), (1.0 - confidence) / (classes - 1))
    probabilities[np.arange(len(indices)), indices] = confidence
    return probabilities


class ExpectedCalibrationErrorTest(unittest.TestCase):
    def test_perfect_confident_predictions_have_zero_error(self):
        probabilities = np.eye(2, dtype=float)
        y_true = np.array([0, 1])
        self.assertAlmostEqual(
            expected_calibration_error(probabilities, y_true), 0.0, places=7
        )

    def test_confident_wrong_predictions_have_maximum_error(self):
        probabilities = np.eye(2, dtype=float)
        y_true = np.array([1, 0])
        self.assertAlmostEqual(
            expected_calibration_error(probabilities, y_true), 1.0, places=7
        )


class ComputeMetricsTest(unittest.TestCase):
    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 2, 0])
        probabilities = one_hot_predictions(y_true)
        metrics = compute_metrics(y_true, probabilities, CLASS_NAMES)

        self.assertEqual(metrics["samples"], 4)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0)
        self.assertAlmostEqual(metrics["weighted_f1"], 1.0)
        self.assertAlmostEqual(metrics["top3_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["loss"], -np.log(0.9), places=6)
        for name in CLASS_NAMES:
            self.assertEqual(metrics["per_class"][name]["support"], 1 if name != "alpha" else 2)

    def test_known_confusion_matrix_and_scores(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        probabilities = one_hot_predictions(y_pred)
        metrics = compute_metrics(y_true, probabilities, CLASS_NAMES)

        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0 / 3.0, places=6)
        self.assertEqual(
            metrics["confusion_matrix"],
            [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
        )
        # With three classes, the top-3 contains every class, so top-3 is 1.0.
        self.assertAlmostEqual(metrics["top3_accuracy"], 1.0)
        self.assertEqual(metrics["most_confused_pairs"][0]["count"], 1)

    def test_confidence_diagnostics(self):
        y_true = np.array([0, 1])
        # One confident correct prediction and one low-confidence correct one.
        probabilities = np.array([[0.9, 0.05, 0.05], [0.4, 0.35, 0.25]])
        metrics = compute_metrics(y_true, probabilities, CLASS_NAMES)
        confidence = metrics["confidence"]

        self.assertAlmostEqual(confidence["mean"], 0.65)
        self.assertAlmostEqual(confidence["high_confidence_coverage"], 0.5)
        self.assertAlmostEqual(confidence["high_confidence_accuracy"], 1.0)

    def test_rejects_one_dimensional_input(self):
        with self.assertRaises(ValueError):
            compute_metrics(np.array([0, 1]), np.array([0.2, 0.8]), CLASS_NAMES)


if __name__ == "__main__":
    unittest.main()
