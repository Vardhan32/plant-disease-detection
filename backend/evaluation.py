"""Model evaluation: metrics, confusion matrices, and error analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from backend.data_loading import build_image_dataset
from backend.data_pipeline import REPO_DIR
from backend.preprocessing import ImageSize, model_input_size

CLASS_NAMES_JSON = REPO_DIR / "backend" / "saved_model" / "class_names.json"


def load_class_names(path: Path = CLASS_NAMES_JSON) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        class_names = json.load(handle)
    if not isinstance(class_names, list) or not class_names:
        raise ValueError(f"Invalid class mapping in {path}")
    if any(not isinstance(name, str) or not name for name in class_names):
        raise ValueError(f"Class mapping in {path} must be a list of non-empty strings")
    return class_names


def predict_rows(
    model,
    rows: list[dict[str, str]],
    class_names: list[str],
    image_size: ImageSize,
    *,
    batch_size: int = 32,
    verbose: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Predict an entire manifest split in manifest order.

    The dataset uses the same Pillow bilinear [0, 255] preprocessing as
    training and the Flask API.  No augmentation or shuffling is applied.
    """
    class_index = {name: index for index, name in enumerate(class_names)}
    missing = sorted({row["label"] for row in rows} - set(class_index))
    if missing:
        raise ValueError(f"Rows contain labels not present in class_names: {missing}")

    dataset = build_image_dataset(
        rows,
        class_names,
        image_size,
        batch_size=batch_size,
        training=False,
        augmentation="none",
    )
    probabilities = np.asarray(model.predict(dataset, verbose=verbose))
    y_true = np.asarray(
        [class_index[row["label"]] for row in rows], dtype=np.int64
    )
    if len(probabilities) != len(y_true):
        raise RuntimeError(
            f"Predicted {len(probabilities)} images but rows contain {len(y_true)} labels"
        )
    if probabilities.shape[1] != len(class_names):
        raise RuntimeError(
            f"Model outputs {probabilities.shape[1]} classes but mapping has "
            f"{len(class_names)}"
        )
    return y_true, probabilities, [row["path"] for row in rows]


def expected_calibration_error(
    probabilities: np.ndarray, y_true: np.ndarray, bins: int = 10
) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == 0:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence > lower) & (confidence <= upper)
        if not np.any(mask):
            continue
        error += float(mask.mean()) * abs(
            float(correctness[mask].mean()) - float(confidence[mask].mean())
        )
    return error


def compute_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, class_names: list[str]
) -> dict[str, Any]:
    if probabilities.ndim != 2:
        raise ValueError(f"Expected 2D probabilities, got {probabilities.shape}")
    y_pred = probabilities.argmax(axis=1)
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    top3_indices = np.argsort(probabilities, axis=1)[:, -3:]
    top3_correct = np.any(top3_indices == y_true[:, None], axis=1)
    confidence = probabilities.max(axis=1)
    high_confidence = confidence >= 0.8
    true_probabilities = probabilities[np.arange(len(y_true)), y_true]
    loss = float(-np.log(np.clip(true_probabilities, 1e-7, 1.0)).mean())
    confused_pairs = []
    for true_index in range(len(class_names)):
        for predicted_index in range(len(class_names)):
            if true_index == predicted_index:
                continue
            count = int(matrix[true_index, predicted_index])
            if count:
                confused_pairs.append(
                    {
                        "true": class_names[true_index],
                        "predicted": class_names[predicted_index],
                        "count": count,
                    }
                )
    confused_pairs.sort(key=lambda item: (-item["count"], item["true"], item["predicted"]))
    return {
        "samples": int(len(y_true)),
        "class_names": class_names,
        "loss": loss,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # Pass explicit labels so macro-F1 always averages over every class in
        # the model contract, even if a class is missing from one subset.
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "top3_accuracy": float(top3_correct.mean()),
        "expected_calibration_error": expected_calibration_error(probabilities, y_true),
        "confidence": {
            "mean": float(confidence.mean()),
            "median": float(np.median(confidence)),
            "high_confidence_threshold": 0.8,
            "high_confidence_coverage": float(high_confidence.mean()),
            "high_confidence_accuracy": float(
                (y_pred[high_confidence] == y_true[high_confidence]).mean()
            )
            if np.any(high_confidence)
            else None,
        },
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1_score": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in class_names
        },
        "confusion_matrix": matrix.tolist(),
        "most_confused_pairs": confused_pairs[:15],
        "prediction_distribution": {
            name: int(count)
            for name, count in zip(class_names, np.bincount(y_pred, minlength=len(class_names)))
        },
    }


def save_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
    *,
    title: str,
) -> None:
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0
    )
    figure, axis = plt.subplots(figsize=(13.5, 11.5))
    image = axis.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=90, ha="center", fontsize=7)
    plt.setp(axis.get_yticklabels(), fontsize=7)
    threshold = normalized.max() / 2.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            count = int(matrix[row, column])
            if count == 0:
                continue
            axis.text(
                column,
                row,
                str(count),
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if normalized[row, column] > threshold else "black",
            )
    axis.set_title(
        f"{title}\nColour = row-normalised rate; text = image count",
        fontsize=11,
        pad=14,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_predictions_csv(
    paths: list[str],
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    y_pred = probabilities.argmax(axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["path", "true_label", "predicted_label", "confidence"]
            + [f"prob_{name}" for name in class_names]
        )
        for row_index, path in enumerate(paths):
            writer.writerow(
                [
                    path,
                    class_names[int(y_true[row_index])],
                    class_names[int(y_pred[row_index])],
                    f"{float(probabilities[row_index, y_pred[row_index]]):.6f}",
                ]
                + [f"{float(value):.6f}" for value in probabilities[row_index]]
            )


def evaluate_rows(
    model,
    rows: list[dict[str, str]],
    class_names: list[str],
    *,
    split_name: str,
    output_dir: Path,
    batch_size: int = 32,
    image_size: ImageSize | None = None,
    save_predictions: bool = True,
    verbose: int = 1,
) -> dict[str, Any]:
    image_size = image_size or model_input_size(model)
    y_true, probabilities, paths = predict_rows(
        model,
        rows,
        class_names,
        image_size,
        batch_size=batch_size,
        verbose=verbose,
    )
    metrics = compute_metrics(y_true, probabilities, class_names)
    metrics.update(
        {
            "split": split_name,
            "input_size": [int(image_size[0]), int(image_size[1])],
            "preprocessing": {
                "resize": "PIL bilinear",
                "pixel_range": [0.0, 255.0],
                "efficientnet_internal_rescaling": True,
            },
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{split_name}_evaluation.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    save_confusion_matrix(
        np.asarray(metrics["confusion_matrix"]),
        class_names,
        output_dir / f"{split_name}_confusion_matrix.png",
        title=f"PlantVillage {split_name} split",
    )
    if save_predictions:
        save_predictions_csv(
            paths,
            y_true,
            probabilities,
            class_names,
            output_dir / f"{split_name}_predictions.csv",
        )
    print(
        f"[{split_name}] accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"top3={metrics['top3_accuracy']:.4f} "
        f"ece={metrics['expected_calibration_error']:.4f}"
    )
    return metrics
