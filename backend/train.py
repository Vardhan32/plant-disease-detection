"""Train, validate, and optionally promote a plant-disease classifier.

The default command keeps the original project behaviour (train and update the
tracked model), while every experiment should use ``--no-promote`` so the
working application model is not touched:

    python backend/train.py --config backend/configs/aug_224_frozen.json

Training never reads the independent test split.  Model selection must use the
validation split only; run ``backend/evaluate.py --split test`` once after the
final configuration is selected.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.data_loading import build_image_dataset
from backend.data_pipeline import (
    DEFAULT_MANIFEST_PATH,
    REPO_DIR,
    load_manifest,
    manifest_sha256,
    validate_manifest,
)
from backend.evaluation import (
    CLASS_NAMES_JSON,
    compute_metrics,
    evaluate_rows,
    load_class_names,
    predict_rows,
)
from backend.modeling import (
    BASE_MODEL_NAME,
    build_model,
    configure_fine_tuning,
    load_trained_model,
)
from backend.preprocessing import preprocess_image

BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BACKEND_DIR / "saved_model"
DEFAULT_OUTPUT_ROOT = BACKEND_DIR / "runs"

MODERATE_AUGMENTATION = {
    "horizontal_flip": True,
    "rotation_degrees": 15,
    "zoom_fraction": 0.10,
    "translation_fraction": 0.05,
    "brightness_fraction": 0.10,
    "vertical_flip": False,
    "fill_mode": "nearest",
    "applied_to": "training batches only",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)
    config: dict[str, Any] = {}
    if pre_args.config is not None:
        with pre_args.config.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if not isinstance(config, dict):
            raise SystemExit(f"Config file must contain a JSON object: {pre_args.config}")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=pre_args.config)
    parser.add_argument("--run-name", default="run")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=0, help="0 selects 32 for 128px and 16 for 224px")
    parser.add_argument("--epochs", type=int, default=5, help="Frozen-head training epochs")
    parser.add_argument("--fine-tune-epochs", type=int, default=0)
    parser.add_argument("--unfreeze-last", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--dense-units", type=int, default=128)
    parser.add_argument(
        "--augmentation", choices=("none", "moderate"), default="none"
    )
    parser.add_argument(
        "--class-weights", choices=("none", "sqrt", "balanced"), default="none"
    )
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--reduce-lr-patience", type=int, default=1)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--subset-fraction", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--weights",
        default="imagenet",
        help="ImageNet weights, 'none', or a local weights path",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("macro_f1", "accuracy"),
        default="macro_f1",
        help="Validation metric used to choose between frozen and fine-tuned stages",
    )
    parser.add_argument(
        "--promote",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate and atomically replace the active saved_model artifacts",
    )
    parser.add_argument(
        "--init-model",
        type=Path,
        default=None,
        help="Start directly from an existing model; skips the frozen-head stage",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty run directory",
    )

    known_destinations = {action.dest for action in parser._actions}
    unknown = set(config) - known_destinations
    if unknown:
        raise SystemExit(f"Unknown config keys: {sorted(unknown)}")
    parser.set_defaults(**config)
    args = parser.parse_args(argv)

    for field in ("manifest", "output_root", "model_dir", "init_model", "config"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, Path(value))
    if args.batch_size <= 0:
        args.batch_size = 32 if args.resolution <= 160 else 16
    if not 0 < args.subset_fraction <= 1:
        raise SystemExit("--subset-fraction must be in (0, 1]")
    if args.epochs < 0 or args.fine_tune_epochs < 0:
        raise SystemExit("Epoch counts cannot be negative")
    if args.epochs == 0 and args.init_model is None:
        raise SystemExit("Provide --epochs > 0 or --init-model for fine-tuning")
    if any(character in args.run_name for character in ("/", "\\", "..")):
        raise SystemExit("--run-name must not contain path separators")
    return args


def set_reproducibility(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False)


def dataframe_subset(rows: list[dict[str, str]], fraction: float, seed: int):
    if fraction >= 1.0:
        return rows
    import pandas as pd
    from sklearn.model_selection import train_test_split

    frame = pd.DataFrame(rows)
    if frame["label"].value_counts().min() < 2:
        raise ValueError("Cannot subsample: at least one class has fewer than two images")
    sampled, _ = train_test_split(
        frame, train_size=fraction, stratify=frame["label"], random_state=seed
    )
    return sampled.to_dict("records")


def cap_rows(rows: list[dict[str, str]], max_samples: int, seed: int):
    if max_samples <= 0 or len(rows) <= max_samples:
        return rows
    import pandas as pd
    from sklearn.model_selection import train_test_split

    frame = pd.DataFrame(rows)
    sampled, _ = train_test_split(
        frame, train_size=int(max_samples), stratify=frame["label"], random_state=seed
    )
    return sampled.to_dict("records")


def make_datasets(
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    class_names: list[str],
    *,
    image_size: int,
    batch_size: int,
    augmentation: str,
    seed: int,
    class_weights: dict[int, float] | None,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    train_dataset = build_image_dataset(
        train_rows,
        class_names,
        (image_size, image_size),
        batch_size=batch_size,
        training=True,
        augmentation=augmentation,
        seed=seed,
        class_weights=class_weights,
    )
    val_dataset = build_image_dataset(
        val_rows,
        class_names,
        (image_size, image_size),
        batch_size=batch_size,
        training=False,
        augmentation="none",
        seed=seed,
    )
    return train_dataset, val_dataset


def class_weight_dict(train_rows: list[dict[str, str]], class_names: list[str], mode: str):
    if mode == "none":
        return None
    counts = Counter(row["label"] for row in train_rows)
    total = len(train_rows)
    classes = len(class_names)
    if mode == "balanced":
        weights = {
            name: total / (classes * counts[name])
            for name in class_names
        }
    elif mode == "sqrt":
        weights = {
            name: (total / (classes * counts[name])) ** 0.5
            for name in class_names
        }
    else:
        raise ValueError(f"Unknown class-weight mode: {mode}")
    return {
        class_names.index(name): float(weights[name])
        for name in class_names
    }


def make_callbacks(
    run_dir: Path,
    stage: str,
    args: argparse.Namespace,
) -> tuple[list[Any], Path]:
    checkpoint_path = run_dir / f"{stage}_best.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=args.early_stopping_patience,
            min_delta=args.min_delta,
            restore_best_weights=False,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=args.reduce_lr_patience,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(run_dir / f"{stage}_history.csv")),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    return callbacks, checkpoint_path


def history_to_dict(history) -> dict[str, list[float]]:
    return {
        key: [float(value) for value in values]
        for key, values in history.history.items()
        if values
    }


def last_epoch_metrics(history: dict[str, list[float]]) -> dict[str, float]:
    return {
        key: float(values[-1])
        for key, values in history.items()
        if values and key in {"accuracy", "loss", "val_accuracy", "val_loss"}
    }


def quick_metrics(
    model,
    rows: list[dict[str, str]],
    class_names: list[str],
    image_size: int,
    batch_size: int,
) -> dict[str, Any]:
    y_true, probabilities, _ = predict_rows(
        model, rows, class_names, (image_size, image_size), batch_size=batch_size, verbose=0
    )
    metrics = compute_metrics(y_true, probabilities, class_names)
    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "loss": metrics["loss"],
    }


def train_stage(
    model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    *,
    stage: str,
    learning_rate: float,
    epochs: int,
    run_dir: Path,
    args: argparse.Namespace,
) -> tuple[Any, Path, dict[str, list[float]]]:
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks, checkpoint_path = make_callbacks(run_dir, stage, args)
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        # The dataset already shuffles with a fixed seed.
        shuffle=False,
        verbose=args.verbose,
    )
    history_dict = history_to_dict(history)
    save_json(run_dir / f"{stage}_history.json", history_dict)
    if not checkpoint_path.is_file():
        raise RuntimeError(
            f"{stage} training did not produce a validation checkpoint; "
            "inspect the run logs for NaN or data errors."
        )
    best_model = load_trained_model(str(checkpoint_path))
    return best_model, checkpoint_path, history_dict


def validate_model_contract(
    model,
    class_names: list[str],
    image_size: int,
    sample_rows: list[dict[str, str]],
) -> None:
    if tuple(model.input_shape[1:]) != (image_size, image_size, 3):
        raise ValueError(
            f"Model input {model.input_shape} does not match configured "
            f"{image_size}x{image_size}x3"
        )
    if int(model.output_shape[-1]) != len(class_names):
        raise ValueError(
            f"Model outputs {model.output_shape[-1]} classes but mapping has "
            f"{len(class_names)}"
        )
    if not sample_rows:
        raise ValueError("No validation images available for a model smoke test")
    batch = np.stack(
        [
            preprocess_image(REPO_DIR / row["path"], (image_size, image_size))
            for row in sample_rows[:3]
        ]
    )
    probabilities = np.asarray(model.predict(batch, verbose=0))
    if probabilities.shape != (len(batch), len(class_names)):
        raise ValueError(f"Unexpected prediction shape: {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError("Model produced non-finite probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("Model probabilities do not sum to one")


def promote_artifacts(
    model,
    class_names: list[str],
    metrics: dict[str, Any],
    *,
    model_dir: Path,
    image_size: int,
    sample_rows: list[dict[str, str]],
) -> list[Path]:
    """Validate new artifacts, back up the active model, then atomically replace."""
    model_dir.mkdir(parents=True, exist_ok=True)
    active_model = model_dir / "plant_disease_model.h5"
    active_classes = model_dir / "class_names.json"
    active_metrics = model_dir / "training_metrics.json"
    temp_model = model_dir / "plant_disease_model.tmp.h5"
    temp_classes = model_dir / "class_names.tmp.json"
    temp_metrics = model_dir / "training_metrics.tmp.json"

    try:
        model.save(str(temp_model), include_optimizer=False)
        with temp_classes.open("w", encoding="utf-8") as handle:
            json.dump(class_names, handle, indent=2, ensure_ascii=False)
        with temp_metrics.open("w", encoding="utf-8") as handle:
            json.dump(json_safe(metrics), handle, indent=2, ensure_ascii=False)

        reloaded = load_trained_model(str(temp_model))
        validate_model_contract(reloaded, class_names, image_size, sample_rows)
        if metrics["class_count"] != len(class_names):
            raise ValueError("Metrics class_count does not match class mapping")
        if metrics["class_names"] != class_names:
            raise ValueError("Metrics class_names do not match class mapping")

        backups = []
        for active, backup in (
            (active_model, model_dir / "plant_disease_model_backup.h5"),
            (active_classes, model_dir / "class_names_backup.json"),
            (active_metrics, model_dir / "training_metrics_backup.json"),
        ):
            if active.is_file():
                shutil.copy2(active, backup)
                backups.append(backup)

        os.replace(temp_model, active_model)
        os.replace(temp_classes, active_classes)
        os.replace(temp_metrics, active_metrics)
        return [active_model, active_classes, active_metrics] + backups
    finally:
        for temp in (temp_model, temp_classes, temp_metrics):
            if temp.exists():
                temp.unlink()


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() or None
    except OSError:
        return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.output_root / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"Run directory already exists and is not empty: {run_dir}. "
            "Use a new run name or pass --overwrite."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    set_reproducibility(args.seed, args.deterministic)

    rows = load_manifest(args.manifest)
    split_validation = validate_manifest(rows)
    class_names = split_validation["class_names"]
    if class_names != load_class_names(CLASS_NAMES_JSON):
        raise SystemExit(
            "Dataset class order does not match backend/saved_model/class_names.json. "
            "Run backend/prepare_data.py and retrain before promoting a model."
        )
    train_rows = dataframe_subset(
        [row for row in rows if row["split"] == "train"],
        args.subset_fraction,
        args.seed,
    )
    train_rows = cap_rows(train_rows, args.max_train_samples, args.seed)
    val_rows = cap_rows(
        [row for row in rows if row["split"] == "val"], args.max_val_samples, args.seed
    )
    test_rows = [row for row in rows if row["split"] == "test"]
    if not train_rows or not val_rows:
        raise SystemExit("Training or validation split is empty")
    if set(row["label"] for row in train_rows) != set(class_names):
        raise SystemExit("Subsampled training split is missing one or more classes")

    config_payload = {
        "args": vars(args),
        "dataset": split_validation,
        "split_manifest_sha256": manifest_sha256(args.manifest),
        "git_commit": git_commit(),
        "tensorflow_version": tf.__version__,
        "keras_version": tf.keras.__version__,
        "numpy_version": np.__version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(run_dir / "run_config.json", config_payload)

    print(
        f"Run {args.run_name}: {len(train_rows)} train / {len(val_rows)} val "
        f"({len(test_rows)} test images reserved, not used) | "
        f"{args.resolution}px | augmentation={args.augmentation} | "
        f"class_weights={args.class_weights}",
        flush=True,
    )

    weights = None if str(args.weights).lower() == "none" else args.weights
    image_size = int(args.resolution)
    stage1_history: dict[str, list[float]] = {}
    stage2_history: dict[str, list[float]] = {}
    stage1_metrics: dict[str, Any] | None = None
    stage2_metrics: dict[str, Any] | None = None
    fine_tuning_info: dict[str, Any] | None = None
    started = time.time()

    if args.init_model is not None:
        print(f"Initialising from {args.init_model}; frozen-head stage skipped", flush=True)
        stage1_model = load_trained_model(str(args.init_model))
        validate_model_contract(stage1_model, class_names, image_size, val_rows)
        stage1_metrics = quick_metrics(
            stage1_model, val_rows, class_names, image_size, args.batch_size
        )
        stage1_path = args.init_model
        stage1_epochs = 0
    else:
        model, _ = build_model(
            len(class_names),
            image_size,
            weights=weights,
            dropout=args.dropout,
            dense_units=args.dense_units,
        )
        class_weights = class_weight_dict(train_rows, class_names, args.class_weights)
        train_dataset, val_dataset = make_datasets(
            train_rows,
            val_rows,
            class_names,
            image_size=image_size,
            batch_size=args.batch_size,
            augmentation=args.augmentation,
            seed=args.seed,
            class_weights=class_weights,
        )
        stage1_model, stage1_path, stage1_history = train_stage(
            model,
            train_dataset,
            val_dataset,
            stage="stage1_frozen",
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            run_dir=run_dir,
            args=args,
        )
        stage1_epochs = len(stage1_history.get("accuracy", []))
        stage1_metrics = quick_metrics(
            stage1_model, val_rows, class_names, image_size, args.batch_size
        )
        print(f"Frozen-head validation: {stage1_metrics}", flush=True)

    selected_model = stage1_model
    selected_path = stage1_path
    selected_stage = "frozen" if args.init_model is None else "initial"
    selected_metrics = stage1_metrics
    stage2_epochs = 0

    if args.fine_tune_epochs > 0:
        fine_tune_model = load_trained_model(str(stage1_path))
        fine_tuning_info = configure_fine_tuning(
            fine_tune_model, unfreeze_last_n=args.unfreeze_last
        )
        print(
            f"Fine-tuning {fine_tuning_info['trainable_base_layers']} base layers "
            f"({fine_tuning_info['frozen_batchnorm_layers']} BatchNorm layers kept frozen); "
            f"{fine_tuning_info['trainable_parameters']} trainable parameters",
            flush=True,
        )
        class_weights = class_weight_dict(train_rows, class_names, args.class_weights)
        train_dataset, val_dataset = make_datasets(
            train_rows,
            val_rows,
            class_names,
            image_size=image_size,
            batch_size=args.batch_size,
            augmentation=args.augmentation,
            seed=args.seed,
            class_weights=class_weights,
        )
        stage2_model, stage2_path, stage2_history = train_stage(
            fine_tune_model,
            train_dataset,
            val_dataset,
            stage="stage2_finetune",
            learning_rate=args.fine_tune_learning_rate,
            epochs=args.fine_tune_epochs,
            run_dir=run_dir,
            args=args,
        )
        stage2_epochs = len(stage2_history.get("accuracy", []))
        stage2_metrics = quick_metrics(
            stage2_model, val_rows, class_names, image_size, args.batch_size
        )
        print(f"Fine-tuned validation: {stage2_metrics}", flush=True)

        primary = args.selection_metric
        secondary = "accuracy" if primary == "macro_f1" else "macro_f1"
        stage1_key = (stage1_metrics[primary], stage1_metrics[secondary])
        stage2_key = (stage2_metrics[primary], stage2_metrics[secondary])
        if stage2_key > stage1_key:
            selected_model = stage2_model
            selected_path = stage2_path
            selected_stage = "fine_tuned"
            selected_metrics = stage2_metrics

    # Full validation report for the selected model, including per-class metrics
    # and a confusion matrix.  The test split is still untouched.
    final_metrics = evaluate_rows(
        selected_model,
        val_rows,
        class_names,
        split_name="val",
        output_dir=run_dir,
        batch_size=args.batch_size,
        image_size=(image_size, image_size),
        save_predictions=True,
        verbose=1,
    )
    selected_model.save(str(run_dir / "best_model.h5"), include_optimizer=False)
    save_json(run_dir / "class_names.json", class_names)
    if fine_tuning_info is not None:
        save_json(run_dir / "fine_tuning_info.json", fine_tuning_info)

    training_seconds = time.time() - started
    metrics_record = {
        "run_name": args.run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": vars(args),
        "git_commit": git_commit(),
        "tensorflow_version": tf.__version__,
        "keras_version": tf.keras.__version__,
        "numpy_version": np.__version__,
        "class_count": len(class_names),
        "class_names": class_names,
        "class_indices": {name: index for index, name in enumerate(class_names)},
        "input_size": [image_size, image_size],
        "preprocessing": {
            "resize": "PIL bilinear",
            "pixel_range": [0.0, 255.0],
            "efficientnet_internal_rescaling": True,
        },
        "augmentation": (
            "none" if args.augmentation == "none" else MODERATE_AUGMENTATION
        ),
        "augmentation_applied_to_training_only": True,
        "class_weights": args.class_weights,
        "fine_tuning": fine_tuning_info,
        "split_manifest": str(args.manifest),
        "split_manifest_sha256": manifest_sha256(args.manifest),
        "dataset": split_validation,
        "train_samples": len(train_rows),
        "validation_samples": len(val_rows),
        "test_samples": len(test_rows),
        "test_set_evaluated": False,
        "stage1_epochs": stage1_epochs,
        "stage2_epochs": stage2_epochs,
        "epochs_completed": stage1_epochs + stage2_epochs,
        "selected_stage": selected_stage,
        "selection_metric": args.selection_metric,
        "best_stage1_metrics": stage1_metrics,
        "best_stage2_metrics": stage2_metrics,
        "best_val_accuracy": final_metrics["accuracy"],
        "best_val_loss": final_metrics["loss"],
        "best_macro_f1": final_metrics["macro_f1"],
        "weighted_f1": final_metrics["weighted_f1"],
        "top3_accuracy": final_metrics["top3_accuracy"],
        "expected_calibration_error": final_metrics["expected_calibration_error"],
        "per_class": final_metrics["per_class"],
        "confusion_matrix": final_metrics["confusion_matrix"],
        "most_confused_pairs": final_metrics["most_confused_pairs"],
        "confidence": final_metrics["confidence"],
        "history": {"stage1": stage1_history, "stage2": stage2_history},
        "stage1_last_epoch_metrics": last_epoch_metrics(stage1_history),
        "stage2_last_epoch_metrics": last_epoch_metrics(stage2_history),
        "final_epoch_metrics": (
            last_epoch_metrics(stage2_history)
            if selected_stage == "fine_tuned"
            else last_epoch_metrics(stage1_history)
        ),
        "training_seconds": training_seconds,
        "resource_notes": {
            "device": "CPU",
            "batch_size": args.batch_size,
            "trainable_parameters": fine_tuning_info["trainable_parameters"]
            if fine_tuning_info
            else int(sum(np.prod(weight.shape) for weight in selected_model.trainable_weights)),
            "total_parameters": int(selected_model.count_params()),
        },
        "note": (
            "All reported metrics are from the group-aware validation split. "
            "The independent test split has not been evaluated. Validation "
            "accuracy is not test accuracy."
        ),
    }
    save_json(run_dir / "training_metrics.json", metrics_record)
    save_json(
        run_dir / "run_config.json",
        {
            **config_payload,
            "selected_stage": selected_stage,
            "training_seconds": training_seconds,
            "selected_checkpoint": str(selected_path),
        },
    )

    print(
        f"Selected {selected_stage} model: val accuracy="
        f"{final_metrics['accuracy']:.4f}, val macro-F1={final_metrics['macro_f1']:.4f}, "
        f"loss={final_metrics['loss']:.4f}",
        flush=True,
    )
    print(f"Run artifacts: {run_dir}", flush=True)

    if args.promote:
        promoted = promote_artifacts(
            selected_model,
            class_names,
            metrics_record,
            model_dir=args.model_dir,
            image_size=image_size,
            sample_rows=val_rows,
        )
        print("Promoted artifacts:", flush=True)
        for path in promoted:
            print(f"  {path}", flush=True)
    else:
        print(
            "Active saved_model artifacts were not changed (--no-promote).",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
