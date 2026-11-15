"""Evaluate a saved model on a manifest split.

Examples (from the repository root):

    python backend/evaluate.py --split val
    python backend/evaluate.py --split test --allow-test \\
        --output-dir backend/reports

The test split is guarded by ``--allow-test`` so it cannot be evaluated
accidentally while experiments are still being selected on validation data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.data_pipeline import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_ORIGINAL_VAL_PATH,
    REPO_DIR,
    load_manifest,
    manifest_sha256,
    rows_for_split,
    validate_manifest,
)
from backend.evaluation import evaluate_rows, load_class_names
from backend.modeling import load_trained_model
from backend.preprocessing import model_input_size

DEFAULT_MODEL_PATH = REPO_DIR / "backend" / "saved_model" / "plant_disease_model.h5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--class-names",
        type=Path,
        default=REPO_DIR / "backend" / "saved_model" / "class_names.json",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "original_val"),
        default="val",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required before evaluating the held-out test split",
    )
    parser.add_argument(
        "--no-predictions-csv",
        action="store_true",
        help="Do not write the per-image prediction CSV",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit(
            "Refusing to evaluate the independent test split without --allow-test. "
            "Select the final model on validation data first."
        )
    if args.split == "original_val":
        manifest_path = args.manifest or DEFAULT_ORIGINAL_VAL_PATH
    else:
        manifest_path = args.manifest or DEFAULT_MANIFEST_PATH

    model = load_trained_model(str(args.model))
    class_names = load_class_names(args.class_names)
    output_classes = int(model.output_shape[-1])
    if output_classes != len(class_names):
        raise SystemExit(
            f"Model outputs {output_classes} classes but {args.class_names} has "
            f"{len(class_names)}. Refusing to evaluate an inconsistent pair."
        )

    rows = load_manifest(manifest_path)
    if args.split == "original_val":
        selected_rows = [row for row in rows if row["split"] == "original_val"]
        if not selected_rows:
            selected_rows = rows
    else:
        validate_manifest(rows)
        selected_rows = rows_for_split(rows, args.split)

    image_size = model_input_size(model)
    model_name = args.model_name or args.model.stem
    output_dir = args.output_dir or (
        REPO_DIR / "backend" / "reports" / "evaluations" / f"{model_name}_{args.split}"
    )
    metrics = evaluate_rows(
        model,
        selected_rows,
        class_names,
        split_name=args.split,
        output_dir=output_dir,
        batch_size=args.batch_size,
        image_size=image_size,
        save_predictions=not args.no_predictions_csv,
        verbose=1,
    )
    metrics.update(
        {
            "model_path": str(args.model),
            "model_sha256": sha256_file(args.model),
            "class_names_path": str(args.class_names),
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256(manifest_path),
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    report_path = output_dir / f"{args.split}_evaluation.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"\nEvaluation report: {report_path}")
    print(f"Confusion matrix: {output_dir / f'{args.split}_confusion_matrix.png'}")
    print(
        f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"weighted_f1={metrics['weighted_f1']:.4f} top3={metrics['top3_accuracy']:.4f} "
        f"loss={metrics['loss']:.4f}"
    )
    print(f"{'class':45s} {'precision':>9s} {'recall':>7s} {'f1':>7s} {'support':>8s}")
    for name in class_names:
        values = metrics["per_class"][name]
        print(
            f"{name:45s} {values['precision']:9.3f} {values['recall']:7.3f} "
            f"{values['f1_score']:7.3f} {values['support']:8d}"
        )
    if metrics["most_confused_pairs"]:
        print("\nMost confused pairs:")
        for pair in metrics["most_confused_pairs"][:10]:
            print(f"  {pair['count']:4d}  {pair['true']} -> {pair['predicted']}")
    if args.split == "test":
        print(
            "\nThis test-set result must not be used to tune the model. "
            "Report it as a one-time held-out evaluation."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
