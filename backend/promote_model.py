"""Validate a completed experiment and promote it to the active saved model.

Experiments run with ``--no-promote`` so the serving model stays untouched
until model selection is finished.  Once a run is selected on validation data,
promote it with:

    python backend/promote_model.py --run-dir backend/runs/aug_224_frozen

The utility reloads the run's ``best_model.h5``, checks the 15-class contract,
copies the artifacts to temporary files, validates them again, and only then
atomically replaces ``backend/saved_model``.  The previous active artifacts are
kept as ``*_backup.*`` files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.data_pipeline import (
    DEFAULT_MANIFEST_PATH,
    REPO_DIR,
    load_manifest,
    rows_for_split,
    validate_manifest,
)
from backend.modeling import load_trained_model
from backend.preprocessing import model_input_size
from backend.train import promote_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_DIR / "backend" / "saved_model",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    model_path = run_dir / "best_model.h5"
    class_names_path = run_dir / "class_names.json"
    metrics_path = run_dir / "training_metrics.json"
    for path in (model_path, class_names_path, metrics_path):
        if not path.is_file():
            raise SystemExit(f"Missing required run artifact: {path}")

    rows = load_manifest(args.manifest)
    validate_manifest(rows)
    val_rows = rows_for_split(rows, "val")

    model = load_trained_model(str(model_path))
    class_names = load_json(class_names_path)
    metrics = load_json(metrics_path)
    if metrics.get("test_set_evaluated", False):
        raise SystemExit(
            "Refusing to promote a run whose metrics claim the test split was "
            "already evaluated. Select the model on validation data first."
        )

    image_size = model_input_size(model)
    promoted = promote_artifacts(
        model,
        class_names,
        metrics,
        model_dir=args.model_dir,
        image_size=image_size,
        sample_rows=val_rows,
    )
    print("Promoted run artifacts:")
    for path in promoted:
        print(f"  {path}")
    print(f"Active input size: {image_size[0]}x{image_size[1]} (width x height)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
