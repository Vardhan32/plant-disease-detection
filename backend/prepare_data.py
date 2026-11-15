"""Audit PlantVillage and create the committed group-aware split manifest.

Usage (from the repository root):

    python backend/prepare_data.py

Re-running this command is safe: it rewrites the manifest only after the audit
and split validation succeed.  Use ``--reuse-audit`` to rebuild the split from
an existing ``backend/data/dataset_audit_details.json`` without re-reading all
images.

The independent test split is created here and is not used by training.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.data_pipeline import (
    DEFAULT_AUDIT_DETAILS_PATH,
    DEFAULT_AUDIT_SUMMARY_PATH,
    DEFAULT_DATA_DIR,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_ORIGINAL_VAL_PATH,
    REPO_DIR,
    assign_group_aware_split,
    audit_dataset,
    load_manifest,
    reconstruct_original_validation_split,
    summarize_audit,
    validate_manifest,
    write_audit_details,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_AUDIT_SUMMARY_PATH)
    parser.add_argument("--details", type=Path, default=DEFAULT_AUDIT_DETAILS_PATH)
    parser.add_argument("--original-val", type=Path, default=DEFAULT_ORIGINAL_VAL_PATH)
    parser.add_argument(
        "--reuse-audit",
        type=Path,
        default=None,
        help="Reuse an existing dataset_audit_details.json instead of re-reading images.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--skip-original-val",
        action="store_true",
        help="Do not write the reconstructed original 80/20 validation manifest.",
    )
    return parser.parse_args()


def load_details(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload["records"]
    groups = payload["groups"]
    corrupt = payload.get("corrupt", [])
    path_to_group = {
        image_path: group_id
        for group_id, paths in groups.items()
        for image_path in paths
    }
    return records, groups, path_to_group, corrupt


def main() -> None:
    args = parse_args()
    if args.reuse_audit is not None:
        print(f"Reusing audit from {args.reuse_audit}")
        records, groups, path_to_group, corrupt = load_details(args.reuse_audit)
    else:
        records, groups, path_to_group, corrupt = audit_dataset(
            args.data_dir, workers=args.workers
        )
        write_audit_details(args.details, records, groups, corrupt)
        print(f"Wrote local audit details to {args.details}")

    summary = summarize_audit(records, groups, corrupt, path_to_group)
    rows = assign_group_aware_split(
        records,
        path_to_group,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    validation = validate_manifest(rows, data_dir=args.data_dir)

    # Quantify duplicate leakage in the original deterministic 80/20 split.
    original_val_rows = reconstruct_original_validation_split(args.data_dir)
    original_val_paths = {row["path"] for row in original_val_rows}
    leaked_paths: set[str] = set()
    for paths in groups.values():
        in_val = [path for path in paths if path in original_val_paths]
        in_rest = [path for path in paths if path not in original_val_paths]
        if in_val and in_rest:
            leaked_paths.update(in_val)
    summary["original_split"] = {
        "description": (
            "Original ImageDataGenerator(validation_split=0.2) partition: "
            "first int(0.2*n) sorted files per class are validation."
        ),
        "validation_images": len(original_val_rows),
        "validation_class_counts": dict(
            sorted(Counter(row["label"] for row in original_val_rows).items())
        ),
        "validation_images_with_duplicate_in_training": len(leaked_paths),
        "validation_duplicate_leak_fraction": round(
            len(leaked_paths) / len(original_val_rows), 6
        )
        if original_val_rows
        else 0.0,
    }
    summary["split"] = validation
    summary["split_manifest"] = str(args.manifest.relative_to(REPO_DIR))
    summary["split_seed"] = args.seed
    summary["test_set_policy"] = (
        "The test split is excluded from training and model selection. "
        "It is evaluated once after the final configuration is selected."
    )

    write_manifest(args.manifest, rows)
    if not args.skip_original_val:
        write_manifest(args.original_val, original_val_rows)
        print(f"Wrote original validation manifest to {args.original_val}")

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    # Re-read what was actually written and validate it, not just the in-memory rows.
    persisted_rows = load_manifest(args.manifest)
    persisted_validation = validate_manifest(persisted_rows, data_dir=args.data_dir)

    print(f"Wrote split manifest to {args.manifest}")
    print(f"Wrote audit summary to {args.summary}")
    print(
        "Split counts: "
        + ", ".join(
            f"{split}={count}"
            for split, count in persisted_validation["split_counts"].items()
        )
    )
    print(f"Classes: {persisted_validation['class_count']}")
    print(f"Duplicate groups: {persisted_validation['duplicate_groups']}")
    print(
        "Original-split duplicate leakage: "
        f"{summary['original_split']['validation_images_with_duplicate_in_training']} "
        f"of {summary['original_split']['validation_images']} validation images "
        f"({summary['original_split']['validation_duplicate_leak_fraction']:.2%})"
    )
    print(
        f"{'class':45s} {'train':>6s} {'val':>5s} {'test':>5s}"
    )
    for label in persisted_validation["class_names"]:
        counts = persisted_validation["class_split_counts"][label]
        print(f"{label:45s} {counts['train']:6d} {counts['val']:5d} {counts['test']:5d}")


if __name__ == "__main__":
    main()
