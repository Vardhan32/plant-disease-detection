"""Reproducible dataset auditing and group-aware train/validation/test splits.

The project commits the generated manifest at ``backend/data/plantvillage_split.csv``.
The manifest is the contract used by training, evaluation, and tests:

* every image belongs to exactly one split,
* images that are exact or perceptual near-duplicates stay in the same split,
* every class is represented in train, validation, and test.

The independent test split is never used by ``train.py``.  It is reserved for
one final evaluation after model selection on the validation split.
"""

from __future__ import annotations

import csv
import hashlib
import os
import random
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

REPO_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_DIR / "dataset" / "PlantVillage"
DEFAULT_MANIFEST_PATH = REPO_DIR / "backend" / "data" / "plantvillage_split.csv"
DEFAULT_AUDIT_SUMMARY_PATH = REPO_DIR / "backend" / "data" / "dataset_audit_summary.json"
DEFAULT_AUDIT_DETAILS_PATH = REPO_DIR / "backend" / "data" / "dataset_audit_details.json"
DEFAULT_ORIGINAL_VAL_PATH = (
    REPO_DIR / "backend" / "data" / "original_validation_split.csv"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
SPLIT_NAMES = ("train", "val", "test")
_POPCOUNT = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)


def iter_image_files(data_dir: Path = DEFAULT_DATA_DIR) -> list[tuple[Path, str]]:
    """Return sorted ``(path, class_name)`` pairs for image files only."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}. "
            "Place the PlantVillage class folders inside dataset/PlantVillage/."
        )
    rows: list[tuple[Path, str]] = []
    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        files = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        rows.extend((path, class_dir.name) for path in files)
    if not rows:
        raise ValueError(f"No supported images found under {data_dir}")
    return rows


def class_names_from_data(data_dir: Path = DEFAULT_DATA_DIR) -> list[str]:
    """Return the sorted class names used for class-to-index mapping."""
    return sorted({class_name for _, class_name in iter_image_files(data_dir)})


def dhash64(image: Image.Image) -> int:
    """64-bit difference hash; used only for duplicate auditing."""
    gray = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def _audit_one(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    try:
        with Image.open(path) as image:
            image.load()
            mode = image.mode
            width, height = image.size
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            dhash = dhash64(image)
            signature = np.asarray(
                image.convert("L").resize((32, 32), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
        return {
            "path": path.relative_to(REPO_DIR).as_posix(),
            "class": path.parent.name,
            "mode": mode,
            "size": [int(width), int(height)],
            "sha256": digest,
            "dhash": dhash,
            "signature": signature.tobytes(),
        }
    except Exception as exc:  # noqa: BLE001 - report every unreadable file
        return {
            "path": path.relative_to(REPO_DIR).as_posix(),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _popcount_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distance for uint64 arrays using a byte lookup table."""
    xor = np.bitwise_xor(left[:, None], right[None, :])
    return _POPCOUNT[xor.view(np.uint8).reshape(*xor.shape, 8)].sum(axis=-1)


def _find(parent: list[int], index: int) -> int:
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


def _union(parent: list[int], left: int, right: int) -> None:
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root != right_root:
        parent[right_root] = left_root


def build_duplicate_groups(
    records: list[dict[str, Any]],
    *,
    near_threshold: int = 6,
    signature_mse_threshold: float = 150.0,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Build duplicate components from exact SHA-256 and perceptual dHash.

    Near-duplicate comparisons are restricted to images in the same class.  A
    candidate pair must have a dHash Hamming distance <= ``near_threshold`` and
    a mean squared error <= ``signature_mse_threshold`` on 32x32 grayscale
    thumbnails.  This is conservative compared with embedding-based dedup and
    reduces the chance of grouping merely similar-looking leaves.
    """
    records = sorted(records, key=lambda item: item["path"])
    index_by_path = {record["path"]: index for index, record in enumerate(records)}
    parent = list(range(len(records)))

    exact: dict[str, list[str]] = defaultdict(list)
    for record in records:
        exact[record["sha256"]].append(record["path"])
    for paths in exact.values():
        for path in paths[1:]:
            _union(parent, index_by_path[paths[0]], index_by_path[path])

    by_class: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_class[record["class"]].append(index)

    signatures = {
        index: np.frombuffer(record["signature"], dtype=np.uint8)
        for index, record in enumerate(records)
    }
    for class_indexes in by_class.values():
        hashes = np.array(
            [records[index]["dhash"] for index in class_indexes], dtype=np.uint64
        )
        class_size = len(class_indexes)
        for start in range(0, class_size, 512):
            block = hashes[start : start + 512]
            distances = _popcount_matrix(block, hashes)
            rows, columns = np.nonzero(distances <= near_threshold)
            for row, column in zip(rows, columns):
                left_local = start + int(row)
                right_local = int(column)
                if right_local <= left_local:
                    continue
                left = class_indexes[left_local]
                right = class_indexes[right_local]
                left_signature = signatures[left].astype(np.float32)
                right_signature = signatures[right].astype(np.float32)
                mse = float(np.mean((left_signature - right_signature) ** 2))
                if mse <= signature_mse_threshold:
                    _union(parent, left, right)

    components: dict[int, list[str]] = defaultdict(list)
    for index, record in enumerate(records):
        components[_find(parent, index)].append(record["path"])

    groups: dict[str, list[str]] = {}
    path_to_group: dict[str, str] = {}
    for paths in components.values():
        paths = sorted(paths)
        digest = hashlib.sha1("\n".join(paths).encode("utf-8")).hexdigest()[:16]
        groups[digest] = paths
        for path in paths:
            path_to_group[path] = digest
    return groups, path_to_group


def audit_dataset(
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    workers: int | None = None,
    near_threshold: int = 6,
) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, str], list[dict[str, str]]]:
    """Read every image, check integrity, and group exact/near duplicates."""
    files = iter_image_files(data_dir)
    worker_count = workers or min(10, os.cpu_count() or 1)
    records: list[dict[str, Any]] = []
    corrupt: list[dict[str, str]] = []
    started = time.time()
    print(
        f"Auditing {len(files)} images from {len({label for _, label in files})} "
        f"classes with {worker_count} workers...",
        flush=True,
    )
    paths = [str(path) for path, _ in files]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        for result in executor.map(_audit_one, paths, chunksize=32):
            if "error" in result:
                corrupt.append({"path": result["path"], "error": result["error"]})
                continue
            records.append(result)
    records.sort(key=lambda item: item["path"])
    print(
        f"Audited {len(records)} readable images in {time.time() - started:.1f}s; "
        f"{len(corrupt)} unreadable.",
        flush=True,
    )

    groups, path_to_group = build_duplicate_groups(
        records, near_threshold=near_threshold
    )
    return records, groups, path_to_group, corrupt


def summarize_audit(
    records: list[dict[str, Any]],
    groups: dict[str, list[str]],
    corrupt: list[dict[str, str]],
    path_to_group: dict[str, str],
) -> dict[str, Any]:
    class_counts = Counter(record["class"] for record in records)
    modes = Counter(record["mode"] for record in records)
    sizes = Counter(tuple(record["size"]) for record in records)
    exact: dict[str, list[str]] = defaultdict(list)
    for record in records:
        exact[record["sha256"]].append(record["path"])
    exact_groups = [sorted(paths) for paths in exact.values() if len(paths) > 1]
    cross_class_exact = [
        paths
        for paths in exact_groups
        if len({Path(path).parent.name for path in paths}) > 1
    ]
    near_groups = [paths for paths in groups.values() if len(paths) > 1]
    cross_class_near = [
        paths
        for paths in near_groups
        if len({Path(path).parent.name for path in paths}) > 1
    ]
    multi_image_groups = [paths for paths in groups.values() if len(paths) > 1]
    largest_groups = sorted(
        (
            {
                "size": len(paths),
                "class": Path(paths[0]).parent.name,
                "examples": paths[:3],
            }
            for paths in multi_image_groups
        ),
        key=lambda item: (-item["size"], item["examples"]),
    )[:10]
    return {
        "data_dir": str(DEFAULT_DATA_DIR),
        "image_count": len(records),
        "class_count": len(class_counts),
        "class_counts": dict(sorted(class_counts.items())),
        "corrupt_images": corrupt,
        "color_modes": dict(modes),
        "image_sizes": {
            f"{width}x{height}": count
            for (width, height), count in sizes.most_common()
        },
        "exact_duplicate_group_count": len(exact_groups),
        "exact_duplicate_image_count": sum(len(paths) for paths in exact_groups),
        "cross_class_exact_duplicate_groups": cross_class_exact,
        "near_duplicate_group_count": len(near_groups),
        "near_duplicate_image_count": sum(len(paths) for paths in near_groups),
        "cross_class_near_duplicate_groups": cross_class_near,
        "group_count": len(groups),
        "multi_image_group_count": len(multi_image_groups),
        "group_size_histogram": dict(
            Counter(len(paths) for paths in groups.values()).most_common()
        ),
        "largest_duplicate_groups": largest_groups,
        "duplicate_method": {
            "exact": "SHA-256 over file bytes",
            "near": (
                "64-bit dHash Hamming distance <= 6 plus mean squared error "
                "<= 150 on 32x32 grayscale thumbnails, same class only"
            ),
        },
    }


def assign_group_aware_split(
    records: list[dict[str, Any]],
    path_to_group: dict[str, str],
    *,
    seed: int = 42,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> list[dict[str, str]]:
    """Assign whole duplicate groups to train/val/test with class stratification.

    Groups are filled smallest-first into validation and test so rare classes
    still receive a proportional share.  The remainder goes to training.
    """
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("val_fraction and test_fraction must be between 0 and 1")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be less than 1")

    labels = {record["path"]: record["class"] for record in records}
    group_ids = {record["path"]: path_to_group[record["path"]] for record in records}
    members: dict[str, list[str]] = defaultdict(list)
    for path in labels:
        members[group_ids[path]].append(path)
    for group_id, paths in members.items():
        group_labels = {labels[path] for path in paths}
        if len(group_labels) != 1:
            raise ValueError(
                f"Duplicate group {group_id} spans multiple classes: {sorted(group_labels)}. "
                "Resolve the conflicting labels before creating a split."
            )

    assignment: dict[str, str] = {}
    classes = sorted(set(labels.values()))
    for class_name in classes:
        class_groups = [
            sorted(paths)
            for paths in members.values()
            if labels[paths[0]] == class_name
        ]
        class_size = sum(len(paths) for paths in class_groups)
        target_val = round(val_fraction * class_size)
        target_test = round(test_fraction * class_size)

        # Deterministic shuffle inside size buckets, then smallest-first.
        buckets: dict[int, list[list[str]]] = defaultdict(list)
        for paths in class_groups:
            buckets[len(paths)].append(paths)
        rng = random.Random(seed + zlib.crc32(class_name.encode("utf-8")))
        ordered: list[list[str]] = []
        for size in sorted(buckets):
            bucket = buckets[size]
            rng.shuffle(bucket)
            ordered.extend(bucket)

        used: set[str] = set()
        test_groups: list[list[str]] = []
        val_groups: list[list[str]] = []
        for paths in ordered:
            if sum(len(item) for item in test_groups) + len(paths) <= target_test:
                test_groups.append(paths)
                used.add(paths[0])
        for paths in ordered:
            if paths[0] in used:
                continue
            if sum(len(item) for item in val_groups) + len(paths) <= target_val:
                val_groups.append(paths)
                used.add(paths[0])

        # Guarantee representation for tiny classes.
        if not test_groups and len(class_groups) >= 3:
            smallest = min(class_groups, key=lambda item: (len(item), item[0]))
            test_groups.append(smallest)
            used.add(smallest[0])
        if not val_groups and len(class_groups) >= 2:
            remaining = [
                item for item in class_groups if item[0] not in used
            ]
            if remaining:
                smallest = min(remaining, key=lambda item: (len(item), item[0]))
                val_groups.append(smallest)
                used.add(smallest[0])

        for paths in test_groups:
            for path in paths:
                assignment[path] = "test"
        for paths in val_groups:
            for path in paths:
                assignment[path] = "val"
        for paths in class_groups:
            if paths[0] in used:
                continue
            for path in paths:
                assignment[path] = "train"

    missing = set(labels) - set(assignment)
    if missing:
        raise RuntimeError(f"{len(missing)} images were not assigned to a split")
    return [
        {
            "path": path,
            "label": labels[path],
            "group_id": group_ids[path],
            "split": assignment[path],
        }
        for path in sorted(labels)
    ]


def validate_manifest(
    rows: list[dict[str, str]],
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    min_class_coverage: int = 1,
    check_files: bool = True,
) -> dict[str, Any]:
    """Validate split integrity and return a compact summary."""
    if not rows:
        raise ValueError("Split manifest is empty")
    required = {"path", "label", "split", "group_id"}
    missing_columns = required - set(rows[0])
    if missing_columns:
        raise ValueError(f"Split manifest is missing columns: {sorted(missing_columns)}")

    paths = [row["path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("Split manifest contains duplicate image paths")
    labels = sorted({row["label"] for row in rows})
    splits = Counter(row["split"] for row in rows)
    unknown_splits = set(splits) - set(SPLIT_NAMES)
    if unknown_splits:
        raise ValueError(f"Unknown split values: {sorted(unknown_splits)}")
    if set(splits) != set(SPLIT_NAMES):
        raise ValueError(f"Manifest must contain all splits, got {dict(splits)}")

    groups_by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups_by_split[row["group_id"]].add(row["split"])
    crossing = {
        group_id: values
        for group_id, values in groups_by_split.items()
        if len(values) > 1
    }
    if crossing:
        raise ValueError(f"{len(crossing)} duplicate groups cross splits")

    class_split_counts = Counter((row["label"], row["split"]) for row in rows)
    missing_class_splits = [
        (label, split)
        for label in labels
        for split in SPLIT_NAMES
        if class_split_counts[(label, split)] < min_class_coverage
    ]
    if missing_class_splits:
        raise ValueError(
            "Classes missing from a split: " + ", ".join(
                f"{label}/{split}" for label, split in missing_class_splits
            )
        )

    total = len(rows)
    fractions = {split: splits[split] / total for split in SPLIT_NAMES}
    if not 0.65 <= fractions["train"] <= 0.75:
        raise ValueError(f"Training fraction is outside 0.65-0.75: {fractions['train']:.3f}")
    if not 0.12 <= fractions["val"] <= 0.18:
        raise ValueError(f"Validation fraction is outside 0.12-0.18: {fractions['val']:.3f}")
    if not 0.12 <= fractions["test"] <= 0.18:
        raise ValueError(f"Test fraction is outside 0.12-0.18: {fractions['test']:.3f}")

    if check_files:
        missing_files = []
        for row in rows:
            file_path = Path(row["path"])
            if not file_path.is_absolute():
                file_path = REPO_DIR / file_path
            if not file_path.is_file():
                missing_files.append(row["path"])
        if missing_files:
            raise ValueError(
                f"{len(missing_files)} manifest files are missing, e.g. {missing_files[:3]}"
            )

    return {
        "image_count": total,
        "split_counts": dict(splits),
        "split_fractions": {split: round(value, 6) for split, value in fractions.items()},
        "class_count": len(labels),
        "class_names": labels,
        "class_split_counts": {
            label: {split: class_split_counts[(label, split)] for split in SPLIT_NAMES}
            for label in labels
        },
        "duplicate_groups": len(groups_by_split),
        "groups_crossing_splits": 0,
    }


def write_manifest(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "label", "split", "group_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def rows_for_split(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"No rows for split {split!r}")
    return selected


def manifest_sha256(path: Path = DEFAULT_MANIFEST_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconstruct_original_validation_split(
    data_dir: Path = DEFAULT_DATA_DIR,
) -> list[dict[str, str]]:
    """Reproduce the original ImageDataGenerator validation_split=0.2 selection.

    Keras' legacy DirectoryIterator uses the *first* ``int(0.2 * n)`` files in
    sorted order for the validation subset of each class (split=(0, 0.2)) and
    the remainder for training.  The original script passed shuffle=True, but
    that shuffles batch order, not the file partition.  This reconstruction is
    therefore deterministic and does not depend on a random seed.
    """
    rows: list[dict[str, str]] = []
    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        files = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        stop = int(0.2 * len(files))
        for path in files[:stop]:
            rows.append(
                {
                    "path": path.relative_to(REPO_DIR).as_posix(),
                    "label": class_dir.name,
                    "split": "original_val",
                    "group_id": "",
                }
            )
    return rows


def write_audit_details(
    path: Path,
    records: list[dict[str, Any]],
    groups: dict[str, list[str]],
    corrupt: list[dict[str, str]],
) -> None:
    """Write the full audit for local debugging; this file is gitignored."""
    payload = {
        "records": [
            {
                "path": record["path"],
                "class": record["class"],
                "mode": record["mode"],
                "size": record["size"],
                "sha256": record["sha256"],
                "dhash": record["dhash"],
            }
            for record in records
        ],
        "groups": groups,
        "corrupt": corrupt,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
