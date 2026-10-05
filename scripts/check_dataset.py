"""Stage 1: inspect the cleaned rice dataset without changing any images."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path


CLASSES = (
    "bacterial_leaf_blight", "brown_spot", "healthy",
    "leaf_blast", "leaf_scald", "narrow_brown_spot",
)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LOCAL_DATASET = Path("Dataset/clean_dataset-20261001T114830Z-1-001/clean_dataset")
COLAB_DATASET = Path("/content/drive/MyDrive/RiceLeafsDisease/clean_dataset")


def default_dataset():
    candidates = [Path.cwd() / LOCAL_DATASET, COLAB_DATASET]
    if "__file__" in globals():
        candidates.insert(0, Path(__file__).resolve().parent.parent / LOCAL_DATASET)
    return next((path for path in candidates if path.is_dir()), candidates[0])


def check_dataset(dataset):
    """Return counts, exact-file duplicate findings, and environment metadata."""
    dataset = Path(dataset).expanduser().resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(
            f"Dataset not found: {dataset}\nSet the dataset path to clean_dataset. "
            "In Colab, mount Google Drive first."
        )

    issues = []
    counts = {}
    hashes = defaultdict(list)
    ignored_files = []
    for split in ("train", "test"):
        folder = dataset / split
        if not folder.is_dir():
            issues.append(f"Missing split: {split}")
            counts[split] = {}
            continue
        found_classes = {p.name for p in folder.iterdir() if p.is_dir()}
        if found_classes != set(CLASSES):
            issues.append(
                f"{split}: missing classes={sorted(set(CLASSES) - found_classes)}, "
                f"unexpected classes={sorted(found_classes - set(CLASSES))}"
            )
        split_counts = Counter({label: 0 for label in CLASSES})
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(dataset)
            if path.suffix.lower() not in EXTENSIONS:
                ignored_files.append(str(relative))
                continue
            parts = path.relative_to(folder).parts
            if len(parts) != 2 or parts[0] not in CLASSES:
                issues.append(f"Image outside the expected split/class/image layout: {relative}")
                continue
            split_counts[parts[0]] += 1
            try:
                with path.open("rb") as image_file:
                    digest = hashlib.file_digest(image_file, "sha256").hexdigest()
                hashes[digest].append({
                    "path": str(relative), "split": split, "class": parts[0],
                })
            except OSError as exc:
                issues.append(f"Cannot read {relative}: {exc}")
        counts[split] = dict(split_counts)
        for label, count in split_counts.items():
            if count == 0:
                issues.append(f"Empty class: {split}/{label}")

    totals = {split: sum(values.values()) for split, values in counts.items()}
    for split, expected in (("train", 1552), ("test", 389)):
        if totals[split] != expected:
            issues.append(f"{split}: expected {expected} images, found {totals[split]}")

    duplicate_groups = [group for group in hashes.values() if len(group) > 1]
    overlap = [g for g in duplicate_groups if len({x["split"] for x in g}) > 1]
    conflicts = [g for g in duplicate_groups if len({x["class"] for x in g}) > 1]
    if duplicate_groups:
        issues.append(f"Found {len(duplicate_groups)} exact-file duplicate groups")
    if overlap:
        issues.append(f"Train/test leakage: {len(overlap)} shared file hashes")
    if conflicts:
        issues.append(f"Conflicting labels: {len(conflicts)} shared file hashes")

    packages = {}
    for package in ("torch", "torchvision", "numpy", "Pillow", "scikit-learn"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    report = {
        "dataset": str(dataset), "counts": counts, "totals": totals,
        "unique_file_hashes": len(hashes), "duplicate_groups": duplicate_groups,
        "cross_split_duplicate_groups": overlap, "conflicting_label_groups": conflicts,
        "ignored_files": ignored_files, "issues": issues,
        "dataset_checks_passed": not issues,
        "environment": {"python": platform.python_version(), "packages": packages},
        "limitations": "Exact file hashes only; no image decoding or near-duplicate detection. "
        "Package versions do not verify imports, compatibility, or GPU availability.",
    }
    print(f"Dataset: {dataset}\n")
    print(f"{'Class':<25} {'Train':>7} {'Test':>7}")
    for label in CLASSES:
        print(f"{label:<25} {counts['train'].get(label, 0):>7} {counts['test'].get(label, 0):>7}")
    print(f"{'TOTAL':<25} {totals['train']:>7} {totals['test']:>7}")
    print(f"\nUnique file hashes: {len(hashes)}")
    print(f"Duplicate groups: {len(duplicate_groups)}")
    print(f"Train/test overlapping groups: {len(overlap)}")
    print(f"Conflicting-label groups: {len(conflicts)}")
    print(f"Ignored non-image files: {len(ignored_files)}")
    print(f"\nPython: {platform.python_version()}")
    for package, version in packages.items():
        print(f"{package}: {version or 'NOT INSTALLED'}")
    print("\nDataset checks: " + ("PASS" if not issues else "NEEDS REVIEW"))
    for issue in issues:
        print(f"- {issue}")
    if not packages["torch"] or not packages["torchvision"]:
        print("PyTorch/torchvision setup is needed before model training.")
    print("\nScope: " + report["limitations"])
    print("No images were modified.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=default_dataset())
    args = parser.parse_args()
    report = check_dataset(args.dataset)
    return 0 if report["dataset_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
