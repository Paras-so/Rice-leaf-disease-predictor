"""Audit source images and create reproducible, non-destructive split manifests."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

CLASSES = ["bacterial_leaf_blight", "brown_spot", "healthy", "leaf_blast",
           "leaf_scald", "narrow_brown_spot"]
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "Dataset/clean_dataset-20261001T114830Z-1-001/clean_dataset"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def prepare(dataset=DEFAULT_DATASET, output=ROOT / "artifacts", seed=42):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    rows, errors = [], []
    dimensions, modes = Counter(), Counter()
    pixels = defaultdict(list)
    for original_split in ("train", "test"):
        folder = dataset / original_split
        if not folder.is_dir():
            raise ValueError(f"Missing source split: {folder}")
        directories = {p.name for p in folder.iterdir() if p.is_dir()}
        if directories != set(CLASSES):
            raise ValueError(f"Unexpected class directories in {folder}: {sorted(directories)}")
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            if path.parent.name not in CLASSES or path.parent.parent != folder:
                errors.append({"path": str(path), "error": "Unexpected image directory layout"})
                continue
            try:
                with path.open("rb") as stream:
                    file_hash = hashlib.file_digest(stream, "sha256").hexdigest()
                with Image.open(path) as image:
                    image.load()  # Decode completely; header verification alone misses corruption.
                    dimensions[f"{image.width}x{image.height}"] += 1
                    modes[image.mode] += 1
                    rgb = ImageOps.exif_transpose(image).convert("RGB")
                    pixel_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                row = {"path": path.relative_to(dataset).as_posix(),
                       "class_name": path.parent.name, "label": CLASSES.index(path.parent.name),
                       "original_split": original_split, "sha256": file_hash,
                       "pixel_sha256": pixel_hash}
                rows.append(row)
                pixels[pixel_hash].append(row["path"])
            except (OSError, ValueError) as exc:
                errors.append({"path": str(path), "error": str(exc)})
            if len(rows) % 250 == 0:
                print(f"Decoded {len(rows)} images", flush=True)

    duplicates = [paths for paths in pixels.values() if len(paths) > 1]
    counts = Counter(row["class_name"] for row in rows)
    audit = {"dataset": str(dataset), "total_images": len(rows), "class_counts": dict(counts),
             "dimensions": dict(dimensions), "color_modes": dict(modes), "unreadable": errors,
             "identical_pixel_groups": duplicates,
             "imbalance_max_min_ratio": max(counts.values()) / min(counts.values()) if counts else None,
             "segmentation_masks": "No annotation files or mask directories found in the supplied cleaned dataset.",
             "limitations": "Pixel hashes do not detect transformed near-duplicates or shared plant identity."}
    write_json(output / "dataset_audit.json", audit)
    if errors or duplicates or len(rows) != 1941 or set(counts) != set(CLASSES):
        raise ValueError("Dataset needs review; see dataset_audit.json. No split was written.")

    pool = [row for row in rows if row["original_split"] == "train"]
    test = [row for row in rows if row["original_split"] == "test"]
    if len(pool) != 1552 or len(test) != 389:
        raise ValueError("Source split counts changed; review before defining the final experiment.")
    train, validation = train_test_split(pool, test_size=0.20, random_state=seed,
                                        stratify=[row["label"] for row in pool])
    final = []
    for split, subset in (("train", train), ("validation", validation), ("test", test)):
        final.extend(dict(row, split=split) for row in sorted(subset, key=lambda row: row["path"]))
    manifest = output / "split_manifest.csv"
    fieldnames = ["path", "class_name", "label", "original_split", "sha256", "pixel_sha256", "split"]
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(final)
    content = buffer.getvalue()
    if manifest.exists() and manifest.read_text(encoding="utf-8") != content:
        raise ValueError("Refusing to replace a different experimental split. Use another output directory.")
    manifest.write_text(content, encoding="utf-8", newline="")
    summary = {
        "dataset": str(dataset), "seed": seed, "classes": CLASSES,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "split_counts": {split: dict(Counter(r["class_name"] for r in final if r["split"] == split))
                         for split in ("train", "validation", "test")},
        "strategy": "Retain the verified existing 389-image test holdout; stratify 20% of the 1552-image development pool for validation.",
        "reason": "Avoid unnecessary test reassignment; validation supports model selection without test feedback. Overall fractions are approximately 64/16/20.",
        "test_policy": "No test feature extraction, prediction, or scoring until configuration and model selection are frozen.",
    }
    write_json(output / "split_summary.json", summary)
    print(json.dumps({"audit": audit, "split": summary}, indent=2), flush=True)
    return summary


def load_manifest(artifacts=ROOT / "artifacts"):
    artifacts = Path(artifacts)
    summary = json.loads((artifacts / "split_summary.json").read_text(encoding="utf-8"))
    path = artifacts / "split_manifest.csv"
    if hashlib.sha256(path.read_bytes()).hexdigest() != summary["manifest_sha256"]:
        raise ValueError("Split manifest has changed since preparation.")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["label"] = int(row["label"])
    return rows, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.dataset, args.output, args.seed)
