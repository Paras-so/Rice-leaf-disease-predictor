"""Export unreviewed leaf and lesion candidate instances from the cleaned dataset.

This colour/morphology baseline is not a trained disease segmentation model.
Touching objects can merge, and colour changes are not proof of disease.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage as ndi

from rice_disease.data import ROOT, load_manifest, write_json

METHOD = "colour_connected_components_v3"
CONNECTIVITY = np.ones((3, 3), dtype=bool)
PARAMETERS = {"max_side": 800, "leaf_min_fraction": 0.001,
              "lesion_min_pixels": 8, "strong_saturation": 0.30,
              "weak_saturation": 0.20, "foreground_blue_gap": 0.06,
              "lesion_red_green_ratio": 1.22,
              "lesion_min_red_green_difference": 0.045, "lesion_median_size": 3,
              "pale_seed_distance": 3}
LIMITATIONS = [
    "Unreviewed algorithm-generated candidates, not ground-truth annotations.",
    "Colour cannot distinguish disease from aging, shadows, lighting or nutrient stress.",
    "Touching/overlapping leaves or lesions can merge; fragmented objects can split.",
    "Small lesions may be lost at processing resolution; restored edges are approximate.",
    "Pale necrotic tissue can be missed and coloured background can be included.",
    "Conservative colour thresholds reduce false regions but can miss pale/low-contrast damage; an empty candidate mask does not mean healthy.",
    "No trained segmentation model, IoU, Dice, mask AP or severity estimate is produced.",
]


def components(binary, minimum):
    labels, count = ndi.label(binary, structure=CONNECTIVITY)
    sizes = np.bincount(labels.ravel())
    keep = np.flatnonzero(sizes >= minimum)
    keep = keep[keep != 0]
    if len(keep) > 65535:
        raise ValueError("Too many instances for uint16 masks")
    mapping = np.zeros(count + 1, dtype=np.uint16)
    mapping[keep] = np.arange(1, len(keep) + 1, dtype=np.uint16)
    return mapping[labels]


def candidate_masks(image, max_side=800):
    """Return native-size uint16 leaf and lesion IDs; never use class labels."""
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    small = rgb.copy()
    small.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    values = np.asarray(small, dtype=np.float32) / 255.0
    red, green, blue = np.moveaxis(values, -1, 0)
    maximum = values.max(axis=2)
    saturation = (maximum - values.min(axis=2)) / np.maximum(maximum, 1e-6)
    # Neutral/tinted paper and its shadows must not be grown into the leaf.
    # Requiring BOTH red and green above blue keeps yellow/brown leaf tissue
    # while rejecting the old red-only response to warm background shadows.
    vegetation_colour = ((np.minimum(red, green) - blue > PARAMETERS['foreground_blue_gap'])
                         & (green > blue * 1.25))
    strong = (saturation > PARAMETERS["strong_saturation"]) & vegetation_colour & (maximum > 0.10)
    weak = (saturation > PARAMETERS["weak_saturation"]) & vegetation_colour & (maximum > 0.08)
    foreground = ndi.binary_propagation(strong, structure=CONNECTIVITY, mask=weak)
    # Pad so morphology preserves leaves that meet the image boundary.
    foreground = np.pad(foreground, 2, mode="edge")
    foreground = ndi.binary_closing(foreground, structure=CONNECTIVITY)
    foreground = ndi.binary_fill_holes(foreground)[2:-2, 2:-2]
    leaves = components(foreground, max(20, int(foreground.size * PARAMETERS["leaf_min_fraction"])))
    # Brown/tan/necrotic-looking regions only; no forced lesions for disease classes
    # and no forced empty masks for healthy-class images.
    suspect = ((red > green * PARAMETERS["lesion_red_green_ratio"])
               & (red - green > PARAMETERS["lesion_min_red_green_difference"])
               & (red > blue * 1.25) & (saturation > 0.25) & (maximum > 0.12))
    # Pale colour alone is not disease: glare, paper and veins used to become
    # enormous false regions. Only extend a brown seed a few pixels into a
    # pale centre, never propagate across an entire pale leaf/background.
    seed = suspect & (leaves > 0)
    pale = (saturation < 0.22) & (maximum > 0.50)
    suspect |= pale & ndi.binary_dilation(seed, iterations=PARAMETERS['pale_seed_distance'])
    suspect = ndi.median_filter(suspect, size=PARAMETERS["lesion_median_size"])
    suspect &= leaves > 0
    lesions = np.zeros_like(leaves)
    offset = 0
    for leaf_id, region in enumerate(ndi.find_objects(leaves), start=1):
        if region is None:
            continue
        local = components(suspect[region] & (leaves[region] == leaf_id), PARAMETERS["lesion_min_pixels"])
        n = int(local.max())
        if offset + n > 65535:
            raise ValueError("Too many lesion instances for uint16 masks")
        view = lesions[region]
        selected = local > 0
        view[selected] = local[selected] + offset
        offset += n
    def restore(mask):
        return np.asarray(Image.fromarray(mask).resize(rgb.size, Image.Resampling.NEAREST), dtype=np.uint16)
    return rgb, restore(leaves), restore(lesions), small.size


def instance_records(labels, category, leaves=None):
    records = []
    sizes = np.bincount(labels.ravel())
    for instance_id, region in enumerate(ndi.find_objects(labels), start=1):
        if region is None:
            continue
        ys, xs = region
        record = {"instance_id": instance_id, "category": category,
                  "bbox_xywh": [xs.start, ys.start, xs.stop-xs.start, ys.stop-ys.start],
                  "area_pixels": int(sizes[instance_id])}
        if leaves is not None:
            parent_ids = np.unique(leaves[region][labels[region] == instance_id])
            if len(parent_ids) != 1 or parent_ids[0] == 0:
                raise ValueError("Lesion is not contained in exactly one leaf")
            record["leaf_instance_id"] = int(parent_ids[0])
        records.append(record)
    return records


def overlay(rgb, leaves, lesions):
    preview = rgb.copy()
    preview.thumbnail((640, 640), Image.Resampling.LANCZOS)
    leaf = np.asarray(Image.fromarray(leaves).resize(preview.size, Image.Resampling.NEAREST))
    lesion = np.asarray(Image.fromarray(lesions).resize(preview.size, Image.Resampling.NEAREST))
    pixels = np.asarray(preview).copy()
    # Leave leaf colours visible; only its boundary is cyan. A fixed red key
    # avoids mistaking a randomly coloured leaf instance for diseased tissue.
    foreground = leaf > 0
    pixels[foreground & ~ndi.binary_erosion(foreground)] = (0, 220, 255)
    chosen = lesion > 0
    pixels[chosen] = (pixels[chosen] * 0.55 + np.array([255, 40, 30]) * 0.45).astype(np.uint8)
    result = Image.fromarray(pixels)
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, result.width, 20), fill="black")
    draw.text((4, 4), 'DRAFT | cyan: leaf boundary | red: candidate, not confirmed disease', fill="white")
    return result


def prepare_image(row, dataset, max_side):
    source = (dataset / row["path"]).resolve()
    if not source.is_relative_to(dataset):
        raise ValueError("Source path is outside dataset")
    with source.open("rb") as source_file:
        digest = hashlib.file_digest(source_file, "sha256").hexdigest()
    if digest != row["sha256"]:
        raise ValueError("Source checksum differs from audited manifest")
    with Image.open(source) as image:
        rgb, leaves, lesions, work_size = candidate_masks(image, max_side)
    return digest, rgb, leaves, lesions, work_size


def prepared_images(rows, dataset, max_side):
    """Keep four preparations in flight without retaining the whole image dataset."""
    with ThreadPoolExecutor(max_workers=4) as executor:
        pending = {i: executor.submit(prepare_image, row, dataset, max_side)
                   for i, row in enumerate(rows[:4])}
        for i, row in enumerate(rows):
            future = pending.pop(i)
            yield row, future
            if i + 4 < len(rows):
                pending[i+4] = executor.submit(prepare_image, rows[i+4], dataset, max_side)


def run(artifacts, output, max_side=800, limit=None, split=None):
    rows, split_summary = load_manifest(artifacts)
    dataset = Path(split_summary["dataset"]).resolve()
    output = output.resolve()
    if output == dataset or output.is_relative_to(dataset) or dataset.is_relative_to(output):
        raise ValueError("Output must be separate from the source dataset")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output is not empty; choose a new --output to preserve previous results")
    if split:
        rows = [row for row in rows if row["split"] == split]
    if limit:
        # Round-robin classes for a representative training-only pilot.
        groups = {c: [r for r in rows if r['class_name'] == c] for c in sorted({r['class_name'] for r in rows})}
        rows = [group[i] for i in range(max(map(len, groups.values()))) for group in groups.values() if i < len(group)][:limit]
    output.mkdir(parents=True, exist_ok=True)
    summary = {"method": METHOD, "annotation_status": "unreviewed_pseudo_masks",
               "dataset": str(dataset), "manifest_sha256": split_summary["manifest_sha256"],
               "parameters": dict(PARAMETERS, max_side=max_side), "limitations": LIMITATIONS,
               "requested_images": len(rows), "processed": 0, "failed": 0,
               "leaf_instances": 0, "lesion_candidates": 0, "review_flags": {},
               "by_class": {}, "by_split": {}, "failures": [], "metrics": None}
    write_json(output / "summary.json", dict(summary, status="running"))
    counts, split_counts, flags_count = {}, Counter(), Counter()
    previews = {}
    start = time.monotonic()
    with (output / "instances.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["source_path", "split", "class_name", "leaf_instances", "lesion_candidates", "leaf_pixels", "candidate_lesion_pixels", "review_flags", "metadata"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, (row, future) in enumerate(prepared_images(rows, dataset, max_side), 1):
            try:
                digest, rgb, leaves, lesions, work_size = future.result()
                leaf_records = instance_records(leaves, "leaf")
                lesion_records = instance_records(lesions, "lesion_candidate", leaves)
                leaf_pixels = int(np.count_nonzero(leaves))
                lesion_pixels = int(np.count_nonzero(lesions))
                flags = ["requires_human_review"]
                if not leaf_records:
                    flags.append("no_leaf_detected")
                if len(leaf_records) > 1:
                    flags.append("multiple_leaf_components")
                if leaf_pixels > leaves.size * 0.85:
                    flags.append("high_foreground_coverage")
                if row["class_name"] == "healthy" and lesion_records:
                    flags.append("candidate_on_healthy_image")
                if row["class_name"] != "healthy" and not lesion_records:
                    flags.append("no_candidate_on_diseased_image")
                if leaf_pixels and lesion_pixels / leaf_pixels > 0.5:
                    flags.append("high_candidate_coverage")
                if any(r['area_pixels'] > leaf_pixels * 0.25 for r in lesion_records):
                    flags.append('large_connected_candidate_review')
                if len(leaf_records) > 3 or leaf_pixels > leaves.size * 0.85:
                    flags.append('complex_foreground_review')
                relative = Path(row["split"]) / row["class_name"] / Path(row["path"]).name
                files = {kind: Path(kind) / relative.parent / (relative.name + suffix)
                         for kind, suffix in (("leaf_instances", ".png"), ("lesion_instances", ".png"),
                                              ("overlays", ".jpg"), ("metadata", ".json"))}
                for path in files.values():
                    (output / path).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(leaves).save(output / files["leaf_instances"], compress_level=3)
                Image.fromarray(lesions).save(output / files["lesion_instances"], compress_level=3)
                preview = overlay(rgb, leaves, lesions)
                preview.save(output / files["overlays"], quality=88)
                previews.setdefault(row["class_name"], [])
                if row['split'] == 'train' and len(previews[row["class_name"]]) < 2:
                    tile = Image.new("RGB", (320, 350), "white")
                    shown = ImageOps.contain(preview, (320, 320))
                    tile.paste(shown, (0, 25))
                    ImageDraw.Draw(tile).text((4, 5), row["class_name"], fill="black")
                    previews[row["class_name"]].append(tile)
                metadata = {"source_path": row["path"], "source_sha256": digest,
                            "split": row["split"], "image_class": row["class_name"],
                            "annotation_status": "unreviewed_pseudo_masks", "method": METHOD,
                            "quality_status": "requires_pixel_review",
                            "width": rgb.width, "height": rgb.height, "processing_size": work_size,
                            "mask_encoding": "uint16 PNG, 0=background; IDs are local to each image and category",
                            "orientation": "EXIF corrected", "files": {k: v.as_posix() for k, v in files.items()},
                            "leaves": leaf_records, "lesion_candidates": lesion_records, "review_flags": flags}
                write_json(output / files["metadata"], metadata)
                writer.writerow({"source_path": row["path"], "split": row["split"], "class_name": row["class_name"],
                                 "leaf_instances": len(leaf_records), "lesion_candidates": len(lesion_records),
                                 "leaf_pixels": leaf_pixels, "candidate_lesion_pixels": lesion_pixels,
                                 "review_flags": ";".join(flags), "metadata": files["metadata"].as_posix()})
                summary["processed"] += 1
                summary["leaf_instances"] += len(leaf_records)
                summary["lesion_candidates"] += len(lesion_records)
                entry = counts.setdefault(row["class_name"], Counter())
                entry.update(images=1, leaf_instances=len(leaf_records), lesion_candidates=len(lesion_records))
                split_counts[row["split"]] += 1
                flags_count.update(flags)
            except (OSError, ValueError) as exc:
                summary["failed"] += 1
                summary["failures"].append({"path": row["path"], "error": str(exc)})
                print(f"FAILED {row['path']}: {exc}", flush=True)
            if index % 50 == 0 or index == len(rows):
                print(f"Processed {index}/{len(rows)} | {summary['leaf_instances']} leaves | {summary['lesion_candidates']} lesion candidates | {summary['failed']} failures", flush=True)
                write_json(output / 'summary.json', dict(summary, status='running',
                    by_class=counts, by_split=dict(split_counts), review_flags=dict(flags_count)))
    tiles = [tile for group in previews.values() for tile in group]
    if tiles:
        sheet = Image.new("RGB", (320*4, 350*((len(tiles)+3)//4)), "white")
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % 4)*320, (i//4)*350))
        sheet.save(output / "preview.jpg", quality=90)
    summary.update(status="complete" if not summary["failed"] else "complete_with_errors",
                   by_class=counts, by_split=dict(split_counts), review_flags=dict(flags_count),
                   elapsed_seconds=round(time.monotonic()-start, 2))
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Results: {output}", flush=True)
    return 1 if summary["failed"] else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/instance_segmentation")
    parser.add_argument("--max-side", type=int, default=800, help="Processing resolution; masks are saved at source dimensions")
    parser.add_argument("--limit", type=int, help="Class-balanced pilot image count")
    parser.add_argument("--split", choices=["train", "validation", "test"])
    args = parser.parse_args(argv)
    if args.max_side < 32 or (args.limit is not None and args.limit < 1):
        parser.error("--max-side must be >=32 and --limit must be positive")
    try:
        return run(args.artifacts, args.output, args.max_side, args.limit, args.split)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Instance segmentation error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
