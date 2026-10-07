"""Full-leaf resize and deterministic training-only augmentation."""

import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs/benchmark.json"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def validate_config(config):
    """The shared JSON config is the source of image_size, mean and std."""
    if not isinstance(config, dict):
        raise ValueError("The preprocessing config must be a JSON object.")
    size = config.get("image_size")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("image_size must be a positive integer in the preprocessing config.")
    for key in ("mean", "std"):
        values = config.get(key)
        if not isinstance(values, (list, tuple)) or len(values) != 3 or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (key == "std" and value <= 0)
            for value in values
        ):
            raise ValueError(f"{key} must contain three finite RGB values; std values must be positive.")


def resized_image(image, config):
    """Return an EXIF-corrected RGB image at the configured square size."""
    validate_config(config)
    image = ImageOps.exif_transpose(image).convert("RGB")
    size = config["image_size"]
    return image.resize((size, size), Image.Resampling.BILINEAR)


def image_tensor(image, config, augmentation_seed=None):
    image = resized_image(image, config)
    if augmentation_seed is not None:
        rng = random.Random(augmentation_seed)
        if rng.random() < 0.5:
            image = ImageOps.mirror(image)
        if rng.random() < 0.5:
            image = ImageOps.flip(image)
        rotations = [None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180,
                     Image.Transpose.ROTATE_270]
        rotation = rng.choice(rotations)
        if rotation is not None:
            image = image.transpose(rotation)
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.9, 1.1))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.9, 1.1))
    return TF.normalize(TF.to_tensor(image), config["mean"], config["std"])


class LeafDataset(Dataset):
    def __init__(self, rows, root, config, cache, training=False):
        validate_config(config)
        self.rows, self.root, self.config = rows, Path(root), config
        self.cache, self.training = Path(cache), training
        self.cache.mkdir(parents=True, exist_ok=True)
        self.views = config["training_views"] if training else 1

    def __len__(self):
        return len(self.rows) * self.views

    def __getitem__(self, index):
        row_index, view = divmod(index, self.views)
        row = self.rows[row_index]
        cached = self.cache / f"{row['sha256']}_{self.config['image_size']}.png"
        if not cached.exists():
            with Image.open(self.root / row["path"]) as original:
                image = resized_image(original, self.config)
                image.save(cached)
        seed = self.config["seed"] + row_index * 100 + view if self.training and view else None
        with Image.open(cached) as image:
            tensor = image_tensor(image, self.config, seed)
        return tensor, torch.tensor(row["label"], dtype=torch.long)


def main(argv=None):
    parser = argparse.ArgumentParser(description=(
        "Resize images and report normalized tensor dimensions. With no --input, "
        "save one training-image preview per class from the prepared manifest."))
    parser.add_argument("--input", type=Path, help="Image or folder (searched recursively)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--image-size", type=int, help="Override the config size for this export only")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/preprocessing")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if args.image_size is not None:
            config["image_size"] = args.image_size
        validate_config(config)
        if args.input is None:
            # Import only for the default dataset preview; direct-file execution works too.
            import sys
            if not __package__:
                sys.path.insert(0, str(ROOT))
            from rice_disease.data import load_manifest
            rows, split = load_manifest()
            source_root = Path(split["dataset"]).resolve()
            samples = {}
            for row in rows:
                if row["split"] == "train":
                    samples.setdefault(row["class_name"], source_root / row["path"])
            paths = list(samples.values())
            print("Preview mode: one training image per class. Use --input for a file or folder.", flush=True)
        else:
            source = args.input.resolve()
            if not source.exists():
                raise ValueError(f"Input does not exist: {source}")
            source_root = source if source.is_dir() else source.parent
            if source.is_dir():
                paths = sorted(p for p in source.rglob("*")
                               if p.is_file() and p.suffix.lower() in EXTENSIONS)
            else:
                paths = [source]
        output = args.output.resolve()
        if (args.input is None or args.input.is_dir()) and (
            output.is_relative_to(source_root) or source_root.is_relative_to(output)
        ):
            raise ValueError("Input and output folders must not contain each other, to preserve source images.")
        if output / "summary.json" in paths:
            raise ValueError("The summary path would overwrite the input image. Choose another output directory.")
        if not paths:
            raise ValueError("No supported images found. Prepare the dataset or supply --input.")
        output.mkdir(parents=True, exist_ok=True)
        size = config["image_size"]
        print(f"Config: {args.config.resolve()}\nImage size: {size}x{size}\n"
              f"Normalization: mean={config['mean']}, std={config['std']}\n"
              f"Processing {len(paths)} image(s)...", flush=True)
        records, errors = [], []
        torch.set_num_threads(config.get("cpu_threads", 4))
        for index, path in enumerate(paths, 1):
            try:
                with Image.open(path) as original:
                    original_size = list(original.size)
                    resized = resized_image(original, config)
                tensor = image_tensor(resized, config)
                relative = path.relative_to(source_root)
                # Keep the original extension in the name to avoid a.jpg/a.png collisions.
                destination = output / "images" / relative.parent / (relative.name + ".png")
                destination.parent.mkdir(parents=True, exist_ok=True)
                resized.save(destination)
                records.append({"source": str(path), "output": str(destination),
                                "original_size": original_size, "output_size": list(resized.size),
                                "tensor_shape": list(tensor.shape), "tensor_dtype": str(tensor.dtype),
                                "tensor_min": tensor.min().item(), "tensor_max": tensor.max().item()})
                if len(paths) <= 10 or index % 100 == 0 or index == len(paths):
                    print(f"[{index}/{len(paths)}] {relative}: {original_size[0]}x{original_size[1]} "
                          f"-> {size}x{size} RGB; tensor {list(tensor.shape)}", flush=True)
            except (OSError, ValueError) as exc:
                errors.append({"source": str(path), "error": str(exc)})
                print(f"ERROR: {path}: {exc}", flush=True)
        report = output / "summary.json"
        report.write_text(json.dumps({"config_path": str(args.config.resolve()),
            "image_size": size, "mean": config["mean"], "std": config["std"],
            "augmentation": False, "processed": len(records), "failed": len(errors),
            "images": records, "errors": errors}, indent=2) + "\n", encoding="utf-8")
        print(f"Completed: {len(records)} processed, {len(errors)} failed.\n"
              f"Resized RGB images: {output / 'images'}\nSummary: {report}\n"
              "Normalization is applied to model tensors; saved PNGs remain viewable RGB images.", flush=True)
        return 1 if errors else 0
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Preprocessing error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
