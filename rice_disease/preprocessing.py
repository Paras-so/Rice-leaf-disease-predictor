"""Full-leaf resize and deterministic training-only augmentation."""

import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


def image_tensor(image, config, augmentation_seed=None):
    image = ImageOps.exif_transpose(image).convert("RGB")
    size = config["image_size"]
    image = image.resize((size, size), Image.Resampling.BILINEAR)
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
                image = ImageOps.exif_transpose(original).convert("RGB")
                image = image.resize((self.config["image_size"], self.config["image_size"]),
                                     Image.Resampling.BILINEAR)
                image.save(cached)
        seed = self.config["seed"] + row_index * 100 + view if self.training and view else None
        with Image.open(cached) as image:
            tensor = image_tensor(image, self.config, seed)
        return tensor, torch.tensor(row["label"], dtype=torch.long)
