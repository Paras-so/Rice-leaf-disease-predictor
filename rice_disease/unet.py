"""Compact three-class U-Net and aspect-preserving segmentation inference."""

from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F

CLASSES = ["background", "unaffected_leaf", "affected_leaf"]


class DoubleConv(nn.Sequential):
    def __init__(self, inputs, outputs):
        super().__init__(
            nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(4, outputs), nn.ReLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(4, outputs), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Four encoder/decoder levels with skip connections and raw class logits."""
    def __init__(self, base_channels=16):
        super().__init__()
        if base_channels < 4 or base_channels % 4:
            raise ValueError("base_channels must be a positive multiple of four")
        widths = [base_channels * 2**i for i in range(5)]
        self.encoder = nn.ModuleList([DoubleConv(3, widths[0])] +
                                    [DoubleConv(a, b) for a, b in zip(widths, widths[1:])])
        self.decoder = nn.ModuleList([DoubleConv(widths[i]+widths[i-1], widths[i-1])
                                     for i in range(4, 0, -1)])
        self.head = nn.Conv2d(widths[0], 3, 1)

    def forward(self, inputs):
        skips = []
        x = inputs
        for i, block in enumerate(self.encoder):
            x = block(F.max_pool2d(x, 2) if i else x)
            skips.append(x)
        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat((x, skip), dim=1))
        return self.head(x)


def letterbox(image, size, mask=None):
    """Pad to a square; ignore padding in the loss rather than invent background."""
    if size < 32 or size % 16:
        raise ValueError("image_size must be >=32 and divisible by 16")
    image = ImageOps.exif_transpose(image).convert("RGB")
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != (image.height, image.width) or not np.isin(mask, [0, 1, 2]).all():
            raise ValueError("Mask must align with the EXIF-corrected source and contain only labels 0, 1, 2")
    scale = min(size/image.width, size/image.height)
    width, height = max(1, round(image.width*scale)), max(1, round(image.height*scale))
    left, top = (size-width)//2, (size-height)//2
    canvas = Image.new("RGB", (size, size))
    canvas.paste(image.resize((width, height), Image.Resampling.BILINEAR), (left, top))
    tensor = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float()/255.0
    target = None
    if mask is not None:
        target = np.full((size, size), 255, dtype=np.uint8)
        target[top:top+height, left:left+width] = np.asarray(
            Image.fromarray(mask.astype(np.uint8)).resize((width, height), Image.Resampling.NEAREST))
        target = torch.from_numpy(target.astype(np.int64))
    return tensor, target, (left, top, width, height), image


def segmentation_loss(logits, target):
    valid = target != 255
    if not valid.any():
        raise ValueError("Batch contains no labelled pixels")
    weights = logits.new_tensor([1.0, 1.0, 3.0])
    ce = F.cross_entropy(logits, target, weight=weights, ignore_index=255)
    probabilities = logits.softmax(dim=1)
    dice = []
    for predicted, actual in ((probabilities[:, 1:].sum(dim=1), (target == 1) | (target == 2)),
                              (probabilities[:, 2], target == 2)):
        predicted, actual = predicted*valid, actual*valid
        dice.append(1-(2*(predicted*actual).sum()+1)/(predicted.sum()+actual.sum()+1))
    return ce + torch.stack(dice).mean()


def measure_prediction(mask):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.isin(mask, [0, 1, 2]).all():
        raise ValueError("Expected a 2D semantic mask with labels 0, 1, 2")
    leaf = int(np.count_nonzero(mask > 0))
    affected = int(np.count_nonzero(mask == 2))
    return {"leaf_pixels": leaf, "affected_pixels": affected,
            "affected_area_percent": affected/leaf*100 if leaf else None,
            "status": "estimated" if leaf else "no_leaf_detected",
            "formula": "100 * affected_pixels / (unaffected_leaf_pixels + affected_pixels)",
            "source": "U-Net predicted mask", "pesticide_quantity": None}


def load_unet(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "unet" or checkpoint.get("classes") != CLASSES:
        raise ValueError("Not a supported three-class U-Net checkpoint")
    if checkpoint.get("training_status") != "trained":
        raise ValueError("Checkpoint does not contain a trained U-Net")
    if checkpoint.get("label_quality") not in ("reviewed", "ai_visual_reviewed"):
        raise ValueError("U-Net must be trained on reviewed annotations")
    if checkpoint.get('label_quality') == 'ai_visual_reviewed':
        count = checkpoint.get('review_methods', {}).get('ai_visual')
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError('AI-reviewed checkpoint is missing review provenance')
    model = UNet(checkpoint["config"]["base_channels"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def predict_mask(model, checkpoint, image):
    tensor, _, (left, top, width, height), rgb = letterbox(image, checkpoint["config"]["image_size"])
    with torch.inference_mode():
        logits = model(tensor.unsqueeze(0).to(next(model.parameters()).device))
        logits = logits[:, :, top:top+height, left:left+width]
        # Restore logits before argmax so the percentage uses the original geometry.
        logits = F.interpolate(logits, size=(rgb.height, rgb.width), mode="bilinear", align_corners=False)
        mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    area = measure_prediction(mask)
    area.update(label_quality=checkpoint["label_quality"],
                quality_note=checkpoint.get('quality_note', 'Reviewed annotations; not field-validated.'),
                field_validated=False,
                note="Estimated visible leaf area; not field disease prevalence or a spray-rate multiplier.")
    return mask, area, rgb


def mask_overlay(image, mask):
    pixels = np.asarray(image).copy()
    for label, colour in ((1, (30, 200, 80)), (2, (255, 60, 50))):
        selected = mask == label
        pixels[selected] = (pixels[selected]*0.55 + np.array(colour)*0.45).astype(np.uint8)
    return Image.fromarray(pixels)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    model = UNet()
    print(f"U-Net: {sum(p.numel() for p in model.parameters()):,} parameters; classes={CLASSES}")
    print("Train: python -m rice_disease.train_segmentation --help")
