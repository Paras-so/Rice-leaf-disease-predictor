"""Load a saved classifier and return predictions without claiming calibrated confidence."""

import argparse
import json
from pathlib import Path

if __name__ == "__main__" and not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "rice_disease"

from PIL import Image
import torch

from .data import ROOT
from .models import load_checkpoint
from .preprocessing import image_tensor
from .binary import health_status
from .quantity import disease_product_reference
from .unet import load_unet, predict_mask


def predict(model, checkpoint, image):
    tensor = image_tensor(image, checkpoint["config"]).unsqueeze(0)
    device = next(model.parameters()).device
    with torch.inference_mode():
        probabilities = model(tensor.to(device)).softmax(dim=1)[0].cpu().tolist()
    scores = dict(zip(checkpoint["classes"], probabilities))
    disease = max(scores, key=scores.get)
    return {"status": health_status(disease), "has_disease": disease != "healthy",
            "disease": disease, "softmax_score": scores[disease], "scores": scores,
            "architecture": checkpoint["architecture"],
            "confidence_note": "Uncalibrated model score; not a probability that the diagnosis is correct. No unknown-class detector is trained.",
            "affected_area_percent": None, "segmentation_status": "Segmentation has not been run; classification scores do not measure affected area."}


def management(disease):
    knowledge = json.loads((ROOT / "knowledge_base/diseases.json").read_text())
    return knowledge["diseases"][disease]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path,
                        help="Defaults to the checkpoint in artifacts/selected_model.json")
    parser.add_argument("--segmentation-checkpoint", type=Path,
                        default=ROOT / 'artifacts/unet/model.pt')
    parser.add_argument("--water-ml", type=float, default=100,
                        help="Water volume for label-reference arithmetic, not a per-plant spray volume (default: 100)")
    args = parser.parse_args()
    if args.checkpoint is None:
        selection = json.loads((ROOT / "artifacts/selected_model.json").read_text())
        args.checkpoint = ROOT / "artifacts" / selection["checkpoint"]
    torch.set_num_threads(4)
    model, checkpoint = load_checkpoint(args.checkpoint)
    with Image.open(args.image) as image:
        result = predict(model, checkpoint, image)
        if args.segmentation_checkpoint.is_file():
            segmentation_model, segmentation_checkpoint = load_unet(args.segmentation_checkpoint)
            _, area, _ = predict_mask(segmentation_model, segmentation_checkpoint, image)
            result['segmentation'] = area
            result['affected_area_percent'] = area['affected_area_percent']
            result['segmentation_status'] = area['status']
        else:
            result['segmentation_status'] = 'unavailable: no trained U-Net checkpoint; reviewed masks are required'
    result["management"] = management(result["disease"])
    try:
        result['pesticide_reference'] = disease_product_reference(result['disease'], water_ml=args.water_ml)
    except ValueError as exc:
        parser.exit(1, f'Quantity error: {exc}\n')
    print(json.dumps(result, indent=2))
