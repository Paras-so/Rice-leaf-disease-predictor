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
            "affected_area_percent": None, "segmentation_status": "No trained segmentation model or labelled masks available."}


def management(disease):
    knowledge = json.loads((ROOT / "knowledge_base/diseases.json").read_text())
    return knowledge["diseases"][disease]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path,
                        help="Defaults to the checkpoint in artifacts/selected_model.json")
    args = parser.parse_args()
    if args.checkpoint is None:
        selection = json.loads((ROOT / "artifacts/selected_model.json").read_text())
        args.checkpoint = ROOT / "artifacts" / selection["checkpoint"]
    torch.set_num_threads(4)
    model, checkpoint = load_checkpoint(args.checkpoint)
    with Image.open(args.image) as image:
        result = predict(model, checkpoint, image)
    result["management"] = management(result["disease"])
    print(json.dumps(result, indent=2))
