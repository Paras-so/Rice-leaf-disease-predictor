"""Evaluate a frozen, hashed selection once on the reserved test holdout."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from .benchmark import metrics
from .data import ROOT, CLASSES, load_manifest, write_json
from .models import load_checkpoint
from .preprocessing import LeafDataset


def evaluate(artifacts):
    artifacts = Path(artifacts).resolve()
    selection = json.loads((artifacts / "selected_model.json").read_text())
    rows, split = load_manifest(artifacts)
    path = artifacts / selection["checkpoint"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != selection["checkpoint_sha256"] or split["manifest_sha256"] != selection["split_sha256"]:
        raise ValueError("Selected checkpoint or dataset split changed after model selection.")
    output = artifacts / "final_evaluation"
    if (output / "metrics.json").exists():
        existing = json.loads((output / "metrics.json").read_text())
        if existing["checkpoint_sha256"] != digest:
            raise ValueError("Holdout results already exist for a different model.")
        print("Returning existing final evaluation; no test inference repeated.")
        return existing
    model, checkpoint = load_checkpoint(path)
    if checkpoint["split_sha256"] != split["manifest_sha256"]:
        raise ValueError("Checkpoint was trained against another split.")
    torch.set_num_threads(checkpoint["config"]["cpu_threads"])
    test_rows = [r for r in rows if r["split"] == "test"]
    dataset = LeafDataset(test_rows, split["dataset"], checkpoint["config"], artifacts / "cache/images")
    loader = DataLoader(dataset, batch_size=checkpoint["config"]["feature_batch_size"], shuffle=False)
    logits, labels = [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for images, target in loader:
            logits.append(model(images).cpu())
            labels.append(target)
    logits, labels = torch.cat(logits), torch.cat(labels)
    prediction = logits.argmax(1)
    result = {"architecture": checkpoint["architecture"], "checkpoint_sha256": digest,
              "split_sha256": split["manifest_sha256"], "test_images": len(test_rows),
              "test": metrics(logits, labels), "evaluation_seconds": time.perf_counter() - started,
              "per_class": classification_report(labels.numpy(), prediction.numpy(), labels=list(range(6)),
                                                  target_names=CLASSES, output_dict=True, zero_division=0),
              "confusion_matrix": confusion_matrix(labels.numpy(), prediction.numpy(), labels=list(range(6))).tolist(),
              "limitations": "One held-out image split, one seed, no field-level grouping metadata, no confidence calibration, no unknown-class detection."}
    examples = [{"path": row["path"], "actual": row["class_name"], "predicted": CLASSES[prediction[i].item()],
                 "softmax_score": logits[i].softmax(0).max().item()} for i, row in enumerate(test_rows)]
    write_json(output / "predictions.json", examples)
    write_json(output / "misclassified.json", [r for r in examples if r["actual"] != r["predicted"]])
    write_json(output / "metrics.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    evaluate(parser.parse_args().artifacts)
