"""After the initial benchmark, compare two learning rates on its top two architectures."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

if __name__ == "__main__" and not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "rice_disease"

import torch

from .benchmark import benchmark_one, summarize
from .data import ROOT, load_manifest, write_json


def tune(artifacts):
    artifacts = Path(artifacts).resolve()
    selection_path = artifacts / "selected_model.json"
    if selection_path.exists():
        selected = json.loads(selection_path.read_text())
        path = artifacts / selected["checkpoint"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != selected["checkpoint_sha256"]:
            raise ValueError("Frozen checkpoint has changed.")
        _, split_info = load_manifest(artifacts)
        if selected["split_sha256"] != split_info["manifest_sha256"]:
            raise ValueError("Frozen selection belongs to another split.")
        print("Returning the existing frozen selection; no further tuning performed.")
        print(json.dumps(selected, indent=2))
        return selected
    os.environ["TORCH_HOME"] = str(artifacts / "cache/torch")
    baseline = artifacts / "runs/frozen_baseline"
    config = json.loads((baseline / "config.json").read_text())
    comparisons = json.loads((baseline / "comparison.json").read_text())
    if {r["architecture"] for r in comparisons} != set(config["models"]):
        raise ValueError("Complete all four baseline models before tuning.")
    rows, split_info = load_manifest(artifacts)
    torch.set_num_threads(config["cpu_threads"])
    candidates = [{"checkpoint": str((baseline / f"{r['architecture']}.pt").relative_to(artifacts)),
                   "run": "frozen_baseline", "metrics": r} for r in comparisons]
    top_models = [r["architecture"] for r in sorted(comparisons, key=lambda r: -r["validation"]["macro_f1"])[:2]]
    search = {"architectures": top_models, "learning_rates": [0.0003, 0.003],
              "fixed": "Every other setting remains the baseline value; feature caches are reused.",
              "selection": "Validation macro F1, then smaller parameter count for exact ties. Test images are not loaded."}
    write_json(artifacts / "tuning_plan.json", search)
    for rate in search["learning_rates"]:
        variant = copy.deepcopy(config)
        variant["learning_rate"] = rate
        run = artifacts / "runs" / f"head_lr_{rate:g}"
        run.mkdir(parents=True, exist_ok=True)
        write_json(run / "config.json", variant)
        for name in top_models:
            metric_path = run / f"{name}_metrics.json"
            if metric_path.exists() and (run / f"{name}.pt").exists():
                result = json.loads(metric_path.read_text())
            else:
                result = benchmark_one(name, variant, rows, split_info, artifacts, run, feature_config=config)
            candidates.append({"checkpoint": str((run / f"{name}.pt").relative_to(artifacts)),
                               "run": run.name, "metrics": result})
        summarize(run, variant)
    candidates.sort(key=lambda c: (-c["metrics"]["validation"]["macro_f1"], c["metrics"]["total_parameters"]))
    selected = candidates[0]
    checkpoint_path = artifacts / selected["checkpoint"]
    selected = {**selected, "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "split_sha256": split_info["manifest_sha256"],
                "selection_basis": "Highest validation macro F1 across four baseline architectures and four targeted head-learning-rate trials; parameter count breaks exact ties. Test set not used.",
                "development_status": "Frozen-feature classifier v1; full-backbone fine-tuning not performed.",
                "test_policy": "This freezes v1 before one final holdout evaluation. Do not tune against its test scores."}
    if selection_path.exists() and json.loads(selection_path.read_text()) != selected:
        raise ValueError("A different final model is already frozen; use a separate experiment directory.")
    write_json(artifacts / "tuning_candidates.json", candidates)
    write_json(selection_path, selected)
    print(json.dumps(selected, indent=2))
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    tune(parser.parse_args().artifacts)
