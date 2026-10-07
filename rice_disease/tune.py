"""Compare training recipes on validation data and activate the single best model."""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

if __name__ == "__main__" and not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "rice_disease"

import torch

from .benchmark import benchmark_one, summarize
from .data import ROOT, load_manifest, write_json


def trial_configs(base):
    """Predeclared comparisons; the original baseline is retained as a candidate."""
    trials = []
    losses = [("ce", "CrossEntropyLoss", {}),
              ("smooth", "LabelSmoothedCrossEntropy", {"label_smoothing": 0.1}),
              ("focal", "FocalLoss", {"focal_gamma": 2.0})]
    for optimizer in ("AdamW", "Adam", "SGD"):
        for suffix, loss, extra in losses:
            if optimizer == "AdamW" and suffix == "ce":
                continue
            settings = {"optimizer": optimizer, "loss": loss, **extra}
            if optimizer == "SGD":
                settings.update(learning_rate=0.01, momentum=0.9)
            trials.append((f"{optimizer.lower()}_{suffix}", {**base, **settings}))
    for name, extra in (
        ("adamw_ce_lr_low", {"learning_rate": 0.0003}),
        ("adamw_ce_lr_high", {"learning_rate": 0.003}),
        ("adamw_ce_batch64", {"batch_size": 64}),
        ("adamw_ce_decay", {"weight_decay": 0.001}),
    ):
        trials.append((name, {**base, **extra}))
    return trials


def candidate_key(candidate):
    result = candidate["metrics"]
    # Stable ties keep the earlier candidate (baseline comes first).
    return (-result["validation"]["macro_f1"], result["total_parameters"])


def checked_candidate(artifacts, run, name, config, split_sha):
    path = run / f"{name}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    result = json.loads((run / f"{name}_metrics.json").read_text())
    if (checkpoint["config"] != config or checkpoint["split_sha256"] != split_sha
            or checkpoint["architecture"] != name or result["architecture"] != name
            or checkpoint["validation"] != result["validation"]):
        raise ValueError(f"Checkpoint/config/metrics mismatch: {path}")
    return {"checkpoint": path.relative_to(artifacts).as_posix(),
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "run": run.name, "config": config, "metrics": result}


def save_comparison(folder, candidates):
    ordered = sorted(candidates, key=candidate_key)
    write_json(folder / "candidates.json", ordered)
    records = []
    for rank, item in enumerate(ordered, 1):
        cfg, result = item["config"], item["metrics"]
        records.append({"rank": rank, "run": item["run"], "architecture": result["architecture"],
                        **{key: cfg.get(key, "") for key in (
                            "optimizer", "loss", "learning_rate", "weight_decay", "batch_size",
                            "momentum", "label_smoothing", "focal_gamma")},
                        "best_epoch": result["best_epoch"],
                        "validation_accuracy": result["validation"]["accuracy"],
                        "validation_macro_f1": result["validation"]["macro_f1"],
                        "train_accuracy": result["train"]["accuracy"]})
    with (folder / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return ordered


def activate(artifacts, selection):
    """Archive the previous pointer and atomically publish one verified winner."""
    checkpoint = artifacts / selection["checkpoint"]
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != selection["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint has changed.")
    _, split = load_manifest(artifacts)
    if split["manifest_sha256"] != selection["split_sha256"]:
        raise ValueError("Selection belongs to another dataset split.")
    path = artifacts / "selected_model.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous == selection:
            return
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        archive = artifacts / "selection_history" / f"selection_{digest}.json"
        if not archive.exists():
            write_json(archive, previous)
    pending = path.with_suffix(".pending.json")
    write_json(pending, selection)
    pending.replace(path)


def tune(artifacts, verbose=1, experiment="hyperparameter_search_v2"):
    artifacts = Path(artifacts).resolve()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", experiment):
        raise ValueError("Experiment name must use letters, numbers, underscores or hyphens.")
    os.environ["TORCH_HOME"] = str(artifacts / "cache/torch")
    rows, split = load_manifest(artifacts)
    baseline = artifacts / "runs/frozen_baseline"
    config = json.loads((baseline / "config.json").read_text())
    torch.set_num_threads(config["cpu_threads"])
    folder = artifacts / "experiments" / experiment
    folder.mkdir(parents=True, exist_ok=True)
    trials = trial_configs(config)
    plan = {"experiment": experiment, "baseline_config": config,
            "split_sha256": split["manifest_sha256"],
            "torch": str(torch.__version__), "torchvision": str(__import__("torchvision").__version__),
            "trials": [{"name": name, "config": cfg} for name, cfg in trials],
            "candidate_count": (len(trials) + 1) * len(config["models"]),
            "selection": "Validation macro F1; then smaller parameter count; stable ties retain baseline.",
            "test_policy": "Test data is not read during search. The holdout was previously evaluated for v1."}
    plan_path = folder / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Experiment plan changed; choose another --experiment name.")
    write_json(plan_path, plan)
    selected_path = folder / "selected_model.json"
    if selected_path.exists():
        selected = json.loads(selected_path.read_text())
        activate(artifacts, selected)
        print(f"Using completed experiment: {selected['metrics']['architecture']} / {selected['run']}")
        return selected
    candidates = [checked_candidate(artifacts, baseline, name, config, split["manifest_sha256"])
                  for name in config["models"]]
    save_comparison(folder, candidates)
    for trial_index, (trial_name, variant) in enumerate(trials, 1):
        run = artifacts / "runs" / f"{experiment}_{trial_name}"
        run.mkdir(parents=True, exist_ok=True)
        if (run / "config.json").exists() and json.loads((run / "config.json").read_text()) != variant:
            raise ValueError(f"Run config changed: {run}")
        write_json(run / "config.json", variant)
        for name in config["models"]:
            print(f"Trial {trial_index}/{len(trials)}: {trial_name} / {name}", flush=True)
            if not ((run / f"{name}_metrics.json").exists() and (run / f"{name}.pt").exists()):
                benchmark_one(name, variant, rows, split, artifacts, run,
                              feature_config=config, verbose=verbose)
            candidates.append(checked_candidate(artifacts, run, name, variant, split["manifest_sha256"]))
            save_comparison(folder, candidates)
        summarize(run, variant)
    ordered = save_comparison(folder, candidates)
    winner = ordered[0]
    # Only one deployed model; research checkpoints stay available for audit/resume.
    deployment = artifacts / "final_model" / experiment
    deployment.mkdir(parents=True, exist_ok=True)
    final_path = deployment / "model.pt"
    shutil.copy2(artifacts / winner["checkpoint"], final_path)
    write_json(deployment / "config.json", winner["config"])
    selected = {**winner, "source_checkpoint": winner["checkpoint"],
                "checkpoint": final_path.relative_to(artifacts).as_posix(),
                "experiment": experiment, "candidate_count": len(ordered),
                "comparison": (folder / "comparison.csv").relative_to(artifacts).as_posix(),
                "candidates": (folder / "candidates.json").relative_to(artifacts).as_posix(),
                "evaluation_dir": f"evaluations/{experiment}",
                "split_sha256": split["manifest_sha256"],
                "selection_basis": plan["selection"],
                "development_status": "Frozen-feature classifier; only the final linear layer is trained.",
                "test_policy": plan["test_policy"]}
    write_json(selected_path, selected)
    activate(artifacts, selected)
    print(f"Selected {winner['metrics']['architecture']} / {winner['run']}: "
          f"validation accuracy={winner['metrics']['validation']['accuracy']:.2%}, "
          f"macro F1={winner['metrics']['validation']['macro_f1']:.4f}", flush=True)
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--experiment", default="hyperparameter_search_v2")
    parser.add_argument("--verbose", type=int, choices=(0, 1), default=1,
                        help="Show live batch and epoch progress (default: 1).")
    args = parser.parse_args()
    tune(args.artifacts, verbose=args.verbose, experiment=args.experiment)
