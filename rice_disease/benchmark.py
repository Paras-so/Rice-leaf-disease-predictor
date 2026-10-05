"""Run comparable frozen-backbone benchmarks without reading test images."""

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import ROOT, CLASSES, load_manifest, write_json
from .models import create_model, remove_head, attach_head
from .preprocessing import LeafDataset


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def metrics(logits, labels):
    prediction = logits.argmax(1).numpy()
    truth = labels.numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, prediction, labels=list(range(len(CLASSES))), average="macro", zero_division=0)
    return {"loss": nn.functional.cross_entropy(logits, labels).item(),
            "accuracy": accuracy_score(truth, prediction), "macro_precision": precision,
            "macro_recall": recall, "macro_f1": f1}


def extract(model, loader, device, name):
    embeddings, labels = [], []
    start = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for index, (images, target) in enumerate(loader):
            embeddings.append(model(images.to(device)).cpu())
            labels.append(target)
            if (index + 1) % 20 == 0 or index + 1 == len(loader):
                print(f"{name}: batch {index + 1}/{len(loader)} ({time.perf_counter()-start:.1f}s)", flush=True)
    return torch.cat(embeddings), torch.cat(labels)


def benchmark_one(name, config, rows, split_info, artifacts, run, feature_config=None):
    seed_everything(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, head = create_model(name)
    counts = {"total_parameters": sum(p.numel() for p in model.parameters()),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    model = model.to(device)
    head = remove_head(model, name).cpu()
    feature_config = feature_config or config
    for key in ("seed", "weights", "image_size", "mean", "std", "training_views", "augmentation"):
        if feature_config[key] != config[key]:
            raise ValueError(f"Cannot reuse features with different {key}")
    cache_key = hashlib.sha256(json.dumps({"config": feature_config, "split": split_info["manifest_sha256"],
                                          "model": name, "torchvision": __import__("torchvision").__version__}, sort_keys=True).encode()).hexdigest()[:20]
    feature_path = artifacts / "cache" / f"{name}_{cache_key}.pt"
    started = time.perf_counter()
    if feature_path.exists():
        features = torch.load(feature_path, weights_only=True)
        extraction_seconds = features["extraction_seconds"]
        print(f"{name}: reusing verified-config feature cache", flush=True)
    else:
        features = {}
        for split in ("train", "validation"):
            subset = [row for row in rows if row["split"] == split]
            dataset = LeafDataset(subset, split_info["dataset"], config,
                                  artifacts / "cache" / "images", training=split == "train")
            loader = DataLoader(dataset, batch_size=config["feature_batch_size"], shuffle=False,
                                num_workers=config["workers"])
            features[split] = extract(model, loader, device, f"{name}/{split}")
        extraction_seconds = time.perf_counter() - started
        features["extraction_seconds"] = extraction_seconds
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(features, feature_path)
    train_x, train_y = features["train"]
    val_x, val_y = features["validation"]
    generator = torch.Generator().manual_seed(config["seed"])
    batches = DataLoader(TensorDataset(train_x, train_y), batch_size=config["batch_size"],
                         shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    loss_function = nn.CrossEntropyLoss()
    history, best_score, stale = [], -1.0, 0
    best_state, best_epoch = None, None
    train_started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        head.train()
        for x, y in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(head(x), y)
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.inference_mode():
            # Original training views only, so reported train/validation scores are comparable.
            train_metrics = metrics(head(train_x[::config["training_views"]]), train_y[::config["training_views"]])
            val_metrics = metrics(head(val_x), val_y)
        entry = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(entry)
        print(f"{name} epoch {epoch:02}: train acc={train_metrics['accuracy']:.4f}, val acc={val_metrics['accuracy']:.4f}, val F1={val_metrics['macro_f1']:.4f}", flush=True)
        if val_metrics["macro_f1"] > best_score + 1e-6:
            best_score, best_epoch, best_state = val_metrics["macro_f1"], epoch, copy.deepcopy(head.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    training_seconds = time.perf_counter() - train_started
    head.load_state_dict(best_state)
    head.eval()
    with torch.inference_mode():
        val_logits = head(val_x)
        train_metrics = metrics(head(train_x[::config["training_views"]]), train_y[::config["training_views"]])
        val_metrics = metrics(val_logits, val_y)
    attach_head(model, name, head.to(device))
    model.eval()
    sample = torch.zeros(1, 3, config["image_size"], config["image_size"], device=device)
    with torch.inference_mode():
        for _ in range(3):
            model(sample)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            model(sample)
        if device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000 / 20
    checkpoint_path = run / f"{name}.pt"
    torch.save({"architecture": name, "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "classes": CLASSES, "config": config, "split_sha256": split_info["manifest_sha256"],
                "best_epoch": best_epoch, "validation": val_metrics}, checkpoint_path)
    gap = train_metrics["accuracy"] - val_metrics["accuracy"]
    result = {"architecture": name, **counts, "best_epoch": best_epoch, "epochs_run": len(history),
              "train": train_metrics, "validation": val_metrics,
              "feature_extraction_seconds": extraction_seconds, "head_training_seconds": training_seconds,
              "total_training_seconds": extraction_seconds + training_seconds,
              "checkpoint_size_mb": checkpoint_path.stat().st_size / 1e6,
              "inference_ms_batch1": latency_ms, "inference_scope": "Model forward only, synthetic input, 3 warmups and 20 repeats; excludes preprocessing.",
              "device": device, "train_validation_accuracy_gap": gap,
              "overfitting_observation": "Possible overfitting: training accuracy exceeds validation by more than 5 percentage points." if gap > .05 else "No large accuracy gap; inspect loss curves and per-class validation performance.",
              "validation_per_class": classification_report(val_y.numpy(), val_logits.argmax(1).numpy(),
                    labels=list(range(6)), target_names=CLASSES, output_dict=True, zero_division=0),
              "validation_confusion_matrix": confusion_matrix(val_y.numpy(), val_logits.argmax(1).numpy(), labels=list(range(6))).tolist()}
    write_json(run / f"{name}_history.json", history)
    write_json(run / f"{name}_metrics.json", result)
    return result


def run_benchmark(config_path, artifacts, run_name, selected_models=None):
    artifacts = Path(artifacts).resolve()
    os.environ["TORCH_HOME"] = str(artifacts / "cache" / "torch")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    torch.set_num_threads(config["cpu_threads"])
    rows, split_info = load_manifest(artifacts)
    run = artifacts / "runs" / run_name
    run.mkdir(parents=True, exist_ok=True)
    if (run / "config.json").exists():
        if json.loads((run / "config.json").read_text()) != config:
            raise ValueError("Run already has another config. Choose a new --run name.")
    if (run / "environment.json").exists():
        recorded = json.loads((run / "environment.json").read_text())
        if recorded["split_sha256"] != split_info["manifest_sha256"]:
            raise ValueError("Run belongs to another dataset split. Choose a new --run name.")
    write_json(run / "config.json", config)
    write_json(run / "environment.json", {"torch": str(torch.__version__),
        "torchvision": str(__import__("torchvision").__version__), "cuda": torch.cuda.is_available(),
        "cpu_threads": torch.get_num_threads(), "split_sha256": split_info["manifest_sha256"]})
    for name in selected_models or config["models"]:
        if name not in config["models"]:
            raise ValueError(f"Architecture not configured: {name}")
        if (run / f"{name}_metrics.json").exists() and (run / f"{name}.pt").exists():
            print(f"{name}: completed; preserving recorded results", flush=True)
            continue
        benchmark_one(name, config, rows, split_info, artifacts, run)
        summarize(run, config)
    return summarize(run, config)


def summarize(run, config):
    results = [json.loads(path.read_text()) for name in config["models"]
               if (path := run / f"{name}_metrics.json").exists()]
    results.sort(key=lambda x: (-x["validation"]["macro_f1"], x["total_parameters"]))
    write_json(run / "comparison.json", results)
    if results:
        with (run / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ["architecture", "total_parameters", "trainable_parameters", "train_accuracy", "validation_accuracy", "validation_macro_f1", "validation_macro_precision", "validation_macro_recall", "total_training_seconds", "checkpoint_size_mb", "inference_ms_batch1"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for item in results:
                row = {key: item[key] for key in fields if key in item}
                row["train_accuracy"] = item["train"]["accuracy"]
                row.update({f"validation_{key}": item["validation"][key] for key in ("accuracy", "macro_f1", "macro_precision", "macro_recall")})
                writer.writerow(row)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmark.json")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--run", default="frozen_baseline")
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()
    run_benchmark(args.config, args.artifacts, args.run, args.models)
