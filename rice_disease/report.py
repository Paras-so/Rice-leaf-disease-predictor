"""Generate figures and a readable report from measured benchmark results."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import ROOT, CLASSES
from .binary import binary_metrics
from PIL import Image, ImageOps


def generate(run):
    run = Path(run)
    results = json.loads((run / "comparison.json").read_text())
    figures = run / "figures"
    figures.mkdir(exist_ok=True)
    for result in results:
        name = result["architecture"]
        history = json.loads((run / f"{name}_history.json").read_text())
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for axis, metric in zip(axes, ("loss", "accuracy")):
            for split in ("train", "validation"):
                axis.plot([h["epoch"] for h in history], [h[split][metric] for h in history], label=split)
            axis.set(xlabel="Epoch", ylabel=metric.title())
            axis.axvline(result["best_epoch"], color="gray", linestyle="--", alpha=.6)
            axis.legend()
        fig.suptitle(name + " — frozen ImageNet features")
        fig.tight_layout()
        fig.savefig(figures / f"{name}_learning_curves.png", dpi=160)
        plt.close(fig)
        matrix = np.array(result["validation_confusion_matrix"])
        fig, axis = plt.subplots(figsize=(8, 7))
        axis.imshow(matrix, cmap="Blues")
        axis.set(xticks=range(6), yticks=range(6), xticklabels=CLASSES, yticklabels=CLASSES,
                 xlabel="Predicted class", ylabel="True class", title=name + " — validation")
        plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
        for i in range(6):
            for j in range(6):
                axis.text(j, i, str(matrix[i, j]), ha="center", va="center",
                          color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        fig.tight_layout()
        fig.savefig(figures / f"{name}_validation_confusion.png", dpi=160)
        plt.close(fig)
    lines = ["# Measured classification benchmark", "",
             "Frozen pretrained ImageNet backbones with a learned six-class final linear layer. "
             "This measures transfer-feature quality; it is not an end-to-end fine-tuning comparison.", "",
             "All models use the same split, 224×224 full-image resize, normalization, two training views, "
             "AdamW configuration, batch size and stopping criterion. The 389-image test holdout has not been scored by this benchmark.", "",
             "| Model | Parameters | Trainable | Train accuracy | Validation accuracy | Macro F1 | Training seconds | Forward ms |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['architecture']} | {r['total_parameters']:,} | {r['trainable_parameters']:,} | {r['train']['accuracy']:.3%} | {r['validation']['accuracy']:.3%} | {r['validation']['macro_f1']:.4f} | {r['total_training_seconds']:.1f} | {r['inference_ms_batch1']:.2f} |")
    lines += ["", "Times include feature extraction plus head training; pretrained downloads are excluded. "
              "Forward latency uses one synthetic input and excludes decoding/preprocessing. CPU load and caching affect timing.", "",
              "## Interpretation", ""]
    if results:
        best = results[0]
        lines.append(f"The highest validation macro F1 is **{best['architecture']}**, at **{best['validation']['macro_f1']:.4f}**. "
                     "This is a candidate for further development, not evidence of statistical superiority or field reliability.")
    lines += ["", "One split and one seed cannot establish a capacity-performance law. Architecture and ImageNet training recipes also differ. "
              "The classifier has no unknown-class or non-rice rejection model, and softmax scores are uncalibrated.", "",
              "## Per-model observations", ""]
    for r in results:
        lines.append(f"- {r['architecture']}: best epoch {r['best_epoch']}; train–validation accuracy gap "
                     f"{r['train_validation_accuracy_gap']:.1%}. {r['overfitting_observation']}")
    lines += ["", "## Sources and experimental choices", "",
              "- [TorchVision pretrained model API](https://docs.pytorch.org/vision/stable/models.html)",
              "- [PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)",
              "- Explicit IMAGENET1K_V1 weights are pinned. Using a common full-image resize instead of the weights' default center crop preserves leaf-edge symptoms; it is a documented benchmark choice.",
              "- Segmentation and severity cannot be inferred from these classification labels; masks are required."]
    binary_rows = []
    lines += ["", "## Healthy versus diseased: validation", "",
              "Fixed mapping of the six-class argmax: healthy stays healthy; every disease class becomes diseased. No separate binary model or threshold tuning.", "",
              "| Model | Binary accuracy | Disease precision | Disease recall | Disease F1 |",
              "|---|---:|---:|---:|---:|"]
    for result in results:
        binary = binary_metrics(result["validation_confusion_matrix"], CLASSES)
        row = {"architecture": result["architecture"], **{k: binary[k] for k in
               ("accuracy", "disease_precision", "disease_recall", "disease_f1", "balanced_accuracy")}}
        binary_rows.append(row)
        lines.append(f"| {row['architecture']} | {row['accuracy']:.2%} | {row['disease_precision']:.4f} | {row['disease_recall']:.4f} | {row['disease_f1']:.4f} |")
    if binary_rows:
        with (run / "binary_comparison.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(binary_rows[0]))
            writer.writeheader()
            writer.writerows(binary_rows)
    (run / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {run / 'REPORT.md'}")


def final_report(artifacts):
    artifacts = Path(artifacts)
    evaluation_path = artifacts / "final_evaluation/metrics.json"
    if not evaluation_path.exists():
        return
    evaluation = json.loads(evaluation_path.read_text())
    selection = json.loads((artifacts / "selected_model.json").read_text())
    candidates = json.loads((artifacts / "tuning_candidates.json").read_text())
    errors = json.loads((artifacts / "final_evaluation/misclassified.json").read_text())
    split = json.loads((artifacts / "split_summary.json").read_text())
    matrix = np.array(evaluation["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(8, 7))
    axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(6), yticks=range(6), xticklabels=CLASSES, yticklabels=CLASSES,
             xlabel="Predicted", ylabel="Actual", title="Selected v1: final test confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    for i in range(6):
        for j in range(6):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center",
                      color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(artifacts / "final_evaluation/confusion_matrix.png", dpi=160)
    plt.close(fig)
    if errors:
        shown = errors[:12]
        fig, axes = plt.subplots((len(shown) + 3) // 4, 4, figsize=(16, 4 * ((len(shown) + 3) // 4)), squeeze=False)
        for axis in axes.flat:
            axis.axis("off")
        for axis, item in zip(axes.flat, shown):
            with Image.open(Path(split["dataset"]) / item["path"]) as image:
                axis.imshow(ImageOps.exif_transpose(image).resize((224, 224)))
            axis.set_title(f"True: {item['actual']}\nPredicted: {item['predicted']}\nScore: {item['softmax_score']:.2f}", fontsize=9)
        fig.tight_layout()
        fig.savefig(artifacts / "final_evaluation/misclassified_examples.png", dpi=140)
        plt.close(fig)
    confusion_pairs = sorted([(int(matrix[i, j]), CLASSES[i], CLASSES[j])
                              for i in range(6) for j in range(6) if i != j and matrix[i, j]], reverse=True)
    measured = selection["metrics"]
    lines = ["# Software/ML project: executed results", "",
        "## Dataset", "",
        "1,941 readable RGB images across six classes. No exact-file or identical-pixel duplicates. "
        "Split: 1,241 train, 311 validation, 389 test (seed 42). Source images are preserved.", "",
        "## Classification v1", "",
        f"Selected **{evaluation['architecture']}**, run **{selection['run']}**, using validation macro F1 before test access. "
        "Four ImageNet backbones were benchmarked; two extra head learning rates were tried on the top two architectures. "
        "Only the final linear layer was trained. Full-backbone fine-tuning has not been performed.", "",
        f"Validation accuracy: **{measured['validation']['accuracy']:.2%}**; macro F1: **{measured['validation']['macro_f1']:.4f}**.", "",
        f"Final test accuracy: **{evaluation['test']['accuracy']:.2%}**; macro precision: **{evaluation['test']['macro_precision']:.4f}**; "
        f"macro recall: **{evaluation['test']['macro_recall']:.4f}**; macro F1: **{evaluation['test']['macro_f1']:.4f}**. "
        f"Misclassified {len(errors)} of {evaluation['test_images']} images.", "",
        "| Class | Test precision | Test recall | Test F1 | Images |",
        "|---|---:|---:|---:|---:|"]
    for label in CLASSES:
        row = evaluation["per_class"][label]
        lines.append(f"| {label} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1-score']:.4f} | {int(row['support'])} |")
    lines += ["", "## Validation-only model development", "",
              "| Run | Architecture | Validation macro F1 |", "|---|---|---:|"]
    for candidate in candidates:
        lines.append(f"| {candidate['run']} | {candidate['metrics']['architecture']} | {candidate['metrics']['validation']['macro_f1']:.4f} |")
    lines += ["", f"The selected checkpoint is {measured['checkpoint_size_mb']:.1f} MB, with "
        f"{measured['total_parameters']:,} parameters ({measured['trainable_parameters']:,} trained). "
        f"Measured CPU batch-one forward latency is {measured['inference_ms_batch1']:.1f} ms, excluding preprocessing. "
        f"Training accuracy exceeds validation by {measured['train_validation_accuracy_gap']:.2%}. "
        "The validation score and per-class results favor this candidate; EfficientNet-B0 is a smaller/faster alternative. "
        "The modest differences from tuned alternatives are not statistically established.", "",
        "Model size alone does not explain the results: EfficientNet-B0 outperformed the larger ResNet18, "
        "while ResNet50 performed best in this particular frozen-feature experiment. Additional seeds, "
        "field-disjoint data and a separate fine-tuning study are needed to establish robust conclusions.", "",
        "## Error analysis", ""]
    for count, actual, predicted in confusion_pairs:
        lines.append(f"- {actual} predicted as {predicted}: {count} {'image' if count == 1 else 'images'}.")
    lines += ["", "See `final_evaluation/confusion_matrix.png`, `misclassified.json`, and "
        "`misclassified_examples.png`. Error inspection does not authorize tuning against this test set.", "",
        "## Software delivered", "",
        "Reproducible CLI pipeline, saved PyTorch checkpoints and configuration, local Streamlit image upload, "
        "class scores, model comparison, dataset views and sourced cultural-management information. "
        "The optional mask calculator measures a supplied annotation and excludes image background.", "",
        "## Unfinished components and limitations", "",
        "- Segmentation: no labelled pixel masks are supplied. No segmentation model, mask prediction, disease severity grade, or image-based affected-area claim has been made.",
        "- Treatment: cultural-management references are present, but current regional product labels and application rules remain unverified. No automatic pesticide, spray timing or quantity recommendation is enabled.",
        "- Generalization: one split/seed; no plant/field identity grouping, unknown-class rejection or confidence calibration. Image-level holdout results are not field validation.",
        "- The final v1 test has now been consumed. Further tuning needs an independent holdout or a documented nested protocol.",
        "- No hardware actuation or growth-stage prediction is included.", "",
        "Next required data: reviewed leaf/lesion pixel masks and formulation-specific current agricultural label sources. "
        "See `docs/SEGMENTATION_PLAN.md` and `docs/KNOWLEDGE_BASE.md`."]
    binary = binary_metrics(evaluation["confusion_matrix"], CLASSES)
    (artifacts / "final_evaluation/binary_metrics.json").write_text(json.dumps(binary, indent=2) + "\n")
    lines += ["", "## Healthy versus diseased: final test", "",
              "Derived from the saved six-class confusion matrix; no training or test inference repeated. "
              "The predicted class maps to healthy or diseased without threshold tuning.", "",
              f"Accuracy: **{binary['accuracy']:.2%}**; disease precision: **{binary['disease_precision']:.4f}**; "
              f"disease recall: **{binary['disease_recall']:.4f}**; disease F1: **{binary['disease_f1']:.4f}**.", "",
              "| Actual / Predicted | Healthy | Diseased |", "|---|---:|---:|",
              f"| Healthy | {binary['confusion_matrix'][0][0]} | {binary['confusion_matrix'][0][1]} |",
              f"| Diseased | {binary['confusion_matrix'][1][0]} | {binary['confusion_matrix'][1][1]} |"]
    (artifacts / "PROJECT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Final report written to {artifacts / 'PROJECT_REPORT.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/runs/frozen_baseline")
    args = parser.parse_args()
    generate(args.run)
    final_report(args.run.parent.parent)
