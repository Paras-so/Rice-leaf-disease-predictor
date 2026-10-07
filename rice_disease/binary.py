"""Healthy/diseased reporting using a fixed mapping of the six-class prediction."""

import numpy as np


def health_status(class_name):
    return "healthy" if class_name == "healthy" else "diseased"


def binary_metrics(matrix, classes):
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.shape != (len(classes), len(classes)) or "healthy" not in classes:
        raise ValueError("Expected a confusion matrix with a healthy class.")
    if (matrix < 0).any() or not matrix.sum():
        raise ValueError("Confusion counts must be nonnegative with at least one image.")
    healthy = classes.index("healthy")
    disease = [i for i in range(len(classes)) if i != healthy]
    tn = int(matrix[healthy, healthy])
    fp = int(matrix[healthy, disease].sum())
    fn = int(matrix[disease, healthy].sum())
    tp = int(matrix[np.ix_(disease, disease)].sum())
    def divide(a, b):
        return a / b if b else 0.0
    return {
        "decision_rule": "Map the top-scoring class to healthy or diseased.",
        "positive_class": "diseased",
        "accuracy": divide(tp + tn, tp + tn + fp + fn),
        "disease_precision": divide(tp, tp + fp),
        "disease_recall": divide(tp, tp + fn),
        "disease_f1": divide(2 * tp, 2 * tp + fp + fn),
        "healthy_recall": divide(tn, tn + fp),
        "balanced_accuracy": (divide(tp, tp + fn) + divide(tn, tn + fp)) / 2,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "confusion_labels": ["healthy", "diseased"],
        "images": tp + tn + fp + fn,
    }


if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from rice_disease.data import CLASSES
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--class-name", choices=CLASSES, help="Map a predicted class to healthy/diseased")
    source.add_argument("--metrics", type=Path, help="Saved final evaluation metrics.json")
    args = parser.parse_args()
    if args.class_name:
        print(json.dumps({"class_name": args.class_name, "status": health_status(args.class_name)}, indent=2))
    elif args.metrics:
        try:
            result = json.loads(args.metrics.read_text(encoding="utf-8"))
            print(json.dumps(binary_metrics(result["confusion_matrix"], CLASSES), indent=2))
        except (OSError, ValueError, KeyError) as exc:
            parser.exit(1, f"Binary reporting error: {exc}\n")
    else:
        parser.print_help()
