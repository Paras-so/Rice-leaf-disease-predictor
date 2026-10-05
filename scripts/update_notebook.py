"""Generate the local, reproducible walkthrough from the project modules."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


cells = [
    markdown("""# Rice disease classification: executed project workflow

This notebook follows `docs/reference/problem and guide.pdf`. Run from the project directory with
the project Python environment. The CLI modules perform the work; saved reports
make every stage reviewable. The original cleaning notebook is preserved.

The current model study compares frozen pretrained features, then tunes the final
linear layer on the top two architectures. No growth-stage model or hardware is included.
"""),
    code("""import json
from pathlib import Path
import pandas as pd
from IPython.display import display
from rice_disease.data import ROOT, load_manifest

artifacts = ROOT / 'artifacts'
print('Project:', ROOT)
"""),
    markdown("""## 1. Dataset verification

Every image was fully decoded and checked for identical decoded pixels. Inspect
the audit below. Source images remain unchanged. To repeat the audit deliberately,
run `python -m rice_disease.data`; it refuses to overwrite a different split.
"""),
    code("""audit = json.loads((artifacts / 'dataset_audit.json').read_text())
display(audit)
"""),
    markdown("""## 2. Train, validation and test membership

Retain the verified 389-image holdout. Stratify the original 1,552-image development
pool into 1,241 training and 311 validation images with seed 42. Validation supports
architecture selection and stopping. Test evaluation happens only after selection.
"""),
    code("""rows, split = load_manifest(artifacts)
display(pd.DataFrame(split['split_counts']))
print('Manifest SHA-256:', split['manifest_sha256'])
"""),
    markdown("""## 3. Common preprocessing and training configuration

RGB, full-image resize to 224x224, ImageNet normalization. Original and one seeded
augmented view per training image; no augmentation for validation or test. Features
are extracted with frozen batch normalization/dropout and cached. Only the six-class
final linear layer is trained. Full-backbone fine-tuning has not been performed.
"""),
    code("""config = json.loads((ROOT / 'configs/benchmark.json').read_text())
display(config)
"""),
    markdown("""## 4. Four-model benchmark

The following call resumes incomplete models and preserves completed results.
First-time execution requires official weights: `python -m rice_disease.download_weights`.
Training can take several minutes on CPU. Selection uses validation macro F1.
"""),
    code("""from rice_disease.benchmark import run_benchmark
results = run_benchmark(ROOT / 'configs/benchmark.json', artifacts, 'frozen_baseline')
display(pd.read_csv(artifacts / 'runs/frozen_baseline/comparison.csv'))
"""),
    markdown("""## 5. Targeted head learning-rate tuning

After all four baseline models complete, compare rates 0.0003 and 0.003 on the two
highest validation-F1 architectures. All other settings remain fixed. Freeze the
winner and its checkpoint hash before using test data. Rerunning returns an existing
frozen selection, rather than starting further tuning.
"""),
    code("""from rice_disease.tune import tune
selection = tune(artifacts)
display({'checkpoint': selection['checkpoint'], 'validation': selection['metrics']['validation']})
"""),
    markdown("""## 6. Final held-out evaluation

This step consumes the reserved test set once for the frozen v1 configuration.
Do not tune against the resulting scores. Rerunning reads the existing report.
"""),
    code("""from rice_disease.evaluate import evaluate
evaluation = evaluate(artifacts)
display(evaluation['test'])
display(pd.DataFrame(evaluation['per_class']).T)
"""),
    markdown("""## 7. Figures, application, and remaining data requirements

Run `python -m rice_disease.report` for learning curves and confusion matrices.
Launch `python -m streamlit run app.py --server.address 127.0.0.1` for the laptop app.

No pixel-level masks exist in the supplied data. Segmentation training requires
reviewed annotations for background, unaffected leaf, and lesions. See
`docs/SEGMENTATION_PLAN.md`. The app can measure a supplied annotation but cannot
predict affected area from classification scores.

Cultural-management references are recorded in `knowledge_base/diseases.json`.
Chemical rules and quantities remain disabled until current product labels,
jurisdiction, formulation, dose and application conditions are verified. See
`docs/KNOWLEDGE_BASE.md`. No dosage or treatment threshold has been invented.
"""),
]

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
            "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 4}
(ROOT / "RiceDiseaseBenchmark.ipynb").write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
print("Notebook updated.")
