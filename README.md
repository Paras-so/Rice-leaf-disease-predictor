# Smart Rice Disease Detection and Pesticide Recommendation System

Laptop-only software following `docs/reference/problem and guide.pdf`. No hardware control or
growth-stage prediction is included. `FinalYearProject.ipynb` remains the original
cleaning record. The latest user instruction authorizes executing and reviewing
stages autonomously, without requesting command output from the user.

## Start here

- [Instance segmentation and U-Net: complete step-by-step guide](docs/SEGMENTATION_STEP_BY_STEP.md)
- [Folder map and cleanup notes](docs/FOLDER_GUIDE.md)
- [Project overview](docs/PROJECT_OVERVIEW.md)
- [Classifier workflow and experiment history](docs/PROJECT_STEP_BY_STEP.md)
- [Project decisions and verified outcomes](docs/PROJECT_LOG.md)

## Run the application

From this project directory:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Install dependencies once, or again if you recreate `.venv`. Use this same project
Python for both installation and launching the app. The project Streamlit config
disables usage telemetry so browser sessions do not need to create a telemetry
identifier in the Windows user profile.

Open http://127.0.0.1:8501. Choose a rice leaf image and click **Analyze leaf** for a six-class prediction,
uncalibrated model scores, and sourced cultural-management information. The app uses
only the final selected checkpoint; there is no classifier dropdown. The Final model
and Dataset tabs show its results and dataset details.

An experimental three-class U-Net was trained for 40 epochs using 56 corrected AI-reviewed masks (44 training,
12 validation); 1,885 masks remain unreviewed. These are approximate references,
not expert ground truth. The app loads `artifacts/unet/model.pt` and identifies
the pilot's annotation quality. An optional supplied annotation can be measured,
explicitly labelled as annotation-derived. No current, region-specific pesticide
label has been verified, so automatic chemical treatment, spray timing, and dosage
outputs remain unavailable. This is a research prototype.

## Verified dataset

For instance-mask inspection, use the single active folder
`artifacts/instance_segmentation/` and open its `index.html`. It shows the full
dataset with originals beside draft overlays and filters for review flags.
Previous pilot/full runs are under `artifacts/archive/segmentation_before_v3_20261009/`.
See [segmentation repair and remaining review requirements](docs/INSTANCE_SEGMENTATION.md).

- 1,941 fully decoded RGB images; no unreadable images or identical-pixel duplicates.
- 1,939 images are 1600x1600; two are 1600x1548.
- Six classes with 297 to 348 images each; maximum/minimum ratio 1.17.
- Final split: **1,241 train / 311 validation / 389 test**, seed 42.
- The verified existing test membership is deliberately retained. Validation is
  stratified from the remaining development pool. Source images are never moved.
- `artifacts/split_manifest.csv` records membership and hashes;
  `artifacts/split_summary.json` records its checksum and class counts.
- Hashes do not establish independence of transformed near-duplicates or images
  from the same plant. Plant/field identifiers are not supplied.

## Reproduce the pipeline

### Inspect preprocessing output

`image_size` is defined in `configs/benchmark.json` (currently **224**), alongside
the RGB normalization `mean` and `std`. Training and inference pass this config to
`image_tensor`; loading the Python file alone previously did not call that function.
Running it now produces console output, viewable resized PNGs, and `summary.json`
with original/output dimensions, normalized tensor shapes, ranges, and any failures.

```powershell
# One training-image preview per class, using the prepared split manifest
.\.venv\Scripts\python.exe rice_disease/preprocessing.py

# Process one image or every supported image recursively in a folder
.\.venv\Scripts\python.exe -m rice_disease.preprocessing --input "path/to/leaf.jpg"
.\.venv\Scripts\python.exe -m rice_disease.preprocessing --input "path/to/images" --output "artifacts/my_preprocessing"

# Optional export-only size override; does not change trained-model settings
.\.venv\Scripts\python.exe -m rice_disease.preprocessing --input "path/to/leaf.jpg" --image-size 224
```

Default output: `artifacts/preprocessing/images/` and
`artifacts/preprocessing/summary.json`. Nested folders and original filenames are
retained (with `.png` appended). Source images are preserved. Exported PNGs show
the resized RGB image; normalization is applied to tensors, not saved as display
colors. This preview/export has no training augmentation. Failures are printed,
recorded in the summary, and return a nonzero exit code.

Every executable module in `rice_disease` supports `--help` and both
`python -m rice_disease.<module>` and `python rice_disease/<module>.py`.
The helper modules now provide these outputs:

| Module | Output / input |
|---|---|
| `models` | Lists available architectures and training/inference commands |
| `binary --class-name healthy` | Prints healthy/diseased status as JSON |
| `binary --metrics artifacts/final_evaluation/metrics.json` | Prints binary metrics from saved confusion counts |
| `segmentation --mask path/to/mask.png` | Prints annotated leaf/disease pixel counts and area percentage |
| `quantity --help` | Shows required rule and volume/area arguments; no verified rules are bundled |

Helpers that require inputs display usage when run without arguments. `inference`
requires an image path. `__init__.py` is a package marker, not a pipeline command.
Existing tuning and evaluation results are printed when reused; benchmark output
includes validation scores and the results path.

### Generate leaf and lesion candidate instances

```powershell
.\.venv\Scripts\python.exe -u -m rice_disease.instance_segmentation
```

This runs a **colour and connected-component baseline**, not a trained instance
segmentation network, over the cleaned dataset. Results are in
`artifacts/instance_segmentation/`: native-size 16-bit PNGs in `leaf_instances/`
and `lesion_instances/`, visual `overlays/`, per-image `metadata/`, `instances.csv`,
`summary.json`, and `preview.jpg`. Each nonzero pixel value identifies an instance
within that image and category. Lesion metadata links each candidate to its leaf.
The existing train/validation/test assignments and source images are preserved.

All masks are **unreviewed pseudo-masks**. Colour changes can be unrelated to
disease, touching objects can merge, and small or pale lesions can be missed.
Image-level disease labels are used for reporting only; they do not determine
the masks. The outputs are not used by the classifier or app, and do not provide
validated disease area, severity, or segmentation accuracy. Human annotation is
still needed to train and evaluate a disease segmentation model.

For a balanced training-only pilot, use `--split train --limit 12 --output
artifacts/my_segmentation_pilot` (on one command line). Existing nonempty output
directories are protected: choose a new `--output` when rerunning. See
[the instance segmentation guide](docs/INSTANCE_SEGMENTATION.md) for mask usage.

### Train U-Net and estimate affected leaf percentage

The semantic annotations are in `artifacts/segmentation_annotations/`.
Open `artifacts/boundary_review_20261010/index.html` to inspect the 56 corrected
references with their original images. Four of 60 inspected sources were deferred.
Correct the masks (0=background, 1=unaffected leaf, 2=affected leaf), then mark
reviewed entries in `annotations.csv` with a reviewer identity. U-Net excludes
unreviewed masks and trains only on the preserved training/validation splits.

```powershell
.\.venv\Scripts\python.exe -u -m rice_disease.train_segmentation train --config configs/segmentation_ai_pilot.json --output artifacts/unet_next
.\.venv\Scripts\python.exe -m rice_disease.train_segmentation predict "path/to/leaf.jpg"
```

The app will use `artifacts/unet/model.pt` when available. Percentage is affected
pixels divided by total leaf pixels, excluding background. See `artifacts/unet/summary.json`
for the actual run. Validation measures agreement with approximate AI-reviewed masks;
no expert accuracy claim is made. Test evaluation requires reviewed test masks.
The selected epoch 32 achieves 0.9688 leaf Dice, 0.7057 lesion Dice and 3.150
percentage points of area error at original resolution. See the
[training report and prediction overlays](artifacts/unet/REPORT.md), including
remaining misses on small lesions and pale scald damage.

The tank configuration is **100 mL (0.1 L), Tripura, India**. Pesticide amount uses
a verified disease-specific product-label rate and volume or treated area; it is
not scaled by affected percentage. No chemical rule is enabled yet.
See [U-Net and annotation instructions](docs/UNET_SEGMENTATION.md) and
[product-label review status](docs/PESTICIDE_TRIPURA_REVIEW.md).

### Train and evaluate classification

The virtual environment contains CPU PyTorch; CUDA is unavailable here. For a new
environment, install `requirements.txt`. Official weights require internet once.

```powershell
# Full image audit and reproducible split manifest
.\.venv\Scripts\python.exe -u -m rice_disease.data

# Official pretrained checkpoints, cached inside the project
.\.venv\Scripts\python.exe -u -m rice_disease.download_weights

# Four models; only train and validation images are loaded
.\.venv\Scripts\python.exe -u -m rice_disease.benchmark

# Compare 52 candidates (original baselines plus optimizer/loss/hyperparameter variants)
.\.venv\Scripts\python.exe -u -m rice_disease.tune

# Evaluate the frozen winner; save separately from the historical v1 evaluation
.\.venv\Scripts\python.exe -u -m rice_disease.evaluate

# Figures and benchmark report
.\.venv\Scripts\python.exe -m rice_disease.report

# Verification
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Completed baseline runs preserve saved results. Tuning resumes the versioned
`hyperparameter_search_v2` experiment and activates its verified winner when complete.
Final evaluation returns its recorded result rather than repeatedly scoring the test
set. The 389-image holdout was already evaluated for v1: the new evaluation is a reused
holdout, not independent confirmation. Search uses validation macro F1 only.

`RiceDiseaseBenchmark.ipynb` provides a local notebook view of stages and outputs.
It uses the package files here and is not a standalone Colab upload. Run it from
this project directory with the project environment.

## Experimental choices

`configs/benchmark.json` specifies shared settings. The initial study uses **frozen
ImageNet backbones and trains only the final linear layer** for MobileNetV3-Small,
EfficientNet-B0, ResNet18, and ResNet50. It measures transfer-feature quality, not
full-backbone fine-tuning performance.

RGB inputs are resized to 224x224 and normalized with ImageNet mean/std. Full-image
resize preserves leaf-edge symptoms instead of center-cropping them. The two slightly
rectangular images undergo a small aspect-ratio change. Each training image has one
original and one seeded augmented view: flips, 90-degree rotations, brightness and
contrast within 10%. Validation/test images have no augmentation. Backbone batch
normalization and dropout remain in evaluation mode. Frozen features are cached.

Unchanged baseline settings: cross-entropy, AdamW, batch size 32, learning rate 0.001, weight
decay 0.0001, up to 40 epochs, early stopping after eight epochs without improved
validation macro F1. Mild class imbalance does not justify initial oversampling.
The new search compares AdamW, Adam and SGD with momentum 0.9 against cross-entropy,
label-smoothed cross-entropy (0.1) and focal loss (gamma 2). Adam/AdamW use 0.001;
SGD uses 0.01. Four additional AdamW/cross-entropy recipes change learning rate
(0.0003 or 0.003), batch size (64), or weight decay (0.001), one at a time.
All four architectures receive every recipe: 48 new trials plus four unchanged
baselines. Epoch limit, patience, seed, split and preprocessing stay fixed.
Candidates rank by validation macro F1, then parameter count; exact ties keep the
earlier candidate. The final deployment directory contains only the winner, while
research checkpoints and comparison records remain saved for reproducibility.
Single-seed differences are descriptive, not statistically established superiority.
Exact checkpoint versions use `IMAGENET1K_V1`.

Sources: [TorchVision models](https://docs.pytorch.org/vision/stable/models.html) and
[PyTorch transfer learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

## Outputs and remaining requirements

- `artifacts/dataset_audit.json`: dimensions, readability and imbalance.
- `artifacts/runs/frozen_baseline/`: weights, histories, metrics, CSV and figures.
- `artifacts/tuning_candidates.json`: historical v1 learning-rate study.
- `artifacts/experiments/hyperparameter_search_v2/`: search plan, ranked CSV, all candidate results and frozen selection.
- `artifacts/final_model/hyperparameter_search_v2/`: one final checkpoint and its training configuration.
- `artifacts/selection_history/`: previous selection records preserved before activation.
- `artifacts/selected_model.json`: frozen checkpoint identity and rationale.
- `artifacts/final_evaluation/`: historical v1 test results.
- `artifacts/evaluations/hyperparameter_search_v2/`: new winner's evaluation on the previously used holdout.
- `docs/PROJECT_LOG.md`: decisions and verified outcomes.
- [Segmentation annotation plan](docs/SEGMENTATION_PLAN.md): required pixel-mask data.
- [Agricultural sources](docs/KNOWLEDGE_BASE.md): source regions and missing label verification.

CLI inference:

```powershell
.\.venv\Scripts\python.exe -m rice_disease.inference "path/to/leaf.jpg"
```

For pixel-mask review status, run `python -m rice_disease.mask_audit` using the
project virtual environment; it saves `artifacts/pixel_mask_audit.json`.
The U-Net trainer requires corrected, reviewed masks. Single-photo inference
automatically uses `artifacts/unet/model.pt` when available and includes a
disease-matched product reference. `--water-ml 100` selects the dilution-reference
volume, not a measured spray requirement for one plant. See
[U-Net commands and measurement limits](docs/UNET_SEGMENTATION.md).

The default is the selected ResNet50 checkpoint in `selected_model.json`.
Optionally supply `--checkpoint` to use another saved model. Output includes
`status` (healthy/diseased), `has_disease`, the six-class prediction, and model scores.
Binary status maps the top-scoring class: healthy stays healthy; all five disease
classes map to diseased. This does not train a separate binary classifier.
The report command also creates `binary_comparison.csv` (validation for all four
models) and `final_evaluation/binary_metrics.json` (selected model's test results)
from saved confusion matrices, without repeating inference or selecting a threshold.
