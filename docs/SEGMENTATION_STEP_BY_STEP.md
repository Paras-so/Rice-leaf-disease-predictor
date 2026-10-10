# Instance segmentation and U-Net training: step by step

This guide describes the code and files in this project. Run commands in PowerShell
from the project root using `.venv\Scripts\python.exe`.

## 1. Understand the two stages

```text
Original rice photos + fixed train/validation/test split
    -> colour/morphology candidate generator
    -> leaf instance IDs + lesion candidate IDs
    -> convert to three-class semantic masks
    -> inspect and correct each mask; record its reviewer
    -> train U-Net on reviewed training masks
    -> select the checkpoint using reviewed validation masks
    -> evaluate once on independently reviewed test masks
    -> predict a new photo's mask and affected-area percentage
```

The instance generator is classical image processing, not a trained instance
segmentation network. Each disconnected leaf/lesion region receives an ID. The
U-Net is a **semantic segmentation** network: each pixel receives a tissue class,
not an individual leaf ID. The separate classifier predicts the disease name.

## 2. Check the environment and existing work

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
```

Only if setting up a new environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The installed environment and downloaded pretrained classifier weights are kept
during cleanup. U-Net itself starts from newly initialized weights when training.

The current project already has the following completed stages:

| Stage | Current location |
|---|---|
| Audited image split | `artifacts/split_manifest.csv`, `split_summary.json` |
| Candidate instances for 1,941 images | `artifacts/instance_segmentation/` |
| Editable semantic masks | `artifacts/segmentation_annotations/` |
| Correction records and original/corrected gallery | `artifacts/boundary_review_20261010/` |
| Active U-Net and training history | `artifacts/unet/` |

There are 56 corrected AI-reviewed masks: 44 training and 12 validation. The other
1,885 masks are unreviewed. There are no reviewed test masks. To continue this
existing work, skip generation in steps 4 and 5 and proceed to mask review.

## 3. Preserve the source dataset and split

`artifacts/split_summary.json` identifies the cleaned dataset directory.
`artifacts/split_manifest.csv` records image paths, class names, hashes and split
membership: 1,241 training, 311 validation and 389 test images.

Do not move the photos or randomly split their masks again. Every derivative of a
photo must retain that photo's split. The raw `archive (7)/` folder is source data,
not a disposable generated cache.

If starting a separate project copy without manifests, the preparation entry point is:

```powershell
.\.venv\Scripts\python.exe -m rice_disease.data --help
```

Use its dataset/output options for that new copy. Do not replace this project's
frozen manifests: the trained classifier and U-Net record their hashes.

## 4. Generate and validate candidate instances

To inspect existing results, open `artifacts/instance_segmentation/index.html`.
For a new full export, choose a fresh output directory:

```powershell
.\.venv\Scripts\python.exe -u -m rice_disease.instance_segmentation --output artifacts/instance_segmentation_next
.\.venv\Scripts\python.exe scripts/validate_instance_masks.py --output artifacts/instance_segmentation_next
.\.venv\Scripts\python.exe scripts/build_segmentation_review.py --output artifacts/instance_segmentation_next
```

The generator verifies source hashes, processes images at a maximum side of 800
pixels by default, finds candidate foreground and lesions from colour/morphology,
and restores instance labels to native dimensions with nearest-neighbour resizing.
Use `--help` to see the available parameters.

| Output | What to inspect |
|---|---|
| `leaf_instances/` | uint16 PNG; 0 background, positive values are leaf IDs |
| `lesion_instances/` | uint16 PNG; 0 no lesion candidate, positive values are candidate IDs |
| `metadata/` | Source hashes, dimensions, component areas, boxes, parent IDs and flags |
| `overlays/` | Cyan leaf outlines and red lesion candidates |
| `instances.csv`, `summary.json` | Processed counts, split provenance and failures |
| `validation.json` | Saved-file consistency checks |
| `index.html`, `review_queue.csv` | Visual review and prioritized problem cases |

Require zero processing failures and a passing validator before conversion.
Validation checks file consistency, not whether a region is biologically diseased.
Review healthy leaves, shadows, pale symptoms and large connected candidates.

For algorithm development, use training images only, for example:

```powershell
.\.venv\Scripts\python.exe -m rice_disease.instance_segmentation --split train --limit 36 --output .cache/segmentation_repair/new_pilot
```

A limited pilot cannot supply the full annotation conversion below. Finalize
parameters on training examples before generating the full export.

## 5. Convert instance IDs to semantic labels

For a fresh full export from step 4:

```powershell
.\.venv\Scripts\python.exe -m rice_disease.segmentation_data --candidates artifacts/instance_segmentation_next --output artifacts/segmentation_annotations_next
```

This creates `annotations.csv`, `summary.json` and native-size PNGs under `masks/`.
All new drafts start as unreviewed. Existing nonempty outputs are protected.

| Semantic value | Meaning |
|---|---|
| 0 | Background |
| 1 | Unaffected leaf |
| 2 | Affected leaf |

Conversion uses `leaf_instances > 0` for leaf pixels, then sets lesion candidates
to 2. Instance IDs must never be interpreted directly as these semantic labels.
The active corrected semantic masks are separate from the original candidate
instances; correcting one does not update the other automatically.

## 6. Review and correct the masks

Choose the workspace you are actually reviewing. For the existing corrected set:

```powershell
$annotationManifest = 'artifacts/segmentation_annotations/annotations.csv'
```

For the freshly generated set, use this instead:

```powershell
$annotationManifest = 'artifacts/segmentation_annotations_next/annotations.csv'
```

For each selected training/validation photo:

1. Inspect the orientation-corrected original beside its mask.
2. Correct the whole leaf boundary, remove background/shadows, and mark the
   affected tissue. A healthy photo still needs an accurate leaf boundary.
3. Save a single-channel integer PNG at the original photo dimensions. Its only
   allowed values are 0, 1 and 2. Do not save a coloured overlay as the label mask.
4. Check alignment and small/pale lesions at native resolution. Defer ambiguous
   images instead of marking uncertain drafts as reviewed.
5. Set that row's `review_status` to `reviewed` and enter `reviewed_by` only after
   inspecting and correcting it. Keep `image_path`, `mask_path`, `source_sha256`,
   `class_name` and `split` consistent with the original records.
6. Record the actual review method. AI visual review uses `review_method=ai_visual`
   and a `review_record` path. It must not be represented as expert annotation.

There is no interactive mask editor in the app. Use a label editor that preserves
these integer values, or traceable image-specific edits. The existing
`boundary_review_20261010/edits.json` and `scripts/apply_boundary_review.py` document
the pilot's polygon/colour-assisted corrections. That committed workspace includes
backups and hashes; do not replay edits into it. Its `render`, `commit` and `report`
commands require a prepared review workspace with `selection.json` and `edits.json`.

## 7. Audit annotation integrity and training readiness

```powershell
.\.venv\Scripts\python.exe -m rice_disease.mask_audit --annotations $annotationManifest --output artifacts/pixel_mask_audit.json
```

Read `artifacts/pixel_mask_audit.json`. Resolve missing images/masks, invalid pixel
values, altered hashes, wrong dimensions, duplicate records or missing reviewers.
The trainer requires at least two reviewed images in both training and validation,
and affected pixels in both subsets. This is only a software minimum; build a
representative reviewed dataset across all classes and symptom types.

Unreviewed masks are excluded from training. Keep test annotations independent
of model and threshold selection.

## 8. Set the U-Net training configuration

The current pilot uses `configs/segmentation_ai_pilot.json`:

| Setting | Value |
|---|---|
| Input size | 256 × 256, aspect-preserving letterbox |
| Base channels / batch size | 16 / 4 |
| Maximum / minimum epochs | 40 / 24 |
| Early-stopping patience | 12 epochs |
| Optimizer / learning rate | AdamW / 0.001 |
| Weight decay / seed | 0.0001 / 42 |
| CPU threads / resized-tensor caching | 2 / enabled |

The model has four downsampling levels, skip connections, GroupNorm and three
output classes. Loss combines weighted cross entropy and leaf/lesion Dice.
Training augmentation applies the same flips/90-degree rotations to image and
mask. Validation receives no augmentation. Letterbox padding uses ignored label
255 internally; do not put 255 in annotation PNGs.

Copy the config to a new file before changing experiment settings. Change input
size only to a multiple of 16, at least 32. Choose settings using training and
validation results, never test results.

## 9. Train into a fresh run directory

```powershell
.\.venv\Scripts\python.exe -u -m rice_disease.train_segmentation train --annotations $annotationManifest --config configs/segmentation_ai_pilot.json --output artifacts/unet_next --device cpu
```

This trains a new model; it does not resume `artifacts/unet/model.pt`. Choose a
different output name if `unet_next` already contains files. CUDA is optional only
when available in the installed PyTorch environment.

Each epoch records train/validation loss, leaf/lesion Dice and the selection score.
The checkpoint with the highest mean validation leaf and lesion Dice is saved.
Training uses reviewed train/validation data only; it does not evaluate the test set.

Keep the whole run directory:

- `model.pt`: selected weights, config, epoch and provenance hashes.
- `config.json`: exact settings.
- `history.json`: epoch metrics and losses.
- `training_manifest.json`: reviewed images, mask hashes and review provenance.
- `summary.json`: run counts, selected score and annotation quality.

The active pilot completed 40 epochs and selected epoch 32. Its original-resolution
validation leaf Dice is 0.9688, lesion Dice 0.7057, and affected-area MAE 3.150
percentage points. These measure agreement with 12 approximate AI-reviewed masks.

## 10. Review validation predictions

For runs trained with the active `artifacts/segmentation_annotations/annotations.csv`:

```powershell
.\.venv\Scripts\python.exe scripts/review_unet_validation.py --checkpoint artifacts/unet_next/model.pt --output artifacts/unet_next/validation_review
```

This helper currently reads the active annotation manifest. For a different
annotation workspace it will reject mismatching provenance; do not bypass that
check. Use that run's `history.json` and the prediction command in step 12 to inspect
its validation images against their corresponding masks.

The helper creates native-resolution metrics, predicted masks/overlays, comparison
images and a training/area plot. It checks that annotations still match the training
manifest. A changed reviewed mask requires a new training run for comparable results.
Look at lesion errors as well as leaf overlap; good leaf Dice can hide missed lesions.

## 11. Evaluate a frozen model on reviewed test masks

The current project has no reviewed test masks, so **this step is currently blocked
by annotation availability**. After independently reviewing the held-out test masks
and freezing the model/settings:

```powershell
.\.venv\Scripts\python.exe -m rice_disease.train_segmentation evaluate --checkpoint artifacts/unet_next/model.pt --annotations $annotationManifest --output artifacts/unet_next/test_metrics.json
```

The command reports native-resolution leaf/lesion IoU and Dice, per-disease results,
affected-area MAE, missing-leaf cases and false positives on empty-lesion images.
Inspect the evaluated mask count: only available reviewed test masks are scored.
Existing evaluation files are protected from overwriting.

## 12. Predict a new photo and run the application

Replace the example image path with an existing photo:

```powershell
.\.venv\Scripts\python.exe -m rice_disease.train_segmentation predict "path/to/leaf.jpg" --checkpoint artifacts/unet_next/model.pt --output artifacts/unet_prediction_next
```

Outputs are `mask.png`, `overlay.png` and `area.json`. Inference restores unpadded
logits to the original image dimensions before selecting pixel labels.

```text
Affected leaf area (%) = 100 × count(label 2) / count(label 1 or 2)
```

Background is excluded; no predicted leaf means unavailable, not 0%. Multiple leaves
produce a combined visible-leaf percentage. The classifier's disease scores are
separate from this percentage and are not per-photo accuracy measurements.

The application uses the existing active U-Net at `artifacts/unet/model.pt`:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501`, upload a photo and click **Analyze leaf**. It shows one
bounded masked image, disease scores, affected area and the catalog's disease-matched
product reference when available. Training `unet_next` does not switch the app's
checkpoint. To deliberately activate a reviewed new run, preserve the current run
and update the app's `segmentation_path` to that run's `model.pt`; keep its associated
config, history and provenance together. The CLI can select it with
`--segmentation-checkpoint artifacts/unet_next/model.pt`.

## 13. Verify changes and retain the useful files

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_instance_segmentation.py'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_unet.py'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p '*app.py'
```

Keep source photos, manifests, corrected masks, review records, model checkpoints,
configs and metrics. Python bytecode, disposable `.cache/` previews, derived image/
feature caches and old console logs can be regenerated. Downloaded pretrained weights
are retained for offline classifier training. See [the folder map](FOLDER_GUIDE.md).
