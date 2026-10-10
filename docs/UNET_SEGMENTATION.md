# U-Net: leaf area and affected-area estimation

**An experimental rice U-Net pilot uses 56 corrected, AI-reviewed semantic masks**:
44 training and 12 validation images across all six classes. Sixty source images
were visually inspected, with four deferred because their boundaries were ambiguous.
The remaining 1,885 masks are unreviewed and excluded. No test image is used in training.

These approximate boundaries were reviewed at 512-pixel display resolution; they
are **not expert ground truth**. The checkpoint and application explicitly identify
this as `ai_visual_reviewed`. Validation scores measure agreement with those
references, not independently established disease-segmentation accuracy.

Open `artifacts/boundary_review_20261010/index.html` for original/corrected pairs.
Image-specific polygons and colour-assisted edits are recorded in `edits.json`,
with source/mask hashes in `correction_records.json` and pre-edit backups in `before/`.
`artifacts/unet/summary.json` and `history.json` record the actual training run.
The run completed 40 epochs and selected epoch 32. At original image dimensions,
the 12 validation references give leaf Dice **0.9688**, lesion Dice **0.7057** and
affected-area MAE **3.150 percentage points**. See
[`artifacts/unet/REPORT.md`](../artifacts/unet/REPORT.md) for per-class results and
overlays. Small narrow brown spots and pale scald damage remain underdetected.

## Model and measurement

`rice_disease/unet.py` implements a compact U-Net with four downsampling levels,
skip connections, bilinear upsampling, GroupNorm and three output classes:

| Pixel label | Meaning |
|---|---|
| 0 | Background |
| 1 | Unaffected rice leaf |
| 2 | Affected rice leaf |

This is semantic segmentation, which is sufficient for area measurement. It does
not assign IDs to overlapping leaves. Use a clear single-leaf image for a per-leaf
percentage; multiple leaves produce a combined visible-leaf percentage.

**Affected area (%) = 100 × pixels labelled 2 / pixels labelled 1 or 2.**

Background is excluded. An empty leaf prediction returns `null`, not zero percent.
The result measures visible projected pixels; it is not physical surface area or
field disease prevalence. Image-level classification scores are not used as area.

Default training: 256×256 aspect-preserving letterbox, base width 16, batch size 4,
AdamW at 0.001, weight decay 0.0001, up to 40 epochs and eight-epoch early stopping.
The loss combines weighted cross entropy (weights 1, 1, 3) with soft Dice for the
whole leaf and affected tissue. Padding uses ignore index 255 **internally only**;
annotation files must contain 0, 1 and 2. Masks use nearest-neighbour resizing.
Flips and 90-degree rotations are shared between each training image and mask.
Validation/test receive no augmentation.

The AI pilot configuration uses two CPU threads and cached resized tensors. Its
second run permits up to 40 epochs, with a 24-epoch minimum and patience of 12.
The initial eight-epoch patience stopped at epoch 11 before lesion learning had
progressed; that run is retained for comparison rather than silently overwritten.

The best mean validation leaf/lesion Dice selects the checkpoint. Undefined Dice
terms receive zero for selection and remain `null` in reported metrics. Training
requires reviewed affected tissue in both train and validation; very small lesions
can still disappear at processing resolution. Assess whether a larger input size
is needed using training/validation data only.

Source: [U-Net paper and original project](https://lmbweb.informatik.uni-freiburg.de/people/ronneber/u-net/).
This compact implementation uses modern padding/normalization choices rather than
reproducing every detail of the original biomedical network.

## Review the prepared masks

Audit the actual CSV, source hashes and mask files (including unreviewed drafts):

```powershell
.\.venv\Scripts\python.exe -m rice_disease.mask_audit
```

The report is `artifacts/pixel_mask_audit.json`. It records review counts by split
and disease, missing records, invalid masks and the reasons training is blocked.
File integrity does not establish annotation accuracy. No model is scored by this audit.

The annotation workspace is `artifacts/segmentation_annotations/`:

- `annotations.csv` maps every source image to a mask and retains its original split.
- `masks/<split>/<class>/<source filename>.png` contains editable semantic drafts.
- `summary.json` describes their origin and encoding. Its initial reviewed count is
  a preparation-time value; `annotations.csv` is the authority for later reviews.

These masks were refreshed from the version 3 heuristic candidate instances.
Their summary records the generator version and source-summary checksum. Previous
drafts are archived with the version 2 instance export. The revision reduces broad
false regions, but still has the limitations documented in `INSTANCE_SEGMENTATION.md`. Correct
the full leaf boundary and every lesion; healthy images also need reviewed leaf
boundaries and truly empty lesion regions. Do not merely change review flags on
uncorrected drafts. Save single-channel integer PNGs at the original EXIF-corrected
image dimensions, preserving values 0, 1 and 2.

After reviewing and correcting a mask, set its CSV `review_status` to `reviewed`
and fill in `reviewed_by`. AI visual reviews additionally use `review_method=ai_visual`
and a `review_record` path; do not represent them as human or expert annotations.
Keep `image_path`, `split` and `source_sha256` unchanged. The loader
rejects missing reviewers, altered source hashes, changed split membership, duplicate
image entries, invalid labels, dimension mismatch and empty-leaf annotations.
Unreviewed masks are excluded. Actual mask hashes and reviewers are recorded at
training time so corrected files have traceable provenance.

Start with a reviewed pilot spanning all disease classes, healthy leaves, small
lesions and pale/edge symptoms. The two-image minimum per train/validation split is
a software guard, **not a recommendation for sufficient training data**. Preserve
test masks for an independent evaluation after model selection.

To prepare another workspace, use a fresh output path:

```powershell
.\.venv\Scripts\python.exe -m rice_disease.segmentation_data --output artifacts/new_annotations
```

## Train, evaluate and predict

For a new run on the reviewed subset (preserve existing outputs):

```powershell
.\.venv\Scripts\python.exe -u -m rice_disease.train_segmentation train --config configs/segmentation_ai_pilot.json --output artifacts/unet_next
.\.venv\Scripts\python.exe -m rice_disease.train_segmentation predict "path/to/leaf.jpg"
```

Use `--help` for paths and device options. Defaults use CPU. Training configuration
is in `configs/segmentation.json`. Existing nonempty training/prediction directories
and existing evaluation reports are protected; use a new output for a new run.

Training writes `artifacts/unet/model.pt`, config, history, annotation provenance
and summary. **Only reviewed training and validation masks are loaded during
training.** Test evaluation is a separate command. It reports original-resolution
leaf/lesion IoU and Dice, per-disease results, affected-area MAE in percentage points,
empty-lesion false positives and missing-leaf predictions. MAE excludes cases
without a predicted/reference leaf denominator and reports the comparable count.
It evaluates the available reviewed test subset; inspect its image count rather
than assuming the entire holdout was annotated.

There are currently no reviewed test masks, so the separate `evaluate` command is
not yet applicable. `scripts/review_unet_validation.py` exports original-resolution
validation metrics and reference/prediction overlays without reading test images.
The pilot enables optional in-memory caching of resized image/mask tensors; paired
augmentation does not modify the cached originals.

Inference restores unpadded logits to native image size before choosing labels.
Prediction exports `mask.png`, `overlay.png` and `area.json` under
`artifacts/unet_prediction/`. Estimated percentages remain model predictions even
after training on reviewed annotations; field validation is a separate requirement.

The Streamlit app automatically loads `artifacts/unet/model.pt` after training.
Before that checkpoint exists it explains what is missing. It never substitutes
the heuristic masks or random weights for a trained U-Net.

## Tripura and the 100 mL spray tank

`configs/application.json` stores **India / Tripura / rice / 100 mL = 0.1 L**.
This volume is not 100 litres. A verified disease-specific product-label rule may
calculate product quantity as:

- For a label rate in g/L or mL/L: rate × **0.1 L**.
- For a rate per hectare: rate × the **actual area treated** by the mixture.

Do not multiply pesticide rate by the percentage of damaged pixels or infer field
area from the image. Crop, disease, exact product/formulation, application conditions,
label units and regional use must be established. The app connects the existing
verified-rule quantity helper to the configured tank, with an explicit confirmation
that diagnosis and label conditions have been checked. It does not enable a product
simply because its disease name matches a classification result.

No product rule has been enabled. Research found an official PPQS rice-blast label
and an ICAR Tripura advisory, but they do not establish a universal disease-to-dose
mapping for all five dataset diseases or this user's crop stage and product pack.
The PPQS label has disease-specific application timing; a leaf photograph does not
provide that timing. See [pesticide source review](PESTICIDE_TRIPURA_REVIEW.md).

## Single-photo workflow (9 October 2026)

The requested input is one rice-plant photograph. The classifier identifies the
disease, U-Net estimates visible affected leaf percentage when a trained checkpoint
exists, and a disease lookup supplies a product reference where verified source
material is available. No field-area input is required for the reference workflow:

```powershell
.\.venv\Scripts\python.exe -m rice_disease.inference "path/to/plant.jpg" --water-ml 100
```

`--water-ml` controls dilution-reference arithmetic only. The photo does not measure
spray coverage or reveal how much mixture the plant needs. Neither image size nor
affected percentage is used to scale product concentration. If U-Net is missing,
the CLI reports an unavailable segmentation result and leaves affected area null.

`knowledge_base/product_references.json` contains a conditional source reference
for blast and brown spot, shared by the CLI and app. Other diseases explicitly
report a missing verified label in the catalog, and healthy predictions suggest
no product. This reference is separate from an application-authorized chemical rule.

For a separately verified product rule and confirmed disease, U-Net prediction can
also export actual label arithmetic in `area.json` alongside the mask and overlay:

```powershell
.\.venv\Scripts\python.exe -m rice_disease.train_segmentation predict "path/to/plant.jpg" --rule "path/to/verified_rule.json" --disease leaf_blast --region "India/Tripura" --tank-ml 100 --application-authorized
```

Supply `--area-hectares` only when that rule uses a per-hectare rate. A per-hectare
rule cannot yield a plant application quantity from a photograph alone. Use
`--output` for a fresh prediction directory. The combined export retains the rate,
units and calculation basis so the numeric result can be checked.
