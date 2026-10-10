# Project decision log

## 2026-10-02 — Stage 1 prepared

- Read `project problem.pdf` and reviewed `FinalYearProject.ipynb`.
- Scope: the four-model PyTorch classification benchmark, before the broader
  segmentation and sprinkler pipeline.
- The existing notebook documents cleaning and a stratified 80/20 train/test split;
  its execution paths reference Google Colab. Preserve it as the cleaning history.
- Located the local cleaned dataset at
  `Dataset/clean_dataset-20261001T114830Z-1-001/clean_dataset`.
- Confirmed the train and test directories expose the six class names specified
  by the guide. Image totals and duplicate checks await the user's Stage 1 run.
- Do not combine the separate `Dataset/Data` collection with this benchmark:
  its class folders differ from the six-class cleaned dataset.
- Added a read-only Stage 1 check and a standalone notebook. The script reports
  counts, SHA-256 file duplicate groups, overlap between splits, conflicting labels,
  and package versions. It neither deletes images nor changes the split.
- Local environment discovery found Python 3.13.6 and no installed `torch` module.
  PyTorch setup and runtime/GPU verification are pending; TensorFlow is not a
  substitute for the PyTorch requirement.
- Proposed validation split: derive it only from the existing training partition.
  Its fraction and seed have not been finalized or applied.
- No training configuration, benchmark measurements, or best-model claims exist yet.
- Next action: user runs Stage 1 and shares the output; review findings before
  introducing validation splitting or model code, as requested by the guide.

## 2026-10-02 — Autonomous execution authorized; updated guide adopted

The latest user instruction overrides the earlier stop-and-request-output workflow.
Read the replacement `docs/reference/problem and guide.pdf`: software only, no hardware actuation,
no growth-stage prediction, and no invented pesticide rules. The original guide
and pending-user-run notes above are historical and superseded.

### Executed preparation

- The complete audit decoded all 1,941 images: all RGB, 1,939 at 1600x1600 and
  two at 1600x1548; no unreadable files and no identical decoded pixels.
- Class totals: bacterial leaf blight 297, brown spot 320, healthy 348, leaf blast
  337, leaf scald 308, narrow brown spot 331. Imbalance ratio 1.1717.
- Deliberately retain the verified existing 389-image test membership, avoiding
  arbitrary holdout reassignment. Stratify 20% of the development pool for
  validation: 1,241 train / 311 validation / 389 test, seed 42.
- Save path/content hashes and split identity in a CSV manifest. No image source
  was modified. Pixel hashes do not rule out transformed near-duplicates or the
  same plant appearing in different photographs; plant IDs are unavailable.
- Installed official CPU torch 2.14.1 and torchvision 0.29.1 inside `.venv`.
  CUDA is unavailable. System has 16 logical CPUs and approximately 16.9 GB RAM;
  restrict PyTorch to four threads and image batches of 16 for this experiment.

### Classification experiment

- Four explicit IMAGENET1K_V1 checkpoints downloaded from the official PyTorch
  model server into the project cache.
- Frozen backbone, only final six-class linear layer trainable. Batch normalization
  and dropout held in evaluation mode during feature extraction. This is a
  frozen-feature baseline, not full-backbone fine-tuning.
- Shared full-image RGB resize to 224x224 and ImageNet normalization. Preserve
  leaf edges rather than cropping; accept the small aspect-ratio change for two
  rectangular images. Training-only original plus seeded augmented view: flips,
  90-degree rotation, brightness/contrast within 10%; no hue shift or random crop.
- Cross-entropy without class weights, AdamW, learning rate 0.001, weight decay
  0.0001, head batch size 32, max 40 epochs, patience eight on validation macro F1.
- Baseline results: MobileNetV3-Small validation accuracy 94.21%, macro F1 0.9431;
  EfficientNet-B0 95.82%, 0.9591; ResNet18 94.53%, 0.9458; ResNet50 97.11%, 0.9718.
- Actual six-class parameter counts: 1,524,006; 4,015,234; 11,179,590; 23,520,326.
  These differ from published 1,000-class counts because the final layer is replaced.
- Targeted tuning plan, defined before test access: rates 0.0003 and 0.003 on the
  top two validation architectures (ResNet50, EfficientNet-B0), all else fixed.
  Select by validation macro F1, record checkpoint/split hashes, then evaluate v1
  once on the holdout. Do not tune further against the resulting test scores.

### Software and verification

- Added CLI training, reporting, frozen selection, evaluation and inference.
- Added a Streamlit image-upload app with scores, model comparison and dataset views.
- Six tests pass: split isolation, deterministic preprocessing, six outputs and
  head-only training for all architectures, feature/full-model equivalence and
  checkpoint roundtrip, mask area arithmetic, and guarded quantity unit arithmetic.
- Streamlit AppTest renders without exceptions. Saved MobileNet weights correctly
  load and classify a real validation sample (bacterial leaf blight).
- Added source-linked cultural-management references for all five diseases.
  Chemical rules remain null pending current product-label and regional verification.
- No segmentation masks were found; documented an annotation protocol. The optional
  mask calculator explicitly measures a supplied annotation and does not predict it.

### Final v1 results and application check

- Completed all four extra learning-rate trials. Baseline ResNet50 remains the
  validation macro-F1 winner (0.971829). ResNet50 at 0.003 reaches 0.971371;
  EfficientNet-B0 at 0.0003 reaches 0.965578. These small differences are not evidence
  of statistical superiority. The selected ResNet50 is 94.4 MB and its measured
  CPU forward latency is 157.4 ms; EfficientNet-B0 is a smaller/faster alternative.
- Saved immutable selection identity before test access: checkpoint SHA-256
  `4f451129451e79367017cebb184b819f1c1b6a18657edc1eca88181cb15572be`.
- Evaluated the selected checkpoint once on 389 reserved images: accuracy
  **94.3445%**, macro precision **0.945570**, macro recall **0.944927**, macro F1
  **0.944281**, cross-entropy **0.180949**. 367 correct, 22 incorrect.
- Lowest test class recall: leaf blast, 57/68 = 83.82%. Six leaf-blast images were
  predicted as brown spot, and three as healthy. This is a material class-specific
  limitation despite the aggregate accuracy.
- Visually inspected the first 12 misclassified images. Several show small lesions
  occupying little of the frame; brown-spot/leaf-blast images share similar isolated
  leaf presentations. Some wrong predictions have high softmax scores. These are
  observations, not verified causes or grounds for relabelling the test images.
  No model or preprocessing was adjusted using these test errors.
- Saved confusion matrix, misclassified image paths, an example contact sheet,
  learning curves, all candidate scores, and `artifacts/PROJECT_REPORT.md`.
- Seven pipeline tests pass, including manifest-tamper detection. Streamlit AppTest covers initial rendering, default
  selected model, and a real validation-image upload through inference, source
  display and final-test metric rendering without exceptions.
- Started the local app at `http://127.0.0.1:8501`; health endpoint returned HTTP 200.
- Verified repeat calls to tuning/evaluation return the frozen selection and saved
  test results without further training or test inference.
- Executed all seven code cells in `RiceDiseaseBenchmark.ipynb` using the project
  interpreter and saved their measured outputs. After the session restart, restored
  the local app and rechecked its health endpoint (HTTP 200).
- Remaining project dependencies: reviewed pixel masks for segmentation and current
  formulation-specific regional labels/application rules for pesticide advice.
  The quantity helper is arithmetic infrastructure only; no real chemical rule is enabled.

### Recovery verification and explicit binary prediction

- Found the completed checkpoints and reports after the interrupted session;
  verified the selected checkpoint and split hashes through the evaluation command.
  It returned saved results without repeating test inference.
- CLI and app now explicitly return healthy/diseased alongside the disease class.
  The fixed rule maps the six-class argmax to healthy or diseased. No new threshold,
  separate binary training, or model reselection was performed.
- Added binary validation comparisons for all four models and binary final-test
  metrics derived from their saved confusion matrices. The selected ResNet50
  achieves 383/389 (98.46%) binary accuracy: 69 healthy true negatives, one false
  positive, 314 disease true positives, and five false negatives.
- MobileNetV3-Small has the highest binary validation accuracy (98.71%). The
  preserved v1 ResNet50 selection used six-class validation macro F1, not binary
  accuracy. These are different objectives; no selection was changed after test access.
- Eight tests pass. The Streamlit app renders without exceptions, and CLI inference
  using the default selected checkpoint correctly labels a real healthy validation
  image and returns status/has_disease fields.

### Upload recovery completed — 5 October 2026

- Finished the interrupted dependency installation in `.venv`; SciPy 1.17.1 is
  installed and `pip check` reports no broken requirements.
- Verified the saved Analyze leaf button and telemetry configuration changes.
  All 10 tests pass, covering upload/replace/clear, model changes, reruns, invalid
  images, and the existing classification pipeline.
- Started the local app at `http://127.0.0.1:8501` with the project interpreter.
  A real Chromium session successfully uploaded `.JPG` and `.jpeg` files, clicked
  Analyze leaf, rendered the expected healthy validation prediction, and downloaded
  matching analysis JSON. An invalid image displayed an error; a subsequent valid
  upload recovered successfully. Upload requests returned HTTP 204, with no browser
  JavaScript errors or server errors observed.
- Updated `UPLOAD_FIX_PROGRESS.md` with completed checks and restart instructions.
  Browser check results and a screenshot are stored in `.cache/`.


## 2026-10-07: expanded training comparison and single-model app

- Kept all four original baselines unchanged. Completed 48 additional trials (12 recipes per architecture), comparing AdamW/Adam/momentum SGD, cross-entropy/label smoothing/focal loss, and learning-rate/batch-size/weight-decay variants.
- Selected ResNet50 + AdamW + label-smoothed cross-entropy (0.1) using validation macro F1: 97.43% accuracy, 0.9749 F1. Learning rate 0.001, weight decay 0.0001, batch size 32; checkpoint epoch 22 of 30 run.
- Exported one deployed model to `artifacts/final_model/hyperparameter_search_v2/model.pt`; archived the old selection. Preserved original research checkpoints and historical test results.
- The winner scored 94.60% accuracy and 0.9472 macro F1 on the previously evaluated holdout (368/389 correct). The holdout was not used to rank trials; this is not fresh independent evaluation.
- Removed the Streamlit model dropdown and baseline fallback. Only the selected checkpoint can run; metrics must match its checksum and split.
- Updated notebook sources, README, PROJECT_STEP_BY_STEP and generated PROJECT_REPORT. All 20 tests passed.
- Search plan and ranked results: `artifacts/experiments/hyperparameter_search_v2/`.

## 2026-10-08: leaf and lesion candidate instance segmentation

- User requested both leaf and lesion instances on the cleaned dataset. Added
  `rice_disease.instance_segmentation`, a reproducible colour/morphology and
  connected-component baseline. No segmentation network was trained; pixel/instance
  annotations are still absent.
- Inspected two 12-image training-only pilots and froze the v2 parameters before
  the full export. Disease class labels do not determine masks. Used SciPy 1.17.1,
  already installed, and declared it in requirements.
- Processed all 1,941 audited source images with zero execution failures in 906
  seconds. Preserved 1,241 train / 311 validation / 389 test assignments and source
  checksums. Exported 1,967 leaf components and 51,818 lesion candidates.
- Saved native-size uint16 leaf/lesion ID PNGs, overlays, parent-leaf links,
  bounding boxes, pixel counts, per-image metadata, a CSV and summary under
  `artifacts/instance_segmentation/`. All masks are explicitly unreviewed pseudo-masks.
- The baseline produces candidates on 347/348 healthy-labelled images and no
  candidates on four diseased-labelled images. This is substantial false-positive
  behaviour; outputs need correction before use as training annotations. No Dice,
  IoU, mask AP, disease-area accuracy or severity claim is made. Classifier and app
  behaviour remain unchanged.
- The full 24-test regression suite passed. The expanded five-test segmentation
  suite also passed (25 unique tests across both runs); dependency and whitespace
  checks passed. Added `scripts/validate_instance_masks.py` for exhaustive saved-file
  verification and report generation. Documentation: `docs/INSTANCE_SEGMENTATION.md`.
- Exhaustive saved-file verification passed for all 1,941 images, covering the
  entire manifest: dimensions, uint16 IDs, areas, bounds, lesion containment,
  source identity, overlays and split membership. Results are in
  `artifacts/instance_segmentation/validation.json` and `REPORT.md`.

## 2026-10-08: reviewed-mask U-Net pipeline and 100 mL tank configuration

- User requested U-Net to estimate affected rice-leaf percentage, chose reviewed
  masks rather than training on heuristic labels, and specified Tripura, India,
  with a 100 mL spray tank and disease-dependent product selection.
- Added a 1,964,131-parameter three-class U-Net (background / unaffected leaf /
  affected leaf), paired geometry, ignored letterbox padding, CE plus leaf/lesion
  Dice loss, validation selection, early stopping, provenance and checkpoint loading.
- Prepared 1,941 editable semantic annotation drafts and `annotations.csv` under
  `artifacts/segmentation_annotations/`; all remain unreviewed. The real training
  command correctly refused zero reviewed train/validation masks and produced no
  rice checkpoint. No rice U-Net accuracy has been claimed.
- Added separate original-resolution test evaluation and image prediction commands.
  Area is 100 times affected pixels divided by all leaf pixels, excludes background,
  and is null when no leaf is detected. Metrics include leaf/lesion Dice/IoU,
  affected-area MAE, empty-lesion false positives and missing-leaf cases.
- Connected the Streamlit app to `artifacts/unet/model.pt` after reviewed-mask
  training. Tested the checkpoint display with a synthetic model fixture; no
  synthetic or random checkpoint was deployed.
- Configured India/Tripura/rice/100 mL (0.1 L) in `configs/application.json`.
  Added crop/disease matching and tank conversion to the verified-label quantity
  helper. Affected percentage never scales dosage. No chemical rule was enabled:
  official references found do not establish this user's product/crop-stage use.
- Full 33-test suite passed. The expanded nine-test U-Net suite and separate app
  test also passed, covering 35 unique tests. A synthetic training/checkpoint/
  evaluation/inference run confirms executable behavior, not agricultural accuracy.
- Guide: `docs/UNET_SEGMENTATION.md`; label research:
  `docs/PESTICIDE_TRIPURA_REVIEW.md`; readiness: `artifacts/unet_readiness.json`.

## 9 October 2026 - pixel-mask audit and single-photo product references

- Rechecked all 1,941 semantic PNGs against source hashes, native dimensions,
  allowed pixel labels and the split manifest. All passed file integrity; none
  is reviewed. Split counts: 1,241 train, 311 validation, 389 test. The audit found
  affected pixels in 347 of 348 healthy-class drafts and no affected pixels in
  four leaf-blast drafts. These are review flags, not measured disease accuracy.
- Added repeatable `python -m rice_disease.mask_audit`; saved the live results in
  `artifacts/pixel_mask_audit.json` and refreshed `artifacts/unet_readiness.json`.
  The real training command refused zero reviewed train/validation masks. There
  is still no trained rice U-Net checkpoint and no rice segmentation score.
- Extended the existing U-Net prediction export with verified-label quantity
  arithmetic, retaining the rate, unit and calculation basis. Added source-class
  and shared-reviewed-mask checks and rejected nonfinite quantity results.
- User clarified that a single plant photo supplies the disease for product
  selection, without field-area input. The app and classification CLI now match
  blast/brown spot to a conditional PPQS formulation reference. Water volume
  controls dilution-reference arithmetic; required spray quantity for one plant
  remains unknown. Other diseases explicitly report a missing catalog label;
  healthy predictions suggest no product. No application-authorized rule enabled.
- Classification CLI automatically uses the trained U-Net when present; otherwise
  affected area remains null with an explicit reason. Saved a successful CLI
  check using one existing validation photo in `artifacts/single_photo_cli_check.json`.
- Validation: 24 relevant unique tests passed (12 U-Net/audit/quantity, eight
  pipeline, two upload app, two segmentation/product-reference app tests).
  Synthetic training tested checkpoint, evaluation and prediction behavior only.
  CLI help and whitespace checks passed. No annotation review flags were changed.

## 9-10 October 2026 - consolidate instance exports and repair broad false regions

- Inspected the three instance folders: the active version 2 export contained
  1,941 images; each older pilot contained 12 (two per class). Rebuilt all images
  using version 3, then promoted the validated results on 10 October. Preserved
  both pilots, the old full export, semantic drafts and audit/readiness snapshots
  under `artifacts/archive/segmentation_before_v3_20261009/`. Exactly one active
  instance export remains at `artifacts/instance_segmentation/`.
- Removed the unconditional pale-pixel lesion rule, tightened brown-tissue and
  foreground colour criteria, and limited pale extensions to brown-seed neighbours.
  Replaced random whole-leaf overlay colours with cyan boundaries and fixed red
  candidate regions. Large connected regions and complex backgrounds are flagged
  for review, not discarded through an arbitrary area cap. Settings were inspected
  on training examples and frozen before the complete export.
- Rebuilt all 1,941 instance masks with zero processing failures: 2,033 leaf
  components and 27,735 lesion candidates. All exported files passed validation.
  Regenerated all 1,941 semantic drafts from the same version; the complete semantic
  audit found zero integrity errors and zero reviewed masks. Updated provenance
  paths and readiness. Source photos and split membership were preserved.
- Candidate coverage above 50% decreased from 259 images to 72. Mean candidate
  coverage across healthy-labelled images decreased from 14.225% to 0.787%, but
  199 healthy-labelled images retain some candidates. Diseased-labelled images
  without candidates increased from four to 24. Lower coverage is not proof of
  accuracy: residual shadows, pale-damage misses and uncertain boundaries require
  manual correction. No claim that every semantic error has been fixed is made.
- Added `index.html` for searchable original/overlay pairs, `review_queue.csv`,
  `repair_comparison.json`, and a training-only before/after `repair_preview.jpg`.
  Verified every gallery source/overlay/metadata link and every semantic-mask path.
  `artifacts/SEGMENTATION_README.md` identifies active versus archived outputs.
- Validation: eight instance regression tests and twelve U-Net/audit/quantity tests
  passed. New regressions cover warm paper, pale highlights, shadows, and retaining
  genuinely broad brown candidates. Compilation and whitespace checks passed.
  No rice U-Net was trained from these unreviewed drafts.

## 10 October 2026 - boundary corrections and trained U-Net pilot

- Visually inspected 60 sources across all six classes, using only existing
  train/validation split members. Corrected and accepted 56 semantic masks
  (44 train, 12 validation); deferred IDs 14, 15, 35 and 41 for blur or ambiguous
  tissue boundaries. Removed cast shadows/highlight false positives, corrected
  outlines and pale lesion centres, and retained genuinely broad visible damage.
  These are approximate AI visual annotations reviewed at 512-pixel display
  resolution, not expert ground truth. The remaining 1,885 masks stay unreviewed.
- Added reproducible image-specific edits, source/mask hashes, pre-edit backups
  and an original/corrected gallery in `artifacts/boundary_review_20261010/`.
  Updated the complete instance gallery to link current semantic masks and review
  provenance. All 1,941 gallery records and all 56 corrected-mask/backups were
  checked. The full semantic audit passed all 1,941 files with no integrity errors.
- Added optional resized-input caching and explicit `ai_visual_reviewed` metadata
  throughout training, checkpoint loading, prediction and the app. Regression
  checks ensure augmentation leaves cached originals intact and AI provenance
  cannot silently become an ordinary reviewed-model claim.
- Initial training stopped after 11 epochs with weak lesion learning; retained it
  under `artifacts/archive/unet_initial_20261010/`. The second run used the same
  fixed references, 256-pixel input, width 16, AdamW at 0.001, a 24-epoch minimum,
  patience 12 and two CPU threads. Completed all 40 epochs and selected epoch 32.
  Added a regression test for the minimum-epoch early-stopping behavior.
- Activated `artifacts/unet/model.pt` with SHA-256
  `aed0f3e6a093b4bd788cc66b2ab2d4d221f2a90bb189c81e638294814f4972f8`.
  Model-resolution validation: leaf Dice 0.968709, lesion Dice 0.718374,
  area MAE 2.883916 percentage points. Original-resolution validation: leaf Dice
  0.968830, lesion Dice 0.705703, area MAE 3.149988 percentage points.
  These measure agreement with 12 approximate AI references, not expert accuracy.
  No test images were used in training, selection or prediction evaluation.
- Reviewed all 12 validation prediction overlays. Small narrow brown spots and
  pale scald damage are still underdetected; documented per-image/class results
  and concrete errors in `artifacts/unet/REPORT.md`. Neither of the two healthy
  validation references produced lesion pixels. The pilot is not field validated.
- Validation: 27 relevant unique unit/app tests passed (15 U-Net, eight instance,
  two upload app, two segmentation app). Actual checkpoint inference and a live
  Streamlit test both returned 8.91% for the same validation source; the experimental
  notice and downloadable result were verified. Source images/splits were preserved.

## 10 October 2026 - folder cleanup, segmentation guide and result display

- Removed 2,353 disposable/generated files (261.81 MiB): scratch caches, bytecode,
  derived classifier image/features, old console logs, a temporary preview and
  the editor scratch file verified identical to `rice_disease/models.py`.
  Kept imported/cleaned photos, reviewed masks, correction backups, trained models,
  experiment records, the installed environment and offline pretrained weights.
- Moved project overview, workflow notes and this log into `docs/`; updated README
  navigation and added `docs/FOLDER_GUIDE.md`. Updated the overview to the selected
  classifier's recorded 368/389 six-class and 381/389 binary results.
- Added `docs/SEGMENTATION_STEP_BY_STEP.md`: source/split preservation, candidate
  generation, conversion, review, audit, configuration, training, validation,
  separate test evaluation and single-photo inference. It distinguishes instance
  IDs from semantic classes and documents the absence of reviewed test masks.
- Restored visible model scores and all six class percentages in Streamlit,
  retaining the single bounded segmentation image and affected-area prediction.
- Verification: all 45 automated tests passed; `pip check` found no broken
  requirements. The complete semantic audit passed 1,941 masks with no missing
  annotations or integrity errors. Instance validation passed 1,941 images,
  2,033 leaf components and 27,735 lesion candidates. Active model hashes were
  unchanged; selected-model references, Markdown links and 8,103 active gallery
  file references resolved. A synthetic display check confirmed visible disease
  percentages and the unchanged segmentation result layout. No models retrained.
