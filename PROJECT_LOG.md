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
