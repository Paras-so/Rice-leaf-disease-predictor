# Software/ML project: executed results

## Dataset

1,941 readable RGB images across six classes. No exact-file or identical-pixel duplicates. Split: 1,241 train, 311 validation, 389 test (seed 42). Source images are preserved.

## Classification v1

Selected **resnet50**, run **frozen_baseline**, using validation macro F1 before test access. Four ImageNet backbones were benchmarked; two extra head learning rates were tried on the top two architectures. Only the final linear layer was trained. Full-backbone fine-tuning has not been performed.

Validation accuracy: **97.11%**; macro F1: **0.9718**.

Final test accuracy: **94.34%**; macro precision: **0.9456**; macro recall: **0.9449**; macro F1: **0.9443**. Misclassified 22 of 389 images.

| Class | Test precision | Test recall | Test F1 | Images |
|---|---:|---:|---:|---:|
| bacterial_leaf_blight | 1.0000 | 1.0000 | 1.0000 | 59 |
| brown_spot | 0.8696 | 0.9375 | 0.9023 | 64 |
| healthy | 0.9324 | 0.9857 | 0.9583 | 70 |
| leaf_blast | 0.9344 | 0.8382 | 0.8837 | 68 |
| leaf_scald | 0.9531 | 0.9839 | 0.9683 | 62 |
| narrow_brown_spot | 0.9839 | 0.9242 | 0.9531 | 66 |

## Validation-only model development

| Run | Architecture | Validation macro F1 |
|---|---|---:|
| frozen_baseline | resnet50 | 0.9718 |
| head_lr_0.003 | resnet50 | 0.9714 |
| head_lr_0.0003 | efficientnet_b0 | 0.9656 |
| frozen_baseline | efficientnet_b0 | 0.9591 |
| head_lr_0.003 | efficientnet_b0 | 0.9590 |
| head_lr_0.0003 | resnet50 | 0.9564 |
| frozen_baseline | resnet18 | 0.9458 |
| frozen_baseline | mobilenet_v3_small | 0.9431 |

The selected checkpoint is 94.4 MB, with 23,520,326 parameters (12,294 trained). Measured CPU batch-one forward latency is 157.4 ms, excluding preprocessing. Training accuracy exceeds validation by 1.93%. The validation score and per-class results favor this candidate; EfficientNet-B0 is a smaller/faster alternative. The modest differences from tuned alternatives are not statistically established.

Model size alone does not explain the results: EfficientNet-B0 outperformed the larger ResNet18, while ResNet50 performed best in this particular frozen-feature experiment. Additional seeds, field-disjoint data and a separate fine-tuning study are needed to establish robust conclusions.

## Error analysis

- leaf_blast predicted as brown_spot: 6 images.
- leaf_blast predicted as healthy: 3 images.
- narrow_brown_spot predicted as leaf_scald: 2 images.
- narrow_brown_spot predicted as leaf_blast: 2 images.
- brown_spot predicted as leaf_blast: 2 images.
- brown_spot predicted as healthy: 2 images.
- narrow_brown_spot predicted as brown_spot: 1 image.
- leaf_scald predicted as brown_spot: 1 image.
- leaf_blast predicted as narrow_brown_spot: 1 image.
- leaf_blast predicted as leaf_scald: 1 image.
- healthy predicted as brown_spot: 1 image.

See `final_evaluation/confusion_matrix.png`, `misclassified.json`, and `misclassified_examples.png`. Error inspection does not authorize tuning against this test set.

## Software delivered

Reproducible CLI pipeline, saved PyTorch checkpoints and configuration, local Streamlit image upload, class scores, model comparison, dataset views and sourced cultural-management information. The optional mask calculator measures a supplied annotation and excludes image background.

## Unfinished components and limitations

- Segmentation: no labelled pixel masks are supplied. No segmentation model, mask prediction, disease severity grade, or image-based affected-area claim has been made.
- Treatment: cultural-management references are present, but current regional product labels and application rules remain unverified. No automatic pesticide, spray timing or quantity recommendation is enabled.
- Generalization: one split/seed; no plant/field identity grouping, unknown-class rejection or confidence calibration. Image-level holdout results are not field validation.
- The final v1 test has now been consumed. Further tuning needs an independent holdout or a documented nested protocol.
- No hardware actuation or growth-stage prediction is included.

Next required data: reviewed leaf/lesion pixel masks and formulation-specific current agricultural label sources. See `docs/SEGMENTATION_PLAN.md` and `docs/KNOWLEDGE_BASE.md`.

## Healthy versus diseased: final test

Derived from the saved six-class confusion matrix; no training or test inference repeated. The predicted class maps to healthy or diseased without threshold tuning.

Accuracy: **98.46%**; disease precision: **0.9968**; disease recall: **0.9843**; disease F1: **0.9905**.

| Actual / Predicted | Healthy | Diseased |
|---|---:|---:|
| Healthy | 69 | 1 |
| Diseased | 5 | 314 |
