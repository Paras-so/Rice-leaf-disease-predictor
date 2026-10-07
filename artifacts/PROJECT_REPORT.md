# Software/ML project: executed results

## Dataset

1,941 readable RGB images across six classes. No exact-file or identical-pixel duplicates. Split: 1,241 train, 311 validation, 389 test (seed 42). Source images are preserved.

## Final classification model

Selected **resnet50**, run **hyperparameter_search_v2_adamw_smooth**, using validation macro F1 across 52 candidates. Only the final linear layer was trained. Full-backbone fine-tuning has not been performed.

Test data is not read during search. The holdout was previously evaluated for v1.

Final training configuration: `{"augmentation": "One original view and one seeded view: horizontal/vertical flips, 90-degree rotations, brightness/contrast within 10%. No crop or hue shift.", "batch_size": 32, "cpu_threads": 4, "epochs": 40, "feature_batch_size": 16, "image_size": 224, "label_smoothing": 0.1, "learning_rate": 0.001, "loss": "LabelSmoothedCrossEntropy", "mean": [0.485, 0.456, 0.406], "models": ["mobilenet_v3_small", "efficientnet_b0", "resnet18", "resnet50"], "optimizer": "AdamW", "patience": 8, "seed": 42, "selection_metric": "validation_macro_f1", "std": [0.229, 0.224, 0.225], "training_mode": "frozen_backbone_linear_probe", "training_views": 2, "weight_decay": 0.0001, "weights": "IMAGENET1K_V1", "workers": 0}`

Validation accuracy: **97.43%**; macro F1: **0.9749**.

Final test accuracy: **94.60%**; macro precision: **0.9503**; macro recall: **0.9477**; macro F1: **0.9472**. Misclassified 21 of 389 images.

| Class | Test precision | Test recall | Test F1 | Images |
|---|---:|---:|---:|---:|
| bacterial_leaf_blight | 1.0000 | 1.0000 | 1.0000 | 59 |
| brown_spot | 0.8611 | 0.9688 | 0.9118 | 64 |
| healthy | 0.9079 | 0.9857 | 0.9452 | 70 |
| leaf_blast | 0.9492 | 0.8235 | 0.8819 | 68 |
| leaf_scald | 0.9839 | 0.9839 | 0.9839 | 62 |
| narrow_brown_spot | 1.0000 | 0.9242 | 0.9606 | 66 |

## Validation-only model development

| Run | Architecture | Validation macro F1 |
|---|---|---:|
| hyperparameter_search_v2_adamw_smooth | resnet50 | 0.9749 |
| hyperparameter_search_v2_sgd_smooth | resnet50 | 0.9719 |
| frozen_baseline | resnet50 | 0.9718 |
| hyperparameter_search_v2_adam_ce | resnet50 | 0.9718 |
| hyperparameter_search_v2_adamw_ce_decay | resnet50 | 0.9718 |
| hyperparameter_search_v2_adam_smooth | resnet50 | 0.9718 |
| hyperparameter_search_v2_sgd_ce | resnet50 | 0.9717 |
| hyperparameter_search_v2_adamw_ce_batch64 | resnet50 | 0.9717 |
| hyperparameter_search_v2_adamw_ce_lr_high | resnet50 | 0.9714 |
| hyperparameter_search_v2_sgd_focal | efficientnet_b0 | 0.9714 |
| hyperparameter_search_v2_adamw_ce_lr_low | efficientnet_b0 | 0.9656 |
| hyperparameter_search_v2_adamw_focal | resnet50 | 0.9623 |
| hyperparameter_search_v2_adam_focal | resnet50 | 0.9623 |
| hyperparameter_search_v2_adamw_focal | efficientnet_b0 | 0.9621 |
| hyperparameter_search_v2_adam_focal | efficientnet_b0 | 0.9621 |
| hyperparameter_search_v2_adamw_ce_lr_high | mobilenet_v3_small | 0.9621 |
| hyperparameter_search_v2_adamw_smooth | mobilenet_v3_small | 0.9621 |
| hyperparameter_search_v2_adam_smooth | mobilenet_v3_small | 0.9621 |
| hyperparameter_search_v2_adamw_smooth | efficientnet_b0 | 0.9618 |
| hyperparameter_search_v2_adam_smooth | efficientnet_b0 | 0.9618 |
| hyperparameter_search_v2_sgd_focal | resnet50 | 0.9616 |
| hyperparameter_search_v2_adamw_ce_batch64 | efficientnet_b0 | 0.9592 |
| frozen_baseline | efficientnet_b0 | 0.9591 |
| hyperparameter_search_v2_adam_ce | efficientnet_b0 | 0.9591 |
| hyperparameter_search_v2_adamw_ce_decay | efficientnet_b0 | 0.9591 |
| hyperparameter_search_v2_adamw_ce_lr_high | efficientnet_b0 | 0.9590 |
| hyperparameter_search_v2_sgd_smooth | mobilenet_v3_small | 0.9588 |
| hyperparameter_search_v2_sgd_ce | efficientnet_b0 | 0.9587 |
| hyperparameter_search_v2_sgd_smooth | efficientnet_b0 | 0.9585 |
| hyperparameter_search_v2_adamw_ce_lr_low | resnet50 | 0.9564 |
| hyperparameter_search_v2_sgd_ce | mobilenet_v3_small | 0.9557 |
| hyperparameter_search_v2_sgd_focal | mobilenet_v3_small | 0.9528 |
| hyperparameter_search_v2_adamw_focal | mobilenet_v3_small | 0.9526 |
| hyperparameter_search_v2_adam_focal | mobilenet_v3_small | 0.9526 |
| hyperparameter_search_v2_sgd_ce | resnet18 | 0.9487 |
| hyperparameter_search_v2_sgd_focal | resnet18 | 0.9461 |
| frozen_baseline | resnet18 | 0.9458 |
| hyperparameter_search_v2_adamw_ce_decay | resnet18 | 0.9458 |
| hyperparameter_search_v2_adam_ce | resnet18 | 0.9456 |
| hyperparameter_search_v2_adamw_focal | resnet18 | 0.9453 |
| hyperparameter_search_v2_adamw_ce_lr_high | resnet18 | 0.9453 |
| hyperparameter_search_v2_adam_focal | resnet18 | 0.9452 |
| frozen_baseline | mobilenet_v3_small | 0.9431 |
| hyperparameter_search_v2_adam_ce | mobilenet_v3_small | 0.9431 |
| hyperparameter_search_v2_adamw_ce_decay | mobilenet_v3_small | 0.9431 |
| hyperparameter_search_v2_adamw_smooth | resnet18 | 0.9420 |
| hyperparameter_search_v2_adam_smooth | resnet18 | 0.9420 |
| hyperparameter_search_v2_adamw_ce_lr_low | mobilenet_v3_small | 0.9397 |
| hyperparameter_search_v2_adamw_ce_batch64 | mobilenet_v3_small | 0.9397 |
| hyperparameter_search_v2_sgd_smooth | resnet18 | 0.9362 |
| hyperparameter_search_v2_adamw_ce_batch64 | resnet18 | 0.9256 |
| hyperparameter_search_v2_adamw_ce_lr_low | resnet18 | 0.9187 |

The selected checkpoint is 94.4 MB, with 23,520,326 parameters (12,294 trained). Measured CPU batch-one forward latency is 153.1 ms, excluding preprocessing. Training accuracy exceeds validation by 1.44%. The validation ranking determines the selected candidate. The modest differences from tuned alternatives are not statistically established.

Model size alone does not explain the results: EfficientNet-B0 outperformed the larger ResNet18, while ResNet50 performed best in this particular frozen-feature experiment. Additional seeds, field-disjoint data and a separate fine-tuning study are needed to establish robust conclusions.

## Error analysis

- leaf_blast predicted as brown_spot: 7 images.
- leaf_blast predicted as healthy: 5 images.
- narrow_brown_spot predicted as leaf_blast: 3 images.
- brown_spot predicted as healthy: 2 images.
- narrow_brown_spot predicted as leaf_scald: 1 image.
- narrow_brown_spot predicted as brown_spot: 1 image.
- leaf_scald predicted as brown_spot: 1 image.
- healthy predicted as brown_spot: 1 image.

See `evaluations/hyperparameter_search_v2/confusion_matrix.png`, `misclassified.json`, and `misclassified_examples.png`. Error inspection does not authorize tuning against this test set.

## Software delivered

Reproducible CLI pipeline, saved PyTorch checkpoints and configuration, local Streamlit image upload, class scores, a single final classifier, dataset views and sourced cultural-management information. The optional mask calculator measures a supplied annotation and excludes image background.

## Unfinished components and limitations

- Segmentation: no labelled pixel masks are supplied. No segmentation model, mask prediction, disease severity grade, or image-based affected-area claim has been made.
- Treatment: cultural-management references are present, but current regional product labels and application rules remain unverified. No automatic pesticide, spray timing or quantity recommendation is enabled.
- Generalization: one split/seed; no plant/field identity grouping, unknown-class rejection or confidence calibration. Image-level holdout results are not field validation.
- The final v1 test has now been consumed. Further tuning needs an independent holdout or a documented nested protocol.
- No hardware actuation or growth-stage prediction is included.

Next required data: reviewed leaf/lesion pixel masks and formulation-specific current agricultural label sources. See `docs/SEGMENTATION_PLAN.md` and `docs/KNOWLEDGE_BASE.md`.

## Healthy versus diseased: final test

Derived from the saved six-class confusion matrix; no training or test inference repeated. The predicted class maps to healthy or diseased without threshold tuning.

Accuracy: **97.94%**; disease precision: **0.9968**; disease recall: **0.9781**; disease F1: **0.9873**.

| Actual / Predicted | Healthy | Diseased |
|---|---:|---:|
| Healthy | 69 | 1 |
| Diseased | 7 | 312 |
