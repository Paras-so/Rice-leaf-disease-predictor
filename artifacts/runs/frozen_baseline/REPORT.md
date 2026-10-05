# Measured classification benchmark

Frozen pretrained ImageNet backbones with a learned six-class final linear layer. This measures transfer-feature quality; it is not an end-to-end fine-tuning comparison.

All models use the same split, 224×224 full-image resize, normalization, two training views, AdamW configuration, batch size and stopping criterion. The 389-image test holdout has not been scored by this benchmark.

| Model | Parameters | Trainable | Train accuracy | Validation accuracy | Macro F1 | Training seconds | Forward ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| resnet50 | 23,520,326 | 12,294 | 99.033% | 97.106% | 0.9718 | 401.6 | 157.36 |
| efficientnet_b0 | 4,015,234 | 7,686 | 98.388% | 95.820% | 0.9591 | 116.3 | 52.23 |
| resnet18 | 11,179,590 | 3,078 | 97.744% | 94.534% | 0.9458 | 166.4 | 60.24 |
| mobilenet_v3_small | 1,524,006 | 6,150 | 97.502% | 94.212% | 0.9431 | 164.5 | 18.87 |

Times include feature extraction plus head training; pretrained downloads are excluded. Forward latency uses one synthetic input and excludes decoding/preprocessing. CPU load and caching affect timing.

## Interpretation

The highest validation macro F1 is **resnet50**, at **0.9718**. This is a candidate for further development, not evidence of statistical superiority or field reliability.

One split and one seed cannot establish a capacity-performance law. Architecture and ImageNet training recipes also differ. The classifier has no unknown-class or non-rice rejection model, and softmax scores are uncalibrated.

## Per-model observations

- resnet50: best epoch 27; train–validation accuracy gap 1.9%. No large accuracy gap; inspect loss curves and per-class validation performance.
- efficientnet_b0: best epoch 12; train–validation accuracy gap 2.6%. No large accuracy gap; inspect loss curves and per-class validation performance.
- resnet18: best epoch 29; train–validation accuracy gap 3.2%. No large accuracy gap; inspect loss curves and per-class validation performance.
- mobilenet_v3_small: best epoch 16; train–validation accuracy gap 3.3%. No large accuracy gap; inspect loss curves and per-class validation performance.

## Sources and experimental choices

- [TorchVision pretrained model API](https://docs.pytorch.org/vision/stable/models.html)
- [PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- Explicit IMAGENET1K_V1 weights are pinned. Using a common full-image resize instead of the weights' default center crop preserves leaf-edge symptoms; it is a documented benchmark choice.
- Segmentation and severity cannot be inferred from these classification labels; masks are required.

## Healthy versus diseased: validation

Fixed mapping of the six-class argmax: healthy stays healthy; every disease class becomes diseased. No separate binary model or threshold tuning.

| Model | Binary accuracy | Disease precision | Disease recall | Disease F1 |
|---|---:|---:|---:|---:|
| resnet50 | 98.39% | 1.0000 | 0.9804 | 0.9901 |
| efficientnet_b0 | 98.07% | 0.9960 | 0.9804 | 0.9881 |
| resnet18 | 98.07% | 0.9921 | 0.9843 | 0.9882 |
| mobilenet_v3_small | 98.71% | 1.0000 | 0.9843 | 0.9921 |
