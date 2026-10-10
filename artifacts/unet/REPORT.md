# Experimental rice U-Net pilot

Trained for 40 epochs; selected epoch **32** by mean validation leaf/lesion Dice.
Uses **44 training and 12 validation images** across all six classes. No test images were used.

**Labels are approximate AI-reviewed boundaries, not expert ground truth.** Sixty photos were inspected; 56 corrections were accepted and four were deferred. There are still **1,885 unreviewed masks**, excluded from training.

## Validation against AI-reviewed references

| Resolution | Leaf Dice | Lesion Dice | Leaf IoU | Lesion IoU | Area MAE (percentage points) |
|---|---:|---:|---:|---:|---:|
| 256-pixel model input | 0.9687 | 0.7184 | 0.9393 | 0.5605 | 2.884 |
| Original image dimensions | 0.9688 | 0.7057 | 0.9395 | 0.5452 | 3.150 |

All 12 validation images had a predicted leaf. Neither of the two healthy reference images had predicted lesion pixels. These small counts do not establish field accuracy.

### Per-class results at original dimensions

| Class | Images | Leaf Dice | Lesion Dice | Area MAE (percentage points) |
|---|---:|---:|---:|---:|
| bacterial_leaf_blight | 2 | 0.9731 | 0.8234 | 3.692 |
| brown_spot | 2 | 0.9828 | 0.6166 | 1.553 |
| healthy | 2 | 0.9520 | undefined (both empty) | 0.000 |
| leaf_blast | 2 | 0.9650 | 0.6886 | 0.808 |
| leaf_scald | 2 | 0.9144 | 0.5744 | 10.873 |
| narrow_brown_spot | 2 | 0.9955 | 0.1476 | 1.974 |

## Observed limitations

The reviewed validation overlays still show missed narrow brown spots and underestimated pale scald damage. Scald reference percentages of 23.21% and 26.45% were predicted as 11.34% and 16.56%. The two narrow-brown-spot references around 2.1-2.2% were predicted as 0.33% and 0.02%. This is a working experimental baseline, not a claim that every segmentation error is fixed.

Reference boundaries were reviewed at a 512-pixel display resolution. They require independent agricultural review. The other 1,885 drafts remain unreviewed; there are no expert-reviewed test masks and no test accuracy claim.

## Files

- [Selected model](model.pt)
- [Corrected annotation gallery](../boundary_review_20261010/index.html)
- [Training and area plot](validation_review/training_and_area.png)
- [Validation overlay page 1](validation_review/comparison_0.jpg)
- [Validation overlay page 2](validation_review/comparison_1.jpg)
- [Validation overlay page 3](validation_review/comparison_2.jpg)
- [Full per-image metrics](validation_review/metrics.json)
- [Training manifest with mask hashes](training_manifest.json)
- [Single-photo CLI result](single_photo_check.json)

The first run stopped at epoch 11 before useful lesion learning; its selected checkpoint and diagnostics are preserved in `../archive/unet_initial_20261010/`. The second run used a 24-epoch minimum, patience 12, two CPU threads, and cached resized inputs. The active app and CLI use this selected checkpoint.

Affected percentage is 100 times predicted affected pixels divided by all predicted leaf pixels. It does not establish the required pesticide quantity for a plant.
