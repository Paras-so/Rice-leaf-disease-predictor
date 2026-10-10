# Cleaned dataset: leaf and lesion candidate instances

**Status: unreviewed pseudo-masks from a colour/morphology baseline.**

Processed **1,941 images** with **0 processing failures**.
Exported **1,967 leaf components** and **51,818 lesion candidates**.
These counts do not establish object-count or disease-segmentation accuracy.

[View training-sample overlays](preview.jpg) | [Per-image results](instances.csv) | [Full summary](summary.json) | [Output verification](validation.json)

## Counts

| Image class | Images | Leaf components | Lesion candidates |
|---|---:|---:|---:|
| bacterial_leaf_blight | 297 | 301 | 5698 |
| brown_spot | 320 | 325 | 10935 |
| healthy | 348 | 348 | 6620 |
| leaf_blast | 337 | 340 | 9020 |
| leaf_scald | 308 | 322 | 4929 |
| narrow_brown_spot | 331 | 331 | 14616 |

Split membership: train=1241, validation=311, test=389.

## Review flags

| Flag | Images |
|---|---:|
| requires_human_review | 1941 |
| multiple_leaf_components | 22 |
| high_candidate_coverage | 259 |
| high_foreground_coverage | 11 |
| candidate_on_healthy_image | 347 |
| no_candidate_on_diseased_image | 4 |

Flags can overlap. All masks require review, including images with no additional flags.

## Interpretation

- Unreviewed algorithm-generated candidates, not ground-truth annotations.
- Colour cannot distinguish disease from aging, shadows, lighting or nutrient stress.
- Touching/overlapping leaves or lesions can merge; fragmented objects can split.
- Small lesions may be lost at processing resolution; restored edges are approximate.
- Pale necrotic tissue can be missed and coloured background can be included.
- No trained segmentation model, IoU, Dice, mask AP or severity estimate is produced.

The saved-file check passed for mask dimensions, integer IDs, pixel areas, parent-leaf containment, source identity, overlays and split membership. It does not evaluate visual correctness.

Leaf and lesion masks are separate uint16 PNGs: zero is background; positive values identify individual components within that image. Lesion metadata stores the parent leaf ID. Instance IDs are not semantic class values and must not be passed to the annotation-area helper.

Source images, the classification model and application were preserved. Review and correct these masks before using them as training labels.
