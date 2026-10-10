# Leaf and lesion candidate instance segmentation

Run from the project directory:

```powershell
.\.venv\Scripts\python.exe -u -m rice_disease.instance_segmentation
.\.venv\Scripts\python.exe scripts/validate_instance_masks.py
.\.venv\Scripts\python.exe scripts/build_segmentation_review.py
```

The input is the audited cleaned dataset located by `artifacts/split_summary.json`.
The script verifies the split manifest and every input file checksum. It preserves
the existing split assignment for all derived masks. It does not change source
images, train classifiers, or score the held-out test set.

**Current active dataset:** `artifacts/instance_segmentation/`. Open `index.html`
there for all images, with filename/class/split/flag filters and original/overlay
pairs. The two former 12-image pilot folders and superseded full export are
preserved under `artifacts/archive/segmentation_before_v3_20261009/`; they are not
additional datasets and must not be mixed with the active masks. New development
pilots belong under `.cache/segmentation_repair/`, not beside the active export.

## Method and limitations

This is a deterministic classical image-processing baseline. Saturation and colour
contrast isolate candidate leaf foreground; morphological filling and connected
components give separate IDs to disconnected regions. Version 3 requires both red
and green to exceed blue for leaf foreground, reducing warm-paper/shadow leakage.
It uses stronger red/green contrast for lesion candidates. Bright low-saturation
pixels are no longer unconditional lesions: they must be within three processing
pixels of a brown seed. A median filter reduces
isolated candidate noise. Eight-connected components define lesion IDs.

The revision was inspected on a 36-image training pilot and high-coverage training
examples. Settings were frozen before the full export; test results were not used
to choose thresholds. They are fixed for the
full export. The algorithm never consumes disease class labels; healthy images may
also have candidate lesions. This behaviour is explicitly flagged for review.

All output masks are **unreviewed pseudo-masks**. Colour does not establish disease.
Touching leaves/lesions may merge and fragmented tissue may split. Leaf scald with
very pale tips and symptoms at leaf edges can be missed; shadows and normal leaf
colour can produce false lesion candidates. Processing uses a maximum side of
800 pixels, preserves aspect ratio, and restores labels to source size with
nearest-neighbour interpolation. Small lesions and exact boundaries are limited
by this processing resolution. No IoU, Dice, mask AP, disease severity or disease
area accuracy can be measured without independent reviewed masks.

Morphology and labeling use [SciPy ndimage](https://docs.scipy.org/doc/scipy/reference/ndimage.html).
No segmentation model weights were trained or downloaded.

## Historical version 2 export (8 October 2026; archived)

All 1,941 images were processed without execution failures: 1,241 training, 311
validation and 389 test images. The output contains 1,967 leaf components and
51,818 lesion candidates. **347 of 348 healthy-labelled images also contain lesion
candidates**, showing substantial false-positive behaviour. Four diseased-labelled
images have no lesion candidates. These outputs are a preliminary annotation aid,
not reliable disease masks. Do not train on them unchanged or present the candidate
counts as confirmed lesions. No threshold was selected using held-out results.

## Version 3 rebuild (9 October 2026)

All 1,941 images were regenerated with zero processing failures and the same
1,241/311/389 split membership. The export contains 2,033 leaf components and
27,735 lesion candidates. Images above 50% candidate coverage decreased from
259 to 72. Healthy-labelled images with any candidate pixels decreased from
347 to 199. Mean candidate coverage on the 222 healthy training images decreased
from 14.503% to 0.808%; nine still exceed 5%, so this is not a complete semantic fix.

There are 147 images flagged for a large connected candidate and 21 for complex
foreground. Twenty-four diseased-labelled images have no candidates, compared
with four previously: the stricter rule also misses some damage. These are
diagnostic tradeoffs, not accuracy measurements. See `repair_comparison.json`
and `repair_preview.jpg` in the active export, and review the prioritized queue.

## Output format

`artifacts/instance_segmentation/` contains:

| Output | Meaning |
|---|---|
| `leaf_instances/<split>/<class>/<source filename>.png` | Native-size uint16 leaf IDs; zero is background |
| `lesion_instances/<split>/<class>/<source filename>.png` | Native-size uint16 candidate IDs; zero means no candidate |
| `metadata/<split>/<class>/<source filename>.json` | Source hash, dimensions, instance IDs, pixel areas, bounding boxes, parent leaf IDs and review flags |
| `overlays/<split>/<class>/<source filename>.jpg` | Coloured leaf and lesion candidates for inspection |
| `instances.csv` | One row per successfully processed image, with flags and metadata path |
| `summary.json` | Method, parameters, counts, errors, split provenance and limitations |
| `validation.json` | Saved-mask consistency checks; does not measure accuracy |
| `preview.jpg` | Training-image overlay contact sheet |
| `index.html` | Searchable full-dataset gallery; original photos beside draft overlays |
| `review_queue.csv` | All images in review-priority order; large/complex regions first |
| `repair_comparison.json` | Before/after candidate statistics, not accuracy scores |

Leaf and lesion IDs each start at one in each image. These are **instance IDs, not
semantic labels**: do not pass them to `segmentation.affected_area`, whose mask
encoding is different. Lesion masks overlap their parent leaf masks by design.
Class folders retain the original image label; individual lesions are not assigned
a confirmed disease category. Bounding boxes are `[x, y, width, height]` in the
EXIF-corrected image coordinates. A zero candidate mask does not confirm health.

```python
import numpy as np
from PIL import Image

ids = np.asarray(Image.open("path/to/leaf_instances/image.JPG.png"))
one_leaf = ids == 1
```

Review flags identify absent leaves, multiple leaf components, large foreground or
candidate coverage, candidates on healthy images, and diseased-class images with
no candidate. Every image requires human review regardless of other flags.

Use `--split train --limit 36 --output .cache/segmentation_repair/new_pilot` for a balanced pilot.
Use a fresh `--output` to rerun; previous output directories are not overwritten.
Use `--max-side` to change processing resolution, noting that minimum pixel-area
thresholds then have a different effective size. Failed images are recorded and
produce a nonzero exit code.

## Next step for trained instance segmentation

Correct leaf boundaries and annotate each lesion separately with unique IDs,
including ambiguous and empty cases. Keep reviewer identity and review status.
Retain split membership and keep annotations derived from the same source together.
Train an instance segmentation network only after sufficient reviewed training
annotations exist, and evaluate on independently reviewed held-out annotations.
Pseudo-mask counts are not a substitute for those accuracy measurements.

## What the repair does and does not establish

Cyan is only the leaf boundary; red is an unconfirmed lesion candidate. The leaf
interior is no longer painted with arbitrary instance colours. This avoids
mistaking a whole-leaf colour overlay for a whole-leaf disease prediction.

Large connected candidates and complex foregrounds are explicitly queued for
review. There is no arbitrary maximum lesion size: a large damaged region can be
real. Stricter colour rules can miss pale damage, and transmitted shadows can
still distort a leaf boundary. The full dataset is regenerated consistently, but
that is not proof that every pixel is correct. Review and correct the native PNGs
before changing annotation review flags. The semantic U-Net drafts are refreshed
from the same version; previous drafts are archived, not silently discarded.
