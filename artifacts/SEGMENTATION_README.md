# Segmentation outputs

Use **`instance_segmentation/index.html`** to browse the complete current export.
It contains all 1,941 images, not a two-image-per-class demonstration.

- `instance_segmentation/`: the only active instance-mask export (version 3).
  Cyan outlines leaf foreground; red shows unconfirmed lesion candidates.
  `review_queue.csv` prioritizes suspicious/large regions; `repair_comparison.json`
  records before/after candidate statistics, not accuracy.
- `segmentation_annotations/`: semantic versions of the same drafts for U-Net.
  Values are 0 background, 1 remaining leaf and 2 candidate affected tissue.
  This is a different mask encoding, not a duplicate instance-segmentation run.
  There are 56 corrected AI-reviewed masks and 1,885 unreviewed drafts. Only reviewed
  entries are used for U-Net. These semantic corrections supersede the corresponding
  heuristic instance drafts; instance overlays retain their original draft status.
- `boundary_review_20261010/index.html`: original/corrected pairs for the 56 accepted
  approximate references. Includes image-specific edits, provenance and backups.
- `unet/`: experimental pilot checkpoint and training records (44 train, 12 validation).
  Annotation quality is `ai_visual_reviewed`, not expert ground truth.
- `archive/segmentation_before_v3_20261009/`: previous full export, both 12-image
  pilots, previous semantic drafts and their audit reports, preserved for rollback.
  These are historical outputs; do not mix them into model training.

Disposable algorithm development previews can be generated under
`.cache/segmentation_repair/`; previous scratch previews were removed during cleanup.
Source photos and dataset split membership are unchanged.

See the [complete step-by-step workflow](../docs/SEGMENTATION_STEP_BY_STEP.md)
and [folder map](../docs/FOLDER_GUIDE.md).

File validation checks encoding, alignment and provenance. It cannot prove every
boundary is correct. Shadows, complex backgrounds and pale damage still require
pixel-level review before these drafts become training ground truth.
