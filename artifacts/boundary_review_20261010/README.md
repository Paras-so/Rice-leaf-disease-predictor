# Boundary review, 10 October 2026

Open `index.html` for original photographs beside corrected semantic overlays.
The corrections were inspected visually by Codex at a 512-pixel display size.
They are approximate AI annotations, not agricultural expert ground truth.

- Inspected: 60 images, spanning all six classes and the existing train/validation splits.
- Accepted: 56 corrected references (44 training, 12 validation).
- Deferred: IDs 14, 15, 35 and 41 because of blur or ambiguous tissue boundaries.
- Remaining unreviewed across the dataset: 1,885, including the deferred images.
- Test images reviewed or used for model selection: zero.

Corrections remove cast shadows from the leaf outline, remove highlight/edge
false positives, restore pale damaged tissue and lesion centres, and replace
broad erroneous regions with image-specific boundaries. Broad damage is retained
when it is visible; no arbitrary maximum affected-area percentage is applied.

`selection.json` identifies the inspected sources. `edits.json` records polygons,
colour-assisted regions and review decisions. Coordinates use a 512 by 512 display
and are mapped back to native image dimensions. `correction_records.json` contains
the original and corrected mask hashes, source identity, counts and review method.
Native corrected masks contain only 0 (background), 1 (unaffected leaf), 2 (affected leaf).

`manifest_before.csv` and `before/` preserve the pre-correction state. Accepted
corrections were copied into the authoritative `../segmentation_annotations/`
workspace, whose CSV records the AI reviewer identity and links to this review.
The heuristic instance export remains a draft; the full dataset gallery links
reviewed images to these semantic corrections.

Further corrections should use a new review workspace. The script prevents
rerendering this committed workspace and overwriting its provenance.
