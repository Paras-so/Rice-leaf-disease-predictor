# Pixel-level annotation is the next data requirement

The supplied dataset contains six image-level classification labels and no pixel
masks. The software therefore does not claim to predict lesion masks or severity.
No growth-stage model or hardware interface is included.

## Annotation protocol

Create a PNG mask for each selected, orientation-corrected source image at the
same dimensions. Use integer labels: 0 for background, 1 for unaffected leaf, 2 for
diseased leaf. A binary lesion mask alone cannot establish the leaf-area denominator.
Keep a mapping from image path to mask path, annotator, review status, and source
plant/field identity where available. Do not classify yellow pixels automatically
as disease: lighting, natural aging and nutrient stress can look similar.

Start with a reviewed pilot spanning all five diseases and healthy leaves; include
small lesions, leaf edges, shadows, and complex backgrounds. Have an agricultural
reviewer resolve ambiguous disease boundaries. Healthy images also require leaf
boundaries; their lesion mask is empty. Do not treat an unannotated lesion as healthy.

Use the existing split manifest to keep original-image derivatives together.
No annotation or augmentation of an image may enter another split. Collect more
images with plant/field identifiers before claiming field-level generalization.

## Model and evaluation plan

After sufficient reviewed masks exist, compare a compact U-Net baseline with a
pretrained segmentation encoder or DeepLab model. Split at plant or field level
where those identifiers are available. Use nearest-neighbor resizing for labels,
and apply exactly the same geometric transforms to an image and its mask.

Report lesion and leaf IoU/Dice separately, including empty-lesion cases and
per-disease results. Overall pixel accuracy can hide poor lesion performance when
background dominates. Evaluate affected-area error against human annotations.

Affected area = diseased leaf pixels / all leaf pixels × 100. The implemented
`rice_disease.segmentation.affected_area` performs this arithmetic only for supplied
label masks. It rejects an empty leaf denominator and invalid labels. The app
labels these results as annotation-derived, never as model predictions.

Disease severity categories and spray thresholds require disease-specific evidence;
no generic percentage cutoffs have been invented.
