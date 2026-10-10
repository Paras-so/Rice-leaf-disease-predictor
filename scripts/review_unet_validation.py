"""Export original-resolution validation diagnostics; never load test images."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from rice_disease.data import ROOT, write_json
from rice_disease.segmentation_data import ReviewedMasks, reviewed_records, sha256
from rice_disease.train_segmentation import SegmentationMetrics
from rice_disease.unet import load_unet, mask_overlay, measure_prediction, predict_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'artifacts/unet/model.pt')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/unet/validation_review')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous validation diagnostics; choose a new output')
    torch.set_num_threads(4)
    model, checkpoint = load_unet(args.checkpoint)
    manifest = ROOT/'artifacts/segmentation_annotations/annotations.csv'
    rows, split = reviewed_records(manifest)
    if (checkpoint['split_sha256'] != split['manifest_sha256']
            or checkpoint['annotation_manifest_sha256'] != sha256(manifest)):
        raise ValueError('Annotations/split changed since training')
    rows = [r for r in rows if r['split'] == 'validation']
    dataset = ReviewedMasks(rows, checkpoint['config']['image_size'])
    provenance = json.loads((args.checkpoint.parent/'training_manifest.json').read_text())
    expected = [r for r in provenance if r['split'] == 'validation']
    if dataset.provenance != expected:
        raise ValueError('Validation pixels changed since training')
    args.output.mkdir(parents=True)
    aggregate, by_class, images = SegmentationMetrics(), {}, []
    sheet = Image.new('RGB', (960, 350*len(rows)), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        with Image.open(row['image']) as source:
            mask, area, rgb = predict_mask(model, checkpoint, source)
        with Image.open(row['mask']) as source:
            reference = np.array(source)
        aggregate.update(mask, reference)
        by_class.setdefault(row['class_name'], SegmentationMetrics()).update(mask, reference)
        meter = SegmentationMetrics()
        meter.update(mask, reference)
        reference_area = measure_prediction(reference)['affected_area_percent']
        images.append({'image_path': row['image_path'], 'class_name': row['class_name'],
                       'reference_area_percent': reference_area, 'prediction': area, 'metrics': meter.result()})
        Image.fromarray(mask).save(args.output/f'{index}_mask.png')
        overlay = mask_overlay(rgb, mask)
        overlay.save(args.output/f'{index}_overlay.jpg')
        predicted_text = f"{area['affected_area_percent']:.2f}%" if area['affected_area_percent'] is not None else 'no leaf'
        columns = [(rgb, row['class_name']), (mask_overlay(rgb, reference), f'AI reference: {reference_area:.2f}%'),
                   (overlay, f'U-Net: {predicted_text}')]
        for col, (picture, label) in enumerate(columns):
            tile = ImageOps.contain(picture, (315, 315))
            sheet.paste(tile, (col*320, index*350+30))
            draw.text((col*320+4, index*350+4), label, fill='black')
    sheet.save(args.output/'comparison.jpg', quality=95)
    for page, start in enumerate(range(0, len(rows), 4)):
        sheet.crop((0, start*350, 960, min(start+4, len(rows))*350)).save(
            args.output/f'comparison_{page}.jpg', quality=95)
    result = {'split': 'validation', 'test_images_used': 0,
              'resolution': 'original EXIF-corrected image dimensions',
              'checkpoint_sha256': sha256(args.checkpoint), 'best_epoch': checkpoint['epoch'],
              'label_quality': checkpoint['label_quality'], 'quality_note': checkpoint['quality_note'],
              'validation': aggregate.result(), 'by_class': {k: v.result() for k, v in by_class.items()},
              'images': images, 'evaluated_masks': dataset.provenance, 'field_validated': False}
    write_json(args.output/'metrics.json', result)
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history = json.loads((args.checkpoint.parent/'history.json').read_text())
    epochs = [r['epoch'] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    axes[0].plot(epochs, [r['train_loss'] for r in history], label='Train')
    axes[0].plot(epochs, [r['validation_loss'] for r in history], label='Validation')
    axes[0].set(title='Loss', xlabel='Epoch')
    for name in ('leaf', 'lesion'):
        axes[1].plot(epochs, [r['validation'][name+'_dice'] for r in history], label=name.title())
    axes[1].set(title='Validation Dice at model resolution', xlabel='Epoch', ylim=(0, 1))
    for ax in axes[:2]:
        ax.axvline(checkpoint['epoch'], color='grey', linestyle=':', label='Selected epoch')
        ax.legend()
    actual = [r['reference_area_percent'] for r in images]
    predicted = [r['prediction']['affected_area_percent'] for r in images]
    axes[2].scatter(actual, predicted)
    upper = max(actual + [p for p in predicted if p is not None] + [1])*1.1
    axes[2].plot([0, upper], [0, upper], color='grey', linestyle=':')
    axes[2].set(title='Visible affected area (%)', xlabel='AI reference', ylabel='U-Net prediction')
    fig.suptitle('Experimental pilot: agreement with approximate AI-reviewed masks; not expert accuracy')
    fig.savefig(args.output/'training_and_area.png', dpi=160)
    plt.close(fig)
    print(json.dumps({k: v for k, v in result.items() if k not in ('images', 'evaluated_masks', 'by_class')}, indent=2))


if __name__ == '__main__':
    main()
