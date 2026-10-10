"""Train/evaluate U-Net with reviewed masks, or predict a semantic mask and area."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random

if __name__ == '__main__' and not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'rice_disease'

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from .data import ROOT, write_json
from .quantity import rice_tank_quantity
from .segmentation_data import ReviewedMasks, reviewed_records, sha256
from .unet import CLASSES, UNet, load_unet, mask_overlay, predict_mask, segmentation_loss


class SegmentationMetrics:
    def __init__(self):
        self.confusion = np.zeros((3, 3), dtype=np.int64)
        self.area_errors = []
        self.images = self.missing_leaf = self.empty_lesion = self.empty_false_positive = 0

    def update(self, predicted, target):
        valid = target != 255
        predicted, target = np.asarray(predicted)[valid], np.asarray(target)[valid]
        self.confusion += np.bincount(3*target.astype(np.int64)+predicted, minlength=9).reshape(3, 3)
        self.images += 1
        leaf, gt_leaf = np.count_nonzero(predicted > 0), np.count_nonzero(target > 0)
        lesion, gt_lesion = np.count_nonzero(predicted == 2), np.count_nonzero(target == 2)
        if not leaf:
            self.missing_leaf += 1
        if leaf and gt_leaf:
            self.area_errors.append(abs(100*lesion/leaf - 100*gt_lesion/gt_leaf))
        if not gt_lesion:
            self.empty_lesion += 1
            self.empty_false_positive += int(lesion > 0)

    def result(self):
        result = {'images': self.images, 'confusion_matrix': self.confusion.tolist()}
        for name, labels in (('leaf', [1, 2]), ('lesion', [2])):
            tp = int(self.confusion[np.ix_(labels, labels)].sum())
            actual = int(self.confusion[labels, :].sum())
            predicted = int(self.confusion[:, labels].sum())
            result[name+'_dice'] = 2*tp/(actual+predicted) if actual+predicted else None
            result[name+'_iou'] = tp/(actual+predicted-tp) if actual+predicted-tp else None
        result.update(affected_area_mae_percentage_points=float(np.mean(self.area_errors)) if self.area_errors else None,
                      area_comparable_images=len(self.area_errors), predicted_no_leaf_images=self.missing_leaf,
                      empty_lesion_images=self.empty_lesion, empty_lesion_false_positive_images=self.empty_false_positive)
        return result


def validate_config(config):
    if not isinstance(config.get('cache_images', False), bool):
        raise ValueError('cache_images must be boolean')
    for name in ('image_size', 'base_channels', 'batch_size', 'epochs', 'patience', 'cpu_threads'):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    minimum = config.get('min_epochs', 1)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1 or minimum > config['epochs']:
        raise ValueError('min_epochs must be between 1 and epochs')
    if config['image_size'] < 32 or config['image_size'] % 16:
        raise ValueError('image_size must be >=32 and divisible by 16')
    if config['base_channels'] % 4:
        raise ValueError('base_channels must be divisible by four')
    if isinstance(config.get('seed'), bool) or not isinstance(config.get('seed'), int) or config['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    for name in ('learning_rate', 'weight_decay'):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (name == 'learning_rate' and value == 0):
            raise ValueError(f'Invalid {name}')


def train(manifest, artifacts, output, config, device='cpu'):
    validate_config(config)
    records, summary = reviewed_records(manifest, artifacts)
    train_rows = [r for r in records if r['split'] == 'train']
    val_rows = [r for r in records if r['split'] == 'validation']
    if len(train_rows) < 2 or len(val_rows) < 2:
        raise ValueError(f'Reviewed masks are required: found {len(train_rows)} train and {len(val_rows)} validation. Correct masks, set review_status=reviewed and record reviewed_by in annotations.csv. No training was performed.')
    output = Path(output).resolve()
    protected = [Path(summary['dataset']).resolve(), Path(manifest).resolve().parent]
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('Use a separate training output directory')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Training output is not empty; choose another --output')
    torch.set_num_threads(config['cpu_threads'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    training = ReviewedMasks(train_rows, config['image_size'], augment=True, cache=config.get('cache_images', False))
    validation = ReviewedMasks(val_rows, config['image_size'], cache=config.get('cache_images', False))
    review_methods = dict(Counter(r.get('review_method') or 'unspecified' for r in train_rows + val_rows))
    label_quality = 'ai_visual_reviewed' if 'ai_visual' in review_methods else 'reviewed'
    quality_note = ('Pilot trained against approximate AI-reviewed masks; validation measures agreement '
                    'with those masks, not expert-validated disease segmentation accuracy.'
                    if label_quality == 'ai_visual_reviewed' else 'Reviewed annotations; not field-validated.')
    if any(dataset.label_pixels[2] == 0 for dataset in (training, validation)):
        raise ValueError('Both training and validation need reviewed affected-leaf pixels')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'config.json', config)
    write_json(output/'training_manifest.json', training.provenance + validation.provenance)
    train_loader = DataLoader(training, batch_size=config['batch_size'], shuffle=True,
                              generator=torch.Generator().manual_seed(config['seed']), num_workers=0)
    val_loader = DataLoader(validation, batch_size=config['batch_size'], num_workers=0)
    model = UNet(config['base_channels']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    best, stale, history = -1.0, 0, []
    for epoch in range(1, config['epochs']+1):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = segmentation_loss(model(inputs), targets)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*len(inputs)
        model.eval()
        metrics, val_loss = SegmentationMetrics(), 0.0
        with torch.inference_mode():
            for inputs, targets in val_loader:
                logits = model(inputs.to(device))
                val_loss += segmentation_loss(logits, targets.to(device)).item()*len(inputs)
                for predicted, target in zip(logits.argmax(1).cpu().numpy(), targets.numpy()):
                    metrics.update(predicted, target)
        result = metrics.result()
        score = ((result['leaf_dice'] or 0.0) + (result['lesion_dice'] or 0.0))/2
        history.append({'epoch': epoch, 'train_loss': total_loss/len(training),
                        'validation_loss': val_loss/len(validation), 'validation': result,
                        'selection_score': score})
        write_json(output/'history.json', history)
        print(f"Epoch {epoch}/{config['epochs']}: loss={history[-1]['train_loss']:.4f} | validation leaf Dice={result['leaf_dice']} lesion Dice={result['lesion_dice']}", flush=True)
        if score > best:
            best, stale = score, 0
            checkpoint = {'architecture': 'unet', 'classes': CLASSES, 'config': config,
                          'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                          'epoch': epoch, 'training_status': 'trained', 'label_quality': label_quality,
                          'review_methods': review_methods, 'quality_note': quality_note,
                          'split_sha256': summary['manifest_sha256'],
                          'annotation_manifest_sha256': sha256(manifest),
                          'training_provenance_sha256': sha256(output/'training_manifest.json'),
                          'validation': result, 'validation_resolution': config['image_size'],
                          'selection_metric': 'mean_leaf_and_lesion_dice',
                          'train_images': len(training), 'validation_images': len(validation)}
            torch.save(checkpoint, output/'model.tmp.pt')
            (output/'model.tmp.pt').replace(output/'model.pt')
        else:
            stale += 1
        if stale >= config['patience'] and epoch >= config.get('min_epochs', 1):
            break
    write_json(output/'summary.json', {'status': 'trained', 'best_validation_score': best,
               'epochs_run': len(history), 'train_images': len(training), 'validation_images': len(validation),
               'test_images_used': 0, 'label_quality': label_quality, 'field_validated': False,
               'review_methods': review_methods, 'quality_note': quality_note,
               'class_counts': {'train': dict(Counter(r['class_name'] for r in train_rows)),
                                'validation': dict(Counter(r['class_name'] for r in val_rows))},
               'checkpoint_sha256': sha256(output/'model.pt')})
    return output/'model.pt'


def evaluate(checkpoint_path, manifest, artifacts, output, device='cpu'):
    output = Path(output)
    if output.exists():
        raise ValueError('Evaluation output exists; preserve the recorded test evaluation')
    model, checkpoint = load_unet(checkpoint_path, device)
    records, summary = reviewed_records(manifest, artifacts)
    if checkpoint['split_sha256'] != summary['manifest_sha256']:
        raise ValueError('Checkpoint and annotation splits differ')
    rows = [r for r in records if r['split'] == 'test']
    if not rows:
        raise ValueError('No reviewed test masks available')
    dataset = ReviewedMasks(rows, checkpoint['config']['image_size'])
    aggregate, by_class = SegmentationMetrics(), {}
    for row in rows:
        with Image.open(row['image']) as image:
            predicted, _, _ = predict_mask(model, checkpoint, image)
        with Image.open(row['mask']) as image:
            actual = np.array(image)
        aggregate.update(predicted, actual)
        by_class.setdefault(row['class_name'], SegmentationMetrics()).update(predicted, actual)
    result = {'checkpoint_sha256': sha256(checkpoint_path), 'split_sha256': summary['manifest_sha256'],
              'resolution': 'original EXIF-corrected image dimensions', 'test': aggregate.result(),
              'by_class': {name: meter.result() for name, meter in by_class.items()},
              'evaluated_masks': dataset.provenance, 'field_validated': False}
    write_json(output, result)
    return result


def export_prediction(checkpoint_path, image_path, output, device='cpu', *,
                      rule=None, disease=None, region=None, tank_ml=None,
                      area_hectares=None, application_authorized=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Prediction output is not empty; choose another output')
    quantity = None
    if rule is not None:
        quantity = rice_tank_quantity(rule, disease=disease, region=region,
                                     tank_ml=tank_ml, area_hectares=area_hectares,
                                     application_authorized=application_authorized)
    model, checkpoint = load_unet(checkpoint_path, device)
    with Image.open(image_path) as image:
        mask, area, rgb = predict_mask(model, checkpoint, image)
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output/'mask.png')
    mask_overlay(rgb, mask).save(output/'overlay.png')
    area.update(checkpoint_sha256=sha256(checkpoint_path), image_sha256=sha256(image_path))
    area.update(pesticide_quantity=quantity,
                pesticide_status='label_quantity_calculated' if quantity else 'requires_verified_product_label',
                disease=disease,
                disease_source='user-supplied diagnosis' if disease else None)
    write_json(output/'area.json', area)
    return area


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    training = sub.add_parser('train', help='Train using reviewed train/validation masks only')
    training.add_argument('--config', type=Path, default=ROOT/'configs/segmentation.json')
    training.add_argument('--output', type=Path, default=ROOT/'artifacts/unet')
    evaluation = sub.add_parser('evaluate', help='Evaluate a frozen checkpoint on reviewed test masks')
    evaluation.add_argument('--output', type=Path, default=ROOT/'artifacts/unet/test_metrics.json')
    prediction = sub.add_parser('predict', help='Export a mask, overlay and estimated affected percentage')
    prediction.add_argument('image', type=Path)
    prediction.add_argument('--output', type=Path, default=ROOT/'artifacts/unet_prediction')
    prediction.add_argument('--rule', type=Path, help='Verified disease-specific rice product-label JSON')
    prediction.add_argument('--disease', help='Confirmed disease; U-Net does not identify the disease species')
    prediction.add_argument('--region', help='Region matching the verified product label rule')
    prediction.add_argument('--tank-ml', type=float, help='Actual spray mixture volume in mL')
    prediction.add_argument('--area-hectares', type=float, help='Actual area treated, required for per-hectare rates')
    prediction.add_argument('--application-authorized', action='store_true',
                            help='Confirm diagnosis and product-label application conditions have been checked')
    for command in (training, evaluation, prediction):
        command.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    for command in (training, evaluation):
        command.add_argument('--annotations', type=Path, default=ROOT/'artifacts/segmentation_annotations/annotations.csv')
        command.add_argument('--artifacts', type=Path, default=ROOT/'artifacts')
    for command in (evaluation, prediction):
        command.add_argument('--checkpoint', type=Path, default=ROOT/'artifacts/unet/model.pt')
    args = parser.parse_args(argv)
    try:
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is not available; use --device cpu')
        if args.command == 'train':
            checkpoint = train(args.annotations, args.artifacts, args.output,
                               json.loads(args.config.read_text()), args.device)
            print(f'Saved trained U-Net: {checkpoint}')
        else:
            torch.set_num_threads(4)
            if args.command == 'evaluate':
                result = evaluate(args.checkpoint, args.annotations, args.artifacts, args.output, args.device)
            else:
                if not args.rule and (args.region or args.tank_ml is not None
                                      or args.area_hectares is not None or args.application_authorized):
                    raise ValueError('Quantity options require --rule with a verified product label')
                rule = json.loads(args.rule.read_text(encoding='utf-8')) if args.rule else None
                result = export_prediction(args.checkpoint, args.image, args.output, args.device,
                    rule=rule, disease=args.disease, region=args.region, tank_ml=args.tank_ml,
                    area_hectares=args.area_hectares, application_authorized=args.application_authorized)
            print(json.dumps(result, indent=2))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'U-Net error: {exc}\n')


if __name__ == '__main__':
    main()
