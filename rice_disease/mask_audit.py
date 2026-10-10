"""Audit actual semantic masks and review progress without training or scoring a model."""
import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

if __name__ == '__main__' and not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'rice_disease'

from .data import ROOT, load_manifest, write_json
from .segmentation_data import inspect_mask, sha256


def audit_masks(manifest, artifacts=ROOT/'artifacts'):
    manifest = Path(manifest).resolve()
    originals, summary = load_manifest(artifacts)
    sources = {r['path']: r for r in originals}
    with manifest.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        required = {'image_path', 'mask_path', 'split', 'class_name', 'source_sha256',
                    'review_status', 'reviewed_by'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Annotation CSV is missing required columns')
        rows = list(reader)
    splits = {name: Counter() for name in ('train', 'validation', 'test')}
    classes, issues, seen, mask_paths = {}, [], set(), set()
    for row in rows:
        identity = row['image_path']
        source = sources.get(identity)
        problems = []
        if identity in seen:
            problems.append('Duplicate annotation image')
        seen.add(identity)
        if not source or any(row[key] != source[key] for key in ('split', 'class_name')):
            problems.append('Source or split/class differs from audited manifest')
        elif row['source_sha256'] != source['sha256']:
            problems.append('Source hash differs from audited manifest')
        mask = (manifest.parent / row['mask_path']).resolve()
        if not row['mask_path'] or not mask.is_relative_to(manifest.parent):
            problems.append('Mask must be inside annotation workspace')
        if mask in mask_paths:
            problems.append('Mask file is shared by multiple source images')
        mask_paths.add(mask)
        reviewed = row['review_status'] == 'reviewed'
        if row['review_status'] not in ('reviewed', 'unreviewed'):
            problems.append('Unknown review status')
        if reviewed and not row['reviewed_by'].strip():
            problems.append('Reviewer identity is missing')
        counts = None
        if not problems:
            try:
                counts = inspect_mask(dict(row, image=Path(summary['dataset'])/identity, mask=mask))
            except (OSError, ValueError) as exc:
                problems.append(str(exc))
        if source:
            split = splits[source['split']]
            by_class = classes.setdefault(source['class_name'], Counter())
            for group in (split, by_class):
                group['annotations'] += 1
                group['reviewed'] += int(reviewed)
                group['unreviewed'] += int(not reviewed)
                group['valid_masks'] += int(counts is not None)
                group['valid_reviewed_masks'] += int(reviewed and counts is not None)
                group['invalid_masks'] += int(bool(problems))
                group['reviewed_affected_pixels'] += int(counts[2]) if reviewed and counts is not None else 0
                if counts is not None:
                    group['healthy_with_affected_pixels'] += int(source['class_name'] == 'healthy' and counts[2] > 0)
                    group['diseased_without_affected_pixels'] += int(source['class_name'] != 'healthy' and counts[2] == 0)
        if problems:
            issues.append({'image_path': identity, 'problems': problems})
    missing = sorted(set(sources) - seen)
    blockers = []
    if issues:
        blockers.append(f'{len(issues)} annotation records have integrity errors; repair them first.')
    for name in ('train', 'validation'):
        if splits[name]['valid_reviewed_masks'] < 2:
            blockers.append(f'{name}: at least two valid reviewed masks required by the trainer.')
        if not splits[name]['reviewed_affected_pixels']:
            blockers.append(f'{name}: reviewed affected-leaf pixels are missing.')
    return {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'invalid_annotations' if issues else ('awaiting_reviewed_masks' if blockers else 'ready_for_training'),
        'training_ready': not blockers,
        'expected_images': len(originals), 'annotation_records': len(rows),
        'valid_masks': sum(s['valid_masks'] for s in splits.values()),
        'reviewed_masks': sum(s['reviewed'] for s in splits.values()),
        'valid_reviewed_masks': sum(s['valid_reviewed_masks'] for s in splits.values()),
        'unreviewed_masks': sum(s['unreviewed'] for s in splits.values()),
        'by_split': {k: dict(v) for k, v in splits.items()},
        'by_class': {k: dict(v) for k, v in classes.items()},
        'missing_annotations': missing, 'integrity_errors': issues, 'training_blockers': blockers,
        'annotation_manifest_sha256': sha256(manifest),
        'split_manifest_sha256': summary['manifest_sha256'],
        'mask_labels': {'0': 'background', '1': 'unaffected_leaf', '2': 'affected_leaf'},
        'accuracy_evaluated': False,
        'note': 'File validity is not boundary accuracy. Unreviewed masks remain drafts. '
                'Minimum training counts are software guards, not sufficient dataset sizes. '
                'Test masks are checked for integrity only; no model predictions or scores are computed.',
    }


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, default=ROOT/'artifacts/segmentation_annotations/annotations.csv')
    parser.add_argument('--artifacts', type=Path, default=ROOT/'artifacts')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/pixel_mask_audit.json')
    args = parser.parse_args()
    try:
        result = audit_masks(args.annotations, args.artifacts)
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'Mask audit error: {exc}\n')


if __name__ == '__main__':
    main()
