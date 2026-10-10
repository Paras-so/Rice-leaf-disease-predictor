"""Validate exported mask geometry and provenance, not segmentation accuracy."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rice_disease.data import ROOT, load_manifest, write_json


def validate(output, artifacts):
    summary = json.loads((output / 'summary.json').read_text())
    rows, split = load_manifest(artifacts)
    expected = {row['path']: row for row in rows}
    if summary['status'] != 'complete' or summary['failed']:
        raise ValueError('Export is not complete or contains failures')
    if summary['manifest_sha256'] != split['manifest_sha256']:
        raise ValueError('Export split checksum differs')
    with (output / 'instances.csv').open(newline='', encoding='utf-8') as stream:
        exported = list(csv.DictReader(stream))
    if len(exported) != summary['processed']:
        raise ValueError('Summary and CSV counts differ')
    seen = set()
    totals = Counter()
    for index, row in enumerate(exported, 1):
        path = row['source_path']
        if path in seen:
            raise ValueError(f'Duplicate source: {path}')
        seen.add(path)
        source = expected[path]
        meta = json.loads((output / row['metadata']).read_text())
        assert meta['source_path'] == path
        assert meta['split'] == row['split'] == source['split']
        assert meta['image_class'] == row['class_name'] == source['class_name']
        assert meta['source_sha256'] == source['sha256']
        assert meta['annotation_status'] == 'unreviewed_pseudo_masks'
        with Image.open(Path(split['dataset']) / path) as source_image:
            native_size = source_image.size
            if source_image.getexif().get(274) in (5, 6, 7, 8):
                native_size = native_size[::-1]
        assert (meta['width'], meta['height']) == native_size
        with Image.open(output / meta['files']['leaf_instances']) as image:
            leaves = np.array(image)
        with Image.open(output / meta['files']['lesion_instances']) as image:
            lesions = np.array(image)
        assert leaves.shape == lesions.shape == (meta['height'], meta['width'])
        assert leaves.dtype == lesions.dtype == np.uint16
        for mask, records in ((leaves, meta['leaves']), (lesions, meta['lesion_candidates'])):
            sizes = np.bincount(mask.ravel())
            assert int(mask.max()) == len(records)
            for record in records:
                assert sizes[record['instance_id']] == record['area_pixels']
                x, y, width, height = record['bbox_xywh']
                assert 0 <= x < x + width <= meta['width']
                assert 0 <= y < y + height <= meta['height']
        parents = np.zeros(len(meta['lesion_candidates'])+1, dtype=np.uint16)
        for record in meta['lesion_candidates']:
            parents[record['instance_id']] = record['leaf_instance_id']
            assert 1 <= record['leaf_instance_id'] <= len(meta['leaves'])
        positive = lesions > 0
        assert np.array_equal(parents[lesions[positive]], leaves[positive])
        assert len(meta['leaves']) == int(row['leaf_instances'])
        assert len(meta['lesion_candidates']) == int(row['lesion_candidates'])
        assert np.count_nonzero(leaves) == int(row['leaf_pixels'])
        assert np.count_nonzero(lesions) == int(row['candidate_lesion_pixels'])
        with Image.open(output / meta['files']['overlays']) as image:
            image.verify()
        totals.update(images=1, leaf_instances=len(meta['leaves']), lesion_candidates=len(meta['lesion_candidates']))
        if index % 250 == 0:
            print(f'Validated {index}/{len(exported)} images', flush=True)
    assert totals['leaf_instances'] == summary['leaf_instances']
    assert totals['lesion_candidates'] == summary['lesion_candidates']
    result = {'status': 'passed', 'checks': 'Saved PNG dimensions, dtype, IDs, areas, bounds, lesion containment, overlays, source identity and split membership',
              'counts': dict(totals), 'covers_entire_manifest': seen == set(expected),
              'accuracy_evaluated': False}
    write_json(output / 'validation.json', result)
    report = [
        '# Cleaned dataset: leaf and lesion candidate instances', '',
        '**Status: unreviewed pseudo-masks from a colour/morphology baseline.**', '',
        f"Processed **{summary['processed']:,} images** with **{summary['failed']} processing failures**.",
        f"Exported **{summary['leaf_instances']:,} leaf components** and **{summary['lesion_candidates']:,} lesion candidates**.",
        'These counts do not establish object-count or disease-segmentation accuracy.', '',
        '[Browse all images](index.html) | [Training-sample overlays](preview.jpg) | [Review queue](review_queue.csv) | [Per-image results](instances.csv) | [Full summary](summary.json) | [Output verification](validation.json)', '',
        '## Counts', '',
        '| Image class | Images | Leaf components | Lesion candidates |',
        '|---|---:|---:|---:|',
    ]
    for class_name, values in summary['by_class'].items():
        report.append(f"| {class_name} | {values['images']} | {values['leaf_instances']} | {values['lesion_candidates']} |")
    report += ['', 'Split membership: ' + ', '.join(f'{key}={value}' for key, value in summary['by_split'].items()) + '.',
               '', '## Review flags', '', '| Flag | Images |', '|---|---:|']
    report += [f'| {flag} | {count} |' for flag, count in summary['review_flags'].items()]
    report += ['', 'Flags can overlap. All masks require review, including images with no additional flags.',
               '', '## Interpretation', '']
    report += ['Cyan outlines the leaf; red marks unconfirmed lesion candidates. The original leaf colours remain visible. Large regions and complex backgrounds are queued for manual review, not automatically declared diseased.', '']
    report += ['- ' + limitation for limitation in summary['limitations']]
    report += ['', 'The saved-file check passed for mask dimensions, integer IDs, pixel areas, parent-leaf containment, source identity, overlays and split membership. It does not evaluate visual correctness.',
               '', 'Leaf and lesion masks are separate uint16 PNGs: zero is background; positive values identify individual components within that image. Lesion metadata stores the parent leaf ID. Instance IDs are not semantic class values and must not be passed to the annotation-area helper.',
               '', 'Source images, the classification model and application were preserved. Review and correct these masks before using them as training labels.', '']
    (output / 'REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/instance_segmentation')
    parser.add_argument('--artifacts', type=Path, default=ROOT/'artifacts')
    args = parser.parse_args()
    validate(args.output, args.artifacts)
