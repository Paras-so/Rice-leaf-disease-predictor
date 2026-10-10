"""Prepare annotation drafts and load only explicitly reviewed semantic masks."""
import csv
import hashlib
import json
from pathlib import Path

if __name__ == "__main__" and not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "rice_disease"

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

from .data import ROOT, load_manifest, write_json
from .unet import letterbox


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare_annotations(candidates, output, artifacts=ROOT/'artifacts'):
    candidates, output = Path(candidates).resolve(), Path(output).resolve()
    rows, summary = load_manifest(artifacts)
    dataset = Path(summary['dataset']).resolve()
    if (output == dataset or output.is_relative_to(dataset) or dataset.is_relative_to(output)
            or output == candidates or output.is_relative_to(candidates) or candidates.is_relative_to(output)):
        raise ValueError('Use a separate annotation output directory')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Annotation output is not empty; preserving existing work')
    candidate_summary = json.loads((candidates/'summary.json').read_text())
    if candidate_summary['manifest_sha256'] != summary['manifest_sha256'] or candidate_summary['processed'] != len(rows) or candidate_summary['failed']:
        raise ValueError('Candidate masks do not cover the audited dataset')
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/'annotations.csv'
    with manifest.open('w', newline='', encoding='utf-8') as stream:
        fields = ['image_path', 'mask_path', 'split', 'class_name', 'source_sha256', 'review_status', 'reviewed_by']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            relative = Path(row['split'])/row['class_name']/(Path(row['path']).name+'.png')
            with Image.open(candidates/'leaf_instances'/relative) as im:
                leaves = np.array(im)
            with Image.open(candidates/'lesion_instances'/relative) as im:
                lesions = np.array(im)
            if leaves.shape != lesions.shape or np.any((lesions > 0) & (leaves == 0)):
                raise ValueError(f'Invalid candidate geometry: {relative}')
            labels = (leaves > 0).astype(np.uint8)
            labels[lesions > 0] = 2
            destination = output/'masks'/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(labels).save(destination)
            writer.writerow({'image_path': row['path'], 'mask_path': destination.relative_to(output).as_posix(),
                             'split': row['split'], 'class_name': row['class_name'], 'source_sha256': row['sha256'],
                             'review_status': 'unreviewed', 'reviewed_by': ''})
            if index % 250 == 0:
                print(f'Prepared {index}/{len(rows)} annotation drafts', flush=True)
    write_json(output/'summary.json', {'images': len(rows), 'reviewed': 0,
               'mask_labels': {'0': 'background', '1': 'unaffected_leaf', '2': 'affected_leaf'},
               'manifest_sha256': summary['manifest_sha256'], 'source_candidates': str(candidates),
               'source_method': candidate_summary.get('method', 'unspecified'),
               'source_summary_sha256': sha256(candidates/'summary.json'),
               'warning': 'Unreviewed colour-based drafts. Correct every leaf and lesion boundary before marking reviewed.'})
    print(f'Annotation manifest: {manifest}', flush=True)
    return manifest


def reviewed_records(manifest, artifacts=ROOT/'artifacts'):
    manifest = Path(manifest).resolve()
    originals, summary = load_manifest(artifacts)
    source_rows = {row['path']: row for row in originals}
    with manifest.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    reviewed, seen, masks_seen = [], set(), set()
    for row in rows:
        identity = row['image_path']
        if identity in seen:
            raise ValueError(f'Duplicate annotation image: {identity}')
        seen.add(identity)
        original = source_rows.get(identity)
        if (not original or row['split'] != original['split']
                or row['class_name'] != original['class_name']
                or row['source_sha256'] != original['sha256']):
            raise ValueError(f'Annotation source/split differs from audited manifest: {identity}')
        if row.get('review_status') != 'reviewed':
            continue
        if not row.get('reviewed_by', '').strip():
            raise ValueError(f'Reviewer identity is missing: {identity}')
        mask = (manifest.parent / row['mask_path']).resolve()
        if not mask.is_relative_to(manifest.parent):
            raise ValueError('Masks must be inside the annotation workspace')
        if mask in masks_seen:
            raise ValueError('Reviewed mask file is shared by multiple source images')
        masks_seen.add(mask)
        reviewed.append(dict(row, image=Path(summary['dataset'])/identity, mask=mask))
    return reviewed, summary


def inspect_mask(row):
    """Check source identity and native-size semantic labels; not annotation accuracy."""
    if sha256(row['image']) != row['source_sha256']:
        raise ValueError(f"Source changed: {row['image_path']}")
    with Image.open(row['image']) as source:
        source.load()
        size = source.size[::-1] if source.getexif().get(274) in (5, 6, 7, 8) else source.size
    with Image.open(row['mask']) as annotated:
        mask = np.array(annotated)
    if mask.ndim != 2 or mask.shape != (size[1], size[0]) or not np.isin(mask, [0, 1, 2]).all():
        raise ValueError(f"Invalid semantic mask: {row['mask']}")
    if not np.any(mask > 0):
        raise ValueError(f"Mask has no leaf: {row['mask']}")
    return np.bincount(mask.ravel(), minlength=3)


class ReviewedMasks(Dataset):
    def __init__(self, records, image_size, augment=False, cache=False):
        self.records, self.image_size, self.augment = records, image_size, augment
        self.cache = {} if cache else None
        self.provenance = []
        self.label_pixels = np.zeros(3, dtype=np.int64)
        for row in records:
            self.label_pixels += inspect_mask(row)
            self.provenance.append({'image_path': row['image_path'], 'source_sha256': row['source_sha256'],
                                    'mask_sha256': sha256(row['mask']), 'split': row['split'],
                                    'reviewed_by': row['reviewed_by'],
                                    'review_method': row.get('review_method') or 'unspecified',
                                    'review_record': row.get('review_record') or None})

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        if self.cache is not None and index in self.cache:
            tensor, mask = self.cache[index]
        else:
            with Image.open(row['image']) as source, Image.open(row['mask']) as annotated:
                tensor, mask, _, _ = letterbox(source, self.image_size, np.array(annotated))
            if self.cache is not None:
                self.cache[index] = (tensor, mask)
        if self.augment:
            # Same geometry for image and mask; validation/test have no augmentation.
            for dimension in (-1, -2):
                if torch.rand(()) < 0.5:
                    tensor, mask = tensor.flip(dimension), mask.flip(dimension)
            turns = int(torch.randint(4, ()))
            tensor, mask = tensor.rot90(turns, (-2, -1)), mask.rot90(turns, (-2, -1))
        return tensor, mask


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, default=ROOT/'artifacts/instance_segmentation')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/segmentation_annotations')
    args = parser.parse_args()
    try:
        prepare_annotations(args.candidates, args.output)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'Annotation error: {exc}\n')
