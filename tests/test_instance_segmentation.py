"""Geometric and output invariants for the unreviewed segmentation baseline."""
import unittest
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from rice_disease.instance_segmentation import candidate_masks, components, instance_records, run


class InstanceSegmentationTests(unittest.TestCase):
    def test_warm_paper_and_pale_leaf_regions_are_not_automatic_lesions(self):
        pixels = np.full((200, 300, 3), [215, 202, 190], dtype=np.uint8)
        pixels[30:170, 80:220] = [60, 140, 30]
        # A pale highlight enclosed by leaf used to be marked diseased solely
        # because it was bright and low-saturation.
        pixels[60:140, 110:190] = [210, 215, 205]
        _, leaves, lesions, _ = candidate_masks(Image.fromarray(pixels))
        self.assertEqual(leaves[0, 0], 0)
        self.assertEqual(leaves[100, 150], 1)
        self.assertEqual(int(lesions.max()), 0)

    def test_brown_shadow_is_not_a_large_disease_region(self):
        pixels = np.full((200, 300, 3), 245, dtype=np.uint8)
        pixels[40:90, :] = [80, 140, 30]
        pixels[90:140, :] = [104, 88, 52]
        _, _, lesions, _ = candidate_masks(Image.fromarray(pixels))
        self.assertEqual(int(lesions.max()), 0)

    def test_real_broad_brown_candidate_is_not_removed_by_area_cap(self):
        pixels = np.full((200, 300, 3), 255, dtype=np.uint8)
        pixels[20:180, 80:220] = [60, 140, 30]
        pixels[20:140, 80:220] = [140, 80, 30]
        _, leaves, lesions, _ = candidate_masks(Image.fromarray(pixels))
        self.assertGreater(np.count_nonzero(lesions)/np.count_nonzero(leaves), 0.7)

    def test_disconnected_leaves_and_lesions_have_distinct_ids_and_parents(self):
        pixels = np.full((200, 300, 3), 255, dtype=np.uint8)
        pixels[20:180, 30:100] = [60, 140, 30]
        pixels[20:180, 190:260] = [60, 140, 30]
        pixels[50:70, 50:70] = [140, 80, 30]
        pixels[120:140, 210:230] = [140, 80, 30]
        _, leaves, lesions, _ = candidate_masks(Image.fromarray(pixels))
        self.assertEqual(leaves.dtype, np.uint16)
        self.assertEqual(int(leaves.max()), 2)
        self.assertEqual(int(lesions.max()), 2)
        self.assertTrue(np.all(leaves[lesions > 0] > 0))
        records = instance_records(lesions, 'lesion_candidate', leaves)
        self.assertEqual([r['leaf_instance_id'] for r in records], [1, 2])
        self.assertTrue(all(390 <= r['area_pixels'] <= 400 for r in records))
        self.assertEqual(records[0]['bbox_xywh'], [50, 50, 20, 20])

    def test_blank_and_uniform_green_have_no_lesion_candidates(self):
        for colour, leaf_count in [('white', 0), ((60, 140, 30), 1)]:
            _, leaves, lesions, _ = candidate_masks(Image.new('RGB', (250, 190), colour), max_side=100)
            self.assertEqual(leaves.shape, (190, 250))
            self.assertEqual(int(leaves.max()), leaf_count)
            self.assertEqual(int(lesions.max()), 0)

    def test_small_noise_is_removed_and_ids_remain_contiguous(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[0, 0] = True
        mask[4:8, 4:8] = True
        mask[12:16, 12:16] = True
        labels = components(mask, 4)
        np.testing.assert_array_equal(np.unique(labels), [0, 1, 2])
        self.assertEqual(labels[0, 0], 0)

    def test_exif_orientation_precedes_mask_generation(self):
        image = Image.new('RGB', (100, 60), (60, 140, 30))
        image.getexif()[274] = 6
        rgb, leaves, lesions, _ = candidate_masks(image)
        self.assertEqual(rgb.size, (60, 100))
        self.assertEqual(leaves.shape, (100, 60))
        self.assertEqual(lesions.shape, leaves.shape)

    def test_export_preserves_split_source_and_reports_checksum_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, artifacts, output = root/'dataset', root/'artifacts', root/'output'
            dataset.mkdir()
            artifacts.mkdir()
            source = dataset/'leaf.png'
            Image.new('RGB', (120, 80), (60, 140, 30)).save(source)
            original = source.read_bytes()
            rows = [{'path': 'leaf.png', 'class_name': 'healthy', 'label': 0,
                     'sha256': hashlib.sha256(original).hexdigest(), 'split': 'validation'},
                    {'path': 'changed.png', 'class_name': 'healthy', 'label': 0,
                     'sha256': 'incorrect', 'split': 'test'}]
            (dataset/'changed.png').write_bytes(original)
            manifest = artifacts/'split_manifest.csv'
            with manifest.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (artifacts/'split_summary.json').write_text(json.dumps({
                'dataset': str(dataset), 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(artifacts, output), 1)
            report = json.loads((output/'summary.json').read_text())
            self.assertEqual((report['processed'], report['failed']), (1, 1))
            self.assertEqual(report['by_split'], {'validation': 1})
            metadata = json.loads((output/'metadata/validation/healthy/leaf.png.json').read_text())
            self.assertEqual(metadata['leaves'][0]['area_pixels'], 120*80)
            with Image.open(output/'leaf_instances/validation/healthy/leaf.png.png') as mask:
                self.assertEqual(mask.size, (120, 80))
                self.assertEqual(int(np.asarray(mask).max()), 1)
            self.assertEqual(source.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, 'not empty'):
                run(artifacts, output)


if __name__ == '__main__':
    unittest.main()
