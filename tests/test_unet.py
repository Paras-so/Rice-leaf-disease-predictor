"""U-Net geometry, reviewed-only training, metrics, and a synthetic end-to-end run."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from rice_disease.data import write_json
from rice_disease.mask_audit import audit_masks
from rice_disease.quantity import rice_tank_quantity, disease_product_reference
from rice_disease.segmentation_data import ReviewedMasks, prepare_annotations, reviewed_records, sha256
from rice_disease.train_segmentation import SegmentationMetrics, evaluate, export_prediction, train, validate_config
from rice_disease.unet import CLASSES, UNet, letterbox, load_unet, measure_prediction, predict_mask, segmentation_loss


class UNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, root):
        """Synthetic test-only masks; not rice annotations or deployable weights."""
        dataset, artifacts, annotations = root/'images', root/'artifacts', root/'annotations'
        for path in (dataset, artifacts, annotations):
            path.mkdir()
        source_rows, rows = [], []
        for index, split in enumerate(['train', 'train', 'validation', 'validation', 'test', 'test']):
            mask = np.zeros((32, 48), np.uint8)
            mask[4:28, 5:43] = 1
            mask[10:18, 15+index:23+index] = 2
            pixels = np.full((32, 48, 3), 235, np.uint8)
            pixels[mask == 1] = (40, 150, 40)
            pixels[mask == 2] = (180, 60, 30)
            name = f'{index}.png'
            Image.fromarray(pixels).save(dataset/name)
            Image.fromarray(mask).save(annotations/name)
            digest = sha256(dataset/name)
            source_rows.append({'path': name, 'label': 0, 'class_name': 'fixture', 'split': split, 'sha256': digest})
            rows.append({'image_path': name, 'mask_path': name, 'class_name': 'fixture', 'split': split,
                         'source_sha256': digest, 'review_status': 'reviewed', 'reviewed_by': 'SYNTHETIC TEST FIXTURE'})
        for path, records in ((artifacts/'split_manifest.csv', source_rows), (annotations/'annotations.csv', rows)):
            self.write_csv(path, records)
        write_json(artifacts/'split_summary.json', {'dataset': str(dataset), 'manifest_sha256': sha256(artifacts/'split_manifest.csv')})
        return annotations/'annotations.csv', artifacts, rows

    def write_csv(self, path, rows):
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def config(self):
        return {'image_size': 32, 'base_channels': 4, 'batch_size': 2, 'epochs': 2,
                'patience': 2, 'learning_rate': 0.001, 'weight_decay': 0.0001, 'seed': 42, 'cpu_threads': 2}

    def test_mask_audit_reads_review_status_and_validates_unreviewed_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, artifacts, rows = self.fixture(root)
            ready = audit_masks(manifest, artifacts)
            self.assertTrue(ready['training_ready'])
            self.assertEqual(ready['valid_reviewed_masks'], 6)
            for row in rows:
                row.update(review_status='unreviewed', reviewed_by='')
            self.write_csv(manifest, rows)
            drafts = audit_masks(manifest, artifacts)
            self.assertFalse(drafts['training_ready'])
            self.assertEqual(drafts['unreviewed_masks'], 6)
            self.assertEqual(drafts['valid_masks'], 6)
            self.assertEqual(drafts['reviewed_masks'], 0)
            self.assertFalse(drafts['accuracy_evaluated'])
            Image.fromarray(np.full((32, 48), 255, np.uint8)).save(manifest.parent/'0.png')
            invalid = audit_masks(manifest, artifacts)
            self.assertEqual(invalid['valid_masks'], 5)
            self.assertEqual(len(invalid['integrity_errors']), 1)

    def test_mask_audit_rejects_false_review_identity_and_tracks_missing_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, artifacts, rows = self.fixture(Path(temporary))
            rows[0]['reviewed_by'] = ''
            rows[1]['mask_path'] = '../outside.png'
            rows[2]['class_name'] = 'wrong'
            self.write_csv(manifest, rows[:-1])
            result = audit_masks(manifest, artifacts)
            self.assertFalse(result['training_ready'])
            self.assertEqual(len(result['integrity_errors']), 3)
            self.assertEqual(result['missing_annotations'], ['5.png'])
            rows[0]['reviewed_by'] = 'SYNTHETIC'
            self.write_csv(manifest, [rows[0], rows[2]])
            with self.assertRaisesRegex(ValueError, 'differs'):
                reviewed_records(manifest, artifacts)

    def test_network_shape_and_finite_gradients(self):
        model = UNet(4)
        logits = model(torch.randn(2, 3, 35, 49))
        self.assertEqual(tuple(logits.shape), (2, 3, 35, 49))
        target = torch.ones(2, 35, 49, dtype=torch.long)
        target[:, 10:20, 10:20] = 2
        target[:, :3] = 255
        loss = segmentation_loss(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_letterbox_preserves_mask_labels_and_ignores_padding(self):
        image = Image.new('RGB', (80, 40), 'green')
        mask = np.ones((40, 80), np.uint8)
        mask[:, 40:] = 2
        tensor, target, box, rgb = letterbox(image, 32, mask)
        self.assertEqual(box, (0, 8, 32, 16))
        self.assertEqual(set(target.unique().tolist()), {1, 2, 255})
        self.assertTrue((target[:8] == 255).all())
        with self.assertRaises(ValueError):
            letterbox(image, 32, np.full((40, 80), 255, np.uint8))

    def test_prediction_restores_native_geometry_and_handles_empty_leaf(self):
        class HalfLeaf(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def forward(self, inputs):
                logits = inputs.new_zeros((len(inputs), 3, *inputs.shape[-2:])) + self.anchor
                logits[:, 1, :, :inputs.shape[-1]//2] = 10
                logits[:, 2, :, inputs.shape[-1]//2:] = 10
                return logits
        checkpoint = {'config': {'image_size': 32}, 'label_quality': 'reviewed'}
        mask, area, _ = predict_mask(HalfLeaf(), checkpoint, Image.new('RGB', (80, 40)))
        self.assertEqual(mask.shape, (40, 80))
        self.assertEqual(area['affected_area_percent'], 50)
        self.assertIsNone(area['pesticide_quantity'])
        empty = measure_prediction(np.zeros((10, 10), np.uint8))
        self.assertIsNone(empty['affected_area_percent'])
        self.assertEqual(empty['status'], 'no_leaf_detected')

    def test_metrics_include_background_exclusion_and_empty_lesion_false_positives(self):
        meter = SegmentationMetrics()
        meter.update(np.array([[0, 1, 2, 2, 0]]), np.array([[0, 1, 1, 2, 255]]))
        metrics = meter.result()
        self.assertEqual(metrics['leaf_dice'], 1)
        self.assertAlmostEqual(metrics['lesion_dice'], 2/3)
        self.assertAlmostEqual(metrics['affected_area_mae_percentage_points'], 100/3)
        meter.update(np.array([[0, 1, 2]]), np.array([[0, 1, 1]]))
        self.assertEqual(meter.result()['empty_lesion_false_positive_images'], 1)

    def test_unreviewed_annotations_cannot_train_and_split_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, artifacts, rows = self.fixture(root)
            for row in rows:
                row['review_status'] = 'unreviewed'
            self.write_csv(manifest, rows)
            with self.assertRaisesRegex(ValueError, 'Reviewed masks are required'):
                train(manifest, artifacts, root/'model', self.config())
            self.assertFalse((root/'model').exists())
            rows[0]['split'] = 'test'
            self.write_csv(manifest, rows)
            with self.assertRaisesRegex(ValueError, 'source/split'):
                reviewed_records(manifest, artifacts)

    def test_source_integrity_and_invalid_mask_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, artifacts, _ = self.fixture(root)
            records, _ = reviewed_records(manifest, artifacts)
            Image.new('L', (48, 32), 255).save(records[0]['mask'])
            with self.assertRaisesRegex(ValueError, 'Invalid semantic mask'):
                ReviewedMasks(records[:1], 32)
            records[1]['image'].write_bytes(b'changed source')
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                ReviewedMasks(records[1:2], 32)

    def test_instance_conversion_keeps_all_drafts_unreviewed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, artifacts, rows = self.fixture(root)
            candidates = root/'candidates'
            for row in rows:
                with Image.open(manifest.parent/row['mask_path']) as image:
                    original = np.array(image)
                relative = Path(row['split'])/'fixture'/(row['image_path']+'.png')
                for kind, mask in (('leaf_instances', (original > 0).astype(np.uint16)*5),
                                   ('lesion_instances', (original == 2).astype(np.uint16)*9)):
                    path = candidates/kind/relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(mask).save(path)
            split = json.loads((artifacts/'split_summary.json').read_text())
            write_json(candidates/'summary.json', {'manifest_sha256': split['manifest_sha256'], 'processed': 6, 'failed': 0})
            with contextlib.redirect_stdout(io.StringIO()):
                prepared = prepare_annotations(candidates, root/'prepared', artifacts)
            reviewed, _ = reviewed_records(prepared, artifacts)
            self.assertEqual(reviewed, [])
            with prepared.open(newline='') as stream:
                drafts = list(csv.DictReader(stream))
            self.assertEqual(len(drafts), 6)
            with Image.open(prepared.parent/drafts[0]['mask_path']) as actual, Image.open(manifest.parent/'0.png') as expected:
                np.testing.assert_array_equal(np.array(actual), np.array(expected))

    def test_cached_augmentation_keeps_original_pixels_and_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, artifacts, _ = self.fixture(Path(temporary))
            records, _ = reviewed_records(manifest, artifacts)
            cached = ReviewedMasks(records[:1], 32, augment=True, cache=True)
            uncached = ReviewedMasks(records[:1], 32, augment=True)
            for seed in range(8):
                torch.manual_seed(seed)
                actual = cached[0]
                torch.manual_seed(seed)
                expected = uncached[0]
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))
            reference = ReviewedMasks(records[:1], 32)[0]
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(cached.cache[0], reference)))

    def test_ai_review_training_preserves_quality_and_requires_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, artifacts, rows = self.fixture(root)
            for row in rows:
                row.update(review_method='ai_visual', review_record='test-only-review.json')
            self.write_csv(manifest, rows)
            with contextlib.redirect_stdout(io.StringIO()):
                path = train(manifest, artifacts, root/'pilot', dict(self.config(), epochs=1, cache_images=True))
            model, checkpoint = load_unet(path)
            self.assertEqual(checkpoint['label_quality'], 'ai_visual_reviewed')
            self.assertEqual(checkpoint['review_methods'], {'ai_visual': 4})
            _, area, _ = predict_mask(model, checkpoint, Image.new('RGB', (48, 32)))
            self.assertIn('not expert-validated', area['quality_note'])
            provenance = json.loads((root/'pilot/training_manifest.json').read_text())
            self.assertTrue(all(r['review_method'] == 'ai_visual' for r in provenance))
            checkpoint['review_methods'] = {}
            torch.save(checkpoint, root/'bad.pt')
            with self.assertRaisesRegex(ValueError, 'provenance'):
                load_unet(root/'bad.pt')

    def test_minimum_epochs_prevents_premature_stopping(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, artifacts, _ = self.fixture(root)
            constant = dict(SegmentationMetrics().result(), leaf_dice=0.5, lesion_dice=0.1)
            config = dict(self.config(), epochs=5, min_epochs=4, patience=1)
            with patch.object(SegmentationMetrics, 'result', return_value=constant), contextlib.redirect_stdout(io.StringIO()):
                train(manifest, artifacts, root/'model', config)
            summary = json.loads((root/'model/summary.json').read_text())
            self.assertEqual(summary['epochs_run'], 4)
            for minimum in (0, 6, True, 1.5):
                with self.assertRaisesRegex(ValueError, 'min_epochs'):
                    validate_config(dict(config, min_epochs=minimum))

    def test_training_checkpoint_evaluation_and_prediction_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, artifacts, _ = self.fixture(root)
            # Test annotations can be unavailable during training: they must not be read.
            held_out = (manifest.parent/'4.png').read_bytes()
            (manifest.parent/'4.png').write_bytes(b'not readable while training')
            with contextlib.redirect_stdout(io.StringIO()):
                checkpoint_path = train(manifest, artifacts, root/'model', self.config())
            (manifest.parent/'4.png').write_bytes(held_out)
            model, checkpoint = load_unet(checkpoint_path)
            torch.manual_seed(42)
            initial = UNet(4)
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(initial.parameters(), model.parameters())))
            self.assertEqual(checkpoint['classes'], CLASSES)
            provenance = json.loads((root/'model/training_manifest.json').read_text())
            self.assertEqual({r['split'] for r in provenance}, {'train', 'validation'})
            result = evaluate(checkpoint_path, manifest, artifacts, root/'test.json')
            self.assertEqual(result['test']['images'], 2)
            area = export_prediction(checkpoint_path, root/'images/4.png', root/'prediction')
            self.assertTrue((root/'prediction/mask.png').is_file())
            self.assertIn('affected_area_percent', area)
            self.assertIsNone(area['pesticide_quantity'])
            rule = {'crop': 'rice', 'diseases': ['fixture'], 'region': 'test-region',
                    'product': 'TEST-ONLY', 'formulation': 'TEST-ONLY', 'dose': 2,
                    'unit': 'g/L', 'label_verified': True, 'source_url': 'https://example.invalid/test'}
            combined = export_prediction(checkpoint_path, root/'images/4.png', root/'combined',
                rule=rule, disease='fixture', region='test-region', tank_ml=100,
                application_authorized=True)
            self.assertEqual(combined['pesticide_quantity']['quantity'], 0.2)
            self.assertEqual(combined['pesticide_quantity']['calculation_basis'], 'spray_volume')
            self.assertEqual(combined['affected_area_percent'], area['affected_area_percent'])
            self.assertFalse(combined['pesticide_quantity']['affected_area_used_to_scale_dose'])
            saved = json.loads((root/'combined/area.json').read_text())
            self.assertEqual(saved['pesticide_status'], 'label_quantity_calculated')
            with self.assertRaises(ValueError):
                export_prediction(checkpoint_path, root/'images/4.png', root/'invalid',
                    rule=rule, disease='wrong', region='test-region', tank_ml=100,
                    application_authorized=True)
            self.assertFalse((root/'invalid').exists())
            with self.assertRaisesRegex(ValueError, 'output exists'):
                evaluate(checkpoint_path, manifest, artifacts, root/'test.json')
            checkpoint['label_quality'] = 'experimental_pseudo_labels'
            torch.save(checkpoint, root/'bad.pt')
            with self.assertRaisesRegex(ValueError, 'reviewed'):
                load_unet(root/'bad.pt')

    def test_100_ml_tank_conversion_and_disease_match(self):
        # Synthetic arithmetic only; deliberately not a real product recommendation.
        rule = {'crop': 'rice', 'diseases': ['fixture'], 'region': 'India/Tripura',
                'product': 'TEST-ONLY', 'formulation': 'TEST-ONLY', 'dose': 2,
                'unit': 'g/L', 'label_verified': True, 'source_url': 'https://example.invalid/test'}
        result = rice_tank_quantity(rule, disease='fixture', region='India/Tripura', tank_ml=100, application_authorized=True)
        self.assertAlmostEqual(result['quantity'], 0.2)
        self.assertEqual(result['tank_litres'], 0.1)
        self.assertFalse(result['affected_area_used_to_scale_dose'])
        with self.assertRaises(ValueError):
            rice_tank_quantity(rule, disease='another', region='India/Tripura', application_authorized=True)
        with self.assertRaises(ValueError):
            rice_tank_quantity(rule, disease='fixture', region='India/Tripura')
        area_rule = dict(rule, unit='g/ha')
        with self.assertRaises(ValueError):
            rice_tank_quantity(area_rule, disease='fixture', region='India/Tripura', application_authorized=True)
        per_area = rice_tank_quantity(area_rule, disease='fixture', region='India/Tripura',
                                     tank_ml=100, area_hectares=0.01, application_authorized=True)
        self.assertEqual(per_area['quantity'], 0.02)
        self.assertEqual(per_area['calculation_basis'], 'treated_area')
        with self.assertRaisesRegex(ValueError, 'not finite'):
            rice_tank_quantity(dict(rule, dose=1e308), disease='fixture', region='India/Tripura',
                               tank_ml=1e308, application_authorized=True)

    def test_disease_reference_does_not_invent_per_plant_dose(self):
        for disease in ('leaf_blast', 'brown_spot'):
            result = disease_product_reference(disease, water_ml=100)
            self.assertEqual(result['status'], 'conditional_label_reference')
            self.assertAlmostEqual(result['reference_quantity']['quantity'], 0.1)
            self.assertEqual(disease_product_reference(disease, water_ml=1000)['reference_quantity']['quantity'], 1)
            self.assertIsNone(result['required_quantity_per_plant'])
            self.assertFalse(result['application_authorized'])
            self.assertFalse(result['affected_area_used_to_scale_dose'])
        for disease in ('bacterial_leaf_blight', 'leaf_scald', 'narrow_brown_spot', 'unknown'):
            result = disease_product_reference(disease)
            self.assertEqual(result['status'], 'no_verified_label_in_catalog')
            self.assertIsNone(result['product'])
            self.assertIsNone(result['reference_quantity'])
        self.assertEqual(disease_product_reference('healthy')['status'], 'no_product_suggested')
        for volume in (0, -1, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                disease_product_reference('brown_spot', water_ml=volume)


if __name__ == '__main__':
    unittest.main()
