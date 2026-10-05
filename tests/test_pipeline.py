"""Checks for split isolation, preprocessing, and checkpoint behavior."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from rice_disease.data import CLASSES, ROOT, load_manifest
from rice_disease.models import BUILDERS, create_model, load_checkpoint, remove_head, attach_head
from rice_disease.preprocessing import image_tensor
from rice_disease.segmentation import affected_area
from rice_disease.quantity import calculate_quantity
from rice_disease.binary import binary_metrics, health_status


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = json.loads((ROOT / "configs/benchmark.json").read_text())

    def test_split_is_disjoint_and_test_membership_preserved(self):
        rows, summary = load_manifest()
        self.assertEqual(len(rows), 1941)
        self.assertEqual(len({row["path"] for row in rows}), 1941)
        sets = {split: {r["pixel_sha256"] for r in rows if r["split"] == split}
                for split in ("train", "validation", "test")}
        self.assertEqual([len(sets[s]) for s in sets], [1241, 311, 389])
        self.assertFalse(sets["train"] & sets["validation"])
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["validation"] & sets["test"])
        self.assertTrue(all((r["split"] == "test") == (r["original_split"] == "test") for r in rows))
        self.assertEqual(summary["classes"], CLASSES)

    def test_preprocessing_handles_grayscale_and_is_reproducible(self):
        image = Image.fromarray(np.arange(10000, dtype=np.uint8).reshape(100, 100))
        plain = image_tensor(image, self.config)
        first = image_tensor(image, self.config, 42)
        self.assertEqual(tuple(plain.shape), (3, 224, 224))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.equal(first, image_tensor(image, self.config, 42)))
        self.assertTrue(torch.equal(plain, image_tensor(image, self.config)))
        self.assertFalse(torch.equal(plain, first))

    def test_binary_metrics_count_disease_class_confusion_as_binary_correct(self):
        # Two disease classes confused with each other still mean diseased.
        result = binary_metrics([[8, 1, 2], [3, 20, 1], [4, 2, 9]],
                                ["blast", "healthy", "brown_spot"])
        self.assertEqual(result["confusion_matrix"], [[20, 4], [3, 23]])
        self.assertAlmostEqual(result["accuracy"], 43 / 50)
        self.assertAlmostEqual(result["disease_recall"], 23 / 26)
        self.assertEqual(health_status("healthy"), "healthy")
        self.assertEqual(health_status("leaf_blast"), "diseased")

    def test_manifest_edit_is_detected(self):
        with tempfile.TemporaryDirectory(prefix="rice-manifest-test-") as temporary:
            folder = Path(temporary)
            for filename in ("split_summary.json", "split_manifest.csv"):
                (folder / filename).write_bytes((ROOT / "artifacts" / filename).read_bytes())
            manifest = folder / "split_manifest.csv"
            manifest.write_bytes(manifest.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                load_manifest(folder)

    def test_every_architecture_has_six_outputs_and_only_head_is_trainable(self):
        for name in BUILDERS:
            with self.subTest(model=name):
                model, head = create_model(name, pretrained=False)
                model.eval()
                self.assertEqual({id(p) for p in model.parameters() if p.requires_grad},
                                 {id(p) for p in head.parameters()})
                with torch.no_grad():
                    output = model(torch.zeros(1, 3, 224, 224))
                self.assertEqual(tuple(output.shape), (1, 6))

    def test_feature_head_matches_full_model_and_checkpoint_roundtrip(self):
        model, _ = create_model("mobilenet_v3_small", pretrained=False)
        model.eval()
        image = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            expected = model(image)
            head = remove_head(model, "mobilenet_v3_small")
            actual = head(model(image))
            attach_head(model, "mobilenet_v3_small", head)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-6))
        with tempfile.TemporaryDirectory(prefix="rice-model-test-") as temporary:
            path = Path(temporary) / "model.pt"
            torch.save({"architecture": "mobilenet_v3_small", "classes": CLASSES,
                        "state_dict": model.state_dict(), "config": self.config}, path)
            loaded, checkpoint = load_checkpoint(path)
            with torch.no_grad():
                self.assertTrue(torch.allclose(expected, loaded(image), atol=1e-6))
            self.assertEqual(checkpoint["classes"], CLASSES)

    def test_affected_area_excludes_background_and_rejects_invalid_masks(self):
        result = affected_area(np.array([[0, 0, 1], [1, 2, 2]]))
        self.assertEqual(result["affected_area_percent"], 50.0)
        with self.assertRaises(ValueError):
            affected_area(np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            affected_area(np.array([[0, 255]]))

    def test_quantity_requires_verified_rule_and_matching_units(self):
        # Synthetic arithmetic fixture, deliberately not an agricultural recommendation.
        rule = {"label_verified": True, "source_url": "https://example.invalid/test-label",
                "region": "test-region", "product": "TEST-ONLY", "formulation": "TEST-ONLY",
                "dose": 2.0, "unit": "g/L"}
        result = calculate_quantity(rule, region="test-region", application_authorized=True, spray_litres=10)
        self.assertEqual(result["quantity"], 20)
        self.assertEqual(result["unit"], "g")
        for bad in (None, dict(rule, label_verified=False), dict(rule, dose=float("nan"))):
            with self.assertRaises(ValueError):
                calculate_quantity(bad, region="test-region", application_authorized=True, spray_litres=10)
        with self.assertRaises(ValueError):
            calculate_quantity(rule, region="another-region", application_authorized=True, spray_litres=10)
        with self.assertRaises(ValueError):
            calculate_quantity(rule, region="test-region", application_authorized=False, spray_litres=10)
        with self.assertRaises(ValueError):
            calculate_quantity(rule, region="test-region", application_authorized=True, area_hectares=1)


if __name__ == "__main__":
    unittest.main()
