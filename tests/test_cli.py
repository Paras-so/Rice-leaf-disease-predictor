"""Regression checks for visible CLI output and configuration-driven exports."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from rice_disease.data import ROOT
from rice_disease.preprocessing import image_tensor, main, resized_image


class PreprocessingCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs/benchmark.json").read_text())
        torch.set_num_threads(2)

    def test_size_comes_from_config_and_invalid_values_are_rejected(self):
        image = Image.new("RGBA", (73, 29), (80, 100, 120, 255))
        config = dict(self.config, image_size=32)
        tensor = image_tensor(image, config)
        self.assertEqual(tuple(tensor.shape), (3, 32, 32))
        expected = (torch.tensor([80, 100, 120]) / 255 - torch.tensor(config["mean"])) / torch.tensor(config["std"])
        torch.testing.assert_close(tensor[:, 0, 0], expected)
        for size in (0, -1, True, 22.4, None):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "image_size"):
                image_tensor(image, dict(config, image_size=size))
        with self.assertRaisesRegex(ValueError, "std"):
            image_tensor(image, dict(config, std=[0, 1, 1]))
        with self.assertRaisesRegex(ValueError, "JSON object"):
            image_tensor(image, [])

    def test_exif_orientation_is_applied_before_resize(self):
        image = Image.new("RGB", (8, 4), "red")
        for x in range(4, 8):
            for y in range(4):
                image.putpixel((x, y), (0, 0, 255))
        image.getexif()[274] = 6
        result = resized_image(image, dict(self.config, image_size=8))
        self.assertEqual(result.getpixel((4, 0)), (255, 0, 0))
        self.assertEqual(result.getpixel((4, 7)), (0, 0, 255))

    def test_folder_export_preserves_sources_and_reports_corrupt_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input" / "class_a"
            source.mkdir(parents=True)
            Image.new("L", (45, 31), 128).save(source / "leaf.png")
            Image.new("RGB", (19, 28), "green").save(source / "leaf.jpg")
            (source / "broken.png").write_bytes(b"not an image")
            originals = {p: p.read_bytes() for p in source.iterdir()}
            output = root / "output"
            console = io.StringIO()
            with contextlib.redirect_stdout(console):
                status = main(["--input", str(source.parent), "--output", str(output), "--image-size", "40"])
            self.assertEqual(status, 1)
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual((report["processed"], report["failed"]), (2, 1))
            self.assertIn("40x40", console.getvalue())
            self.assertIn("Summary:", console.getvalue())
            for record in report["images"]:
                self.assertEqual(record["tensor_shape"], [3, 40, 40])
                with Image.open(record["output"]) as image:
                    self.assertEqual(image.size, (40, 40))
                    self.assertEqual(image.mode, "RGB")
            self.assertEqual(len(list((output / "images/class_a").glob("*.png"))), 2)
            self.assertTrue(all(path.read_bytes() == data for path, data in originals.items()))

    def test_nested_output_and_empty_folder_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input"
            source.mkdir()
            for output in (source / "output", root, root / "output"):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    main(["--input", str(source), "--output", str(output)])
                self.assertEqual(error.exception.code, 1)

    def test_direct_and_module_execution_export_images_from_another_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "leaf.png"
            Image.new("RGB", (28, 44), "green").save(source)
            for direct in (True, False):
                command = [sys.executable, str(ROOT / "rice_disease/preprocessing.py")] if direct else [sys.executable, "-m", "rice_disease.preprocessing"]
                output = root / ("direct" if direct else "module")
                result = subprocess.run(command + ["--input", str(source), "--output", str(output)],
                                        cwd=root if direct else ROOT, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("224x224", result.stdout)
                self.assertTrue((output / "summary.json").is_file())

    def test_all_executable_package_files_support_direct_help(self):
        with tempfile.TemporaryDirectory() as temporary:
            for path in sorted((ROOT / "rice_disease").glob("*.py")):
                if path.name == "__init__.py":
                    continue
                with self.subTest(module=path.stem):
                    result = subprocess.run([sys.executable, str(path), "--help"], cwd=temporary,
                                            capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
