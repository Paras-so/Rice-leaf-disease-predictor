"""Exercise photo selection, explicit analysis, and upload error recovery."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from rice_disease.data import DEFAULT_DATASET, ROOT


class UploadTests(unittest.TestCase):
    def test_missing_checkpoint_explains_how_to_restore_selected_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "artifacts").mkdir()
            selection = json.loads((ROOT / "artifacts/selected_model.json").read_text())
            (root / "artifacts/selected_model.json").write_text(json.dumps(selection))
            with patch("rice_disease.data.ROOT", root):
                app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.error), 1)
            self.assertIn("model file is missing", app.error[0].value)
            self.assertEqual(app.code[0].value, "artifacts/" + selection["checkpoint"])
            self.assertTrue(any("retraining is not required" in item.value for item in app.info))
            self.assertFalse(app.file_uploader)

    def test_missing_selection_explains_how_to_restore_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch("rice_disease.data.ROOT", Path(temporary)):
                app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertTrue(any("artifacts/selected_model.json" in item.value for item in app.info))
            self.assertFalse(app.file_uploader)

    def test_upload_analyze_and_replace_photo(self):
        with (ROOT / "artifacts/split_manifest.csv").open(newline="") as source:
            row = next(row for row in csv.DictReader(source)
                       if row["split"] == "validation" and row["class_name"] == "healthy")
        photo = DEFAULT_DATASET / row["path"]
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.selectbox), 0)
        self.assertEqual([tab.label for tab in app.tabs], ["Analyze a leaf", "Final model", "Dataset"])
        self.assertTrue(app.button[0].disabled)

        app.file_uploader[0].set_value(("leaf.JPG", photo.read_bytes(), "image/jpeg")).run()
        self.assertFalse(app.button[0].disabled)
        self.assertFalse(app.get("download_button"))
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertIn("Healthy", [item.value for item in app.subheader])
        self.assertEqual(len(app.get("download_button")), 1)

        # Unrelated reruns retain the result; replacing the file clears it.
        app.run()
        self.assertEqual(len(app.get("download_button")), 1)
        app.file_uploader[0].set_value(("replacement.jpeg", photo.read_bytes(), "image/jpeg")).run()
        self.assertFalse(app.get("download_button"))
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(len(app.get("download_button")), 1)
        app.file_uploader[0].clear().run()
        self.assertTrue(app.button[0].disabled)
        self.assertFalse(app.get("download_button"))

    def test_invalid_image_reports_error_without_crashing(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        app.file_uploader[0].set_value(("broken.jpg", b"not an image", "image/jpeg")).run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.error), 1)
        self.assertFalse(app.get("download_button"))


if __name__ == "__main__":
    unittest.main()
