"""Exercise photo selection, explicit analysis, and upload error recovery."""

import csv
import unittest

from streamlit.testing.v1 import AppTest

from rice_disease.data import DEFAULT_DATASET, ROOT


class UploadTests(unittest.TestCase):
    def test_upload_analyze_and_replace_photo(self):
        with (ROOT / "artifacts/split_manifest.csv").open(newline="") as source:
            row = next(row for row in csv.DictReader(source)
                       if row["split"] == "validation" and row["class_name"] == "healthy")
        photo = DEFAULT_DATASET / row["path"]
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.button[0].disabled)

        app.file_uploader[0].set_value(("leaf.JPG", photo.read_bytes(), "image/jpeg")).run()
        self.assertFalse(app.button[0].disabled)
        self.assertFalse(app.get("download_button"))
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertIn("Healthy", [item.value for item in app.subheader])
        self.assertEqual(len(app.get("download_button")), 1)

        # Unrelated reruns retain the result; changing the model or file clears it.
        app.run()
        self.assertEqual(len(app.get("download_button")), 1)
        app.selectbox[0].select(app.selectbox[0].options[1]).run()
        self.assertFalse(app.get("download_button"))
        app.button[0].click().run()
        self.assertFalse(app.exception)
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
