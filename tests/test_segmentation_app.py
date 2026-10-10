"""Exercise the U-Net area display with an explicit synthetic inference fixture."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from streamlit.testing.v1 import AppTest

from rice_disease.data import ROOT


class SegmentationAppTests(unittest.TestCase):
    def test_disease_reference_updates_quantity_without_field_area_input(self):
        uploaded = io.BytesIO()
        Image.new('RGB', (80, 80), 'green').save(uploaded, format='PNG')
        prediction = {'status': 'diseased', 'disease': 'brown_spot', 'softmax_score': 0.8,
                      'confidence_note': 'Synthetic test prediction', 'scores': {'brown_spot': 0.8},
                      'affected_area_percent': None}
        with patch('rice_disease.inference.predict', return_value=prediction):
            app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=60).run()
            app.file_uploader[0].set_value(('fixture.png', uploaded.getvalue(), 'image/png')).run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertTrue(any('0.1 mL product for 100 mL water' in item.value for item in app.markdown))
            self.assertEqual([item.label for item in app.number_input], ['Water for dilution reference (mL)'])
            app.number_input[0].set_value(200).run()
            self.assertFalse(app.exception)
            self.assertTrue(any('0.2 mL product for 200 mL water' in item.value for item in app.markdown))
            self.assertEqual(len(app.get('download_button')), 1)

    def test_checkpoint_branch_displays_area_and_preserves_download(self):
        class SyntheticMaskModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def forward(self, inputs):
                logits = inputs.new_zeros((len(inputs), 3, *inputs.shape[-2:])) + self.anchor
                logits[:, 1, :, :inputs.shape[-1]//2] = 10
                logits[:, 2, :, inputs.shape[-1]//2:] = 10
                return logits
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary)/'synthetic-test-only.pt'
            marker.touch()
            script = (ROOT/'app.py').read_text(encoding='utf-8').replace(
                'ROOT / "artifacts/unet/model.pt"', f'Path({str(marker)!r})')
            uploaded = io.BytesIO()
            Image.new('RGB', (160, 80), 'green').save(uploaded, format='PNG')
            checkpoint = {'config': {'image_size': 32}, 'label_quality': 'ai_visual_reviewed',
                          'train_images': 2, 'validation_images': 2,
                          'quality_note': 'Synthetic AI-reviewed test fixture'}
            with patch('rice_disease.unet.load_unet', return_value=(SyntheticMaskModel(), checkpoint)):
                app = AppTest.from_string(script, default_timeout=60).run()
                app.file_uploader[0].set_value(('synthetic.png', uploaded.getvalue(), 'image/png')).run()
                app.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                area_metrics = [item.value for item in app.metric if item.label == 'Estimated affected leaf area']
                self.assertEqual(area_metrics, ['50.00%'])
                self.assertTrue(any('Experimental U-Net pilot' in item.value for item in app.info))
                self.assertEqual(len(app.get('download_button')), 1)
                self.assertTrue(any('100 mL (0.1 L)' in item.value for item in app.caption))


if __name__ == '__main__':
    unittest.main()
