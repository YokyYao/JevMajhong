"""Settings from an earlier/minimal installation must retain a usable default model."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from desktop.jev_sidecar import app


class MinimalSettings(unittest.TestCase):
    def test_missing_empty_or_null_checkpoint_keeps_default(self):
        for checkpoint in ('missing', '', None):
            with self.subTest(checkpoint=checkpoint), tempfile.TemporaryDirectory() as folder:
                config = Path(folder) / 'settings.json'
                data = {'regions': {}, 'upgrade_calibration_sentinel': 'keep-me'}
                if checkpoint != 'missing':
                    data['checkpoint'] = checkpoint
                config.write_text(json.dumps(data), encoding='utf-8')
                monitor = Mock()
                monitor.count.return_value = 1
                panel = SimpleNamespace(monitor=monitor, checkpoint=Mock(), detector_path=Mock())
                with patch.object(app, 'CONFIG', config):
                    app.Sidecar.load_settings(panel)
                panel.checkpoint.setText.assert_not_called()
                self.assertEqual(panel.regions, {})
                self.assertEqual(json.loads(config.read_text()), data)


if __name__ == '__main__':
    unittest.main()
