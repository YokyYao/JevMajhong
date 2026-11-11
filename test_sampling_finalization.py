from pathlib import Path
import tempfile
import unittest
from sampling_finalization import canonical_checkpoint


class FinalizationTests(unittest.TestCase):
    def test_relative_sdk_run(self):
        self.assertEqual(canonical_checkpoint('/tmp/project', 'runs/pilot/training-run', 'checkpoints/step-000064'),
                         'runs/pilot/training-run/checkpoints/step-000064')

    def test_absolute_sdk_run(self):
        self.assertEqual(canonical_checkpoint('/tmp/project', '/tmp/project/runs/pilot', 'checkpoints/step-000064'),
                         'runs/pilot/checkpoints/step-000064')

    def test_rejects_absolute_selected_or_outside_run(self):
        for selected in ('/tmp/project/elsewhere', '../../outside'):
            with self.assertRaises(ValueError):canonical_checkpoint('/tmp/project', 'runs/pilot', selected)
        with self.assertRaises(ValueError):canonical_checkpoint('/tmp/project', '/tmp/other', 'checkpoints/step')

    def test_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'project';run = root / 'runs/pilot'
            run.mkdir(parents=True);outside = Path(directory) / 'outside';outside.mkdir()
            (run / 'checkpoints').symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):canonical_checkpoint(root, run, 'checkpoints/step')


if __name__ == '__main__':
    unittest.main()
