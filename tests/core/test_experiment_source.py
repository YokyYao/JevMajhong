import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import run_experiment


class ExperimentSourceTests(unittest.TestCase):
    def test_controlled_rerun_uses_old_package_and_cannot_change_existing_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);package = root / 'jevmahjong';package.mkdir()
            module = package / '__main__.py';module.write_text('version = 1\n')
            old = run_experiment.freeze_source(root)
            module.write_text('version = 2\n');current = run_experiment.freeze_source(root)
            config = root / 'config.json';config.write_text(json.dumps({'out': 'runs/controlled'}))
            args = ['run_experiment', 'train-context', '--config', str(config)]
            with patch.object(run_experiment, 'ROOT', root), patch.object(run_experiment.subprocess, 'run') as invoke:
                with patch('sys.argv', args + ['--prepare-only', '--source-sha', old.name]), contextlib.redirect_stdout(io.StringIO()):
                    run_experiment.main()
                records = list((root / 'runs/launches').glob('*.json'))
                record = next(p for p in records if not p.name.endswith('-config.json'))
                self.assertEqual(json.loads(record.read_text())['source'], str(old))
                self.assertEqual((old / 'jevmahjong/__main__.py').read_text(), 'version = 1\n')
                with patch('sys.argv', args + ['--registered']):
                    run_experiment.main()
                self.assertEqual(invoke.call_args.args[0][3], str(old))
                invoke.reset_mock()
                with patch('sys.argv', args + ['--registered', '--source-sha', current.name]):
                    with self.assertRaisesRegex(ValueError, 'Registered source identity differs'):
                        run_experiment.main()
                invoke.assert_not_called()
                (old / 'jevmahjong/__main__.py').write_text('tampered = True\n')
                with patch('sys.argv', args + ['--registered']):
                    with self.assertRaisesRegex(ValueError, 'Original source snapshot changed'):
                        run_experiment.main()

    def test_missing_invalid_or_falsely_named_snapshot_never_registers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);config = root / 'config.json'
            config.write_text(json.dumps({'out': 'runs/candidate'}))
            base = ['run_experiment', 'train-context', '--config', str(config), '--prepare-only', '--source-sha']
            with patch.object(run_experiment, 'ROOT', root):
                for identity in ('../outside', '0' * 64):
                    with self.subTest(identity=identity), patch('sys.argv', base + [identity]):
                        with self.assertRaises(ValueError):
                            run_experiment.main()
                fake = root / 'runs/code' / ('0' * 64) / 'jevmahjong';fake.mkdir(parents=True)
                (fake / '__init__.py').write_text('different content\n')
                with patch('sys.argv', base + ['0' * 64]):
                    with self.assertRaisesRegex(ValueError, 'snapshot changed'):
                        run_experiment.main()
                self.assertEqual(list((root / 'runs/launches').glob('*.json')), [])


if __name__ == '__main__':
    unittest.main()
