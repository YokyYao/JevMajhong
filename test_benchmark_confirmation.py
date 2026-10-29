import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_benchmark_confirmation import fingerprint, prepare


class BenchmarkConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / 'runs/source'
        self.source.mkdir(parents=True)
        self.out = self.root / 'runs/final'
        self.checkpoint = self.source / 'checkpoints/update-000004'
        self.checkpoint.mkdir(parents=True)
        self.benchmark = self.root / 'sft2000'
        self.benchmark.mkdir()
        for directory in [self.checkpoint, self.benchmark]:
            (directory / 'head.pt').write_bytes(directory.name.encode())
            for name in ['adapter_config.json', 'adapter_model.safetensors', 'tokenizer.json',
                         'tokenizer_config.json', 'chat_template.jinja']:
                (directory / name).write_text('same frozen file ' + name)
        (self.checkpoint / 'complete.json').write_text('{}')
        (self.root / 'head_rl.py').write_text('fixed numerical code')
        self.config = {'reference': 'anchor8', 'out': str(self.source),
                       'reward_mode': 'net_points', 'game_mode': '4p-red-east',
                       'seed_scheme': 'spaced-v1', 'development_seed_start': 3102000,
                       'development_seeds': 32, 'confirmation_seed_start': 3202000,
                       'confirmation_seeds': 128, 'target_net_points_gain': 250,
                       'target_mean_net_points': 0, 'final_benchmark_reference': 'sft2000',
                       'final_benchmark_head_sha256': fingerprint(self.benchmark / 'head.pt'),
                       'final_confirmation_target_net_points_gain': 500,
                       'final_confirmation_target_mean_net_points': 0}
        self.relative_checkpoint = str(self.checkpoint.relative_to(self.root))
        (self.source / 'training-games-stage-01.json').write_text(json.dumps([{'seed': 3001000}]))
        (self.source / 'pilot-analysis.json').write_text(json.dumps({
            'selected_for_confirmation': {'checkpoint': self.relative_checkpoint, 'decision': 'target_reached'}}))
        self.write_protocol()

    def tearDown(self):
        self.temporary.cleanup()

    def write_protocol(self):
        protocol = {'config': self.config,
                    'sha256': hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest(),
                    'code_tree_sha': 'fixture',
                    'code_sha256': {'head_rl.py': fingerprint(self.root / 'head_rl.py')}}
        (self.source / 'protocol.json').write_text(json.dumps(protocol))
        selection = {'config': self.config, 'checkpoint': self.relative_checkpoint,
                     'candidate_head_sha256': fingerprint(self.checkpoint / 'head.pt'), 'development_update': 4}
        (self.source / 'confirmation-selection.json').write_text(json.dumps(selection))

    def test_final_benchmark_and_threshold_replace_training_phase_reference(self):
        result = prepare(self.source, self.out, self.root)
        self.assertEqual(result['training_reference'], 'anchor8')
        self.assertEqual(result['final_benchmark'], 'sft2000')
        protocol = json.loads((self.out / 'protocol.json').read_text())
        self.assertEqual(protocol['config']['target_net_points_gain'], 500)
        self.assertEqual(protocol['reference_head_sha256'], fingerprint(self.benchmark / 'head.pt'))
        self.assertEqual(protocol['config']['reference'], 'sft2000')
        self.assertEqual(protocol['config']['confirmation_seed_start'], 3202000)
        self.assertTrue((self.out / 'source-training-protocol.json').exists())
        self.assertTrue((self.out / 'training-games-stage-01.json').exists())

    def test_cannot_lower_final_gate_to_phase_target(self):
        self.config['final_confirmation_target_net_points_gain'] = 250
        self.write_protocol()
        with self.assertRaisesRegex(ValueError, 'cannot be weakened'):
            prepare(self.source, self.out, self.root)
        self.assertFalse(self.out.exists())

    def test_changed_selected_head_or_original_benchmark_is_rejected(self):
        original_head = (self.checkpoint / 'head.pt').read_bytes()
        (self.checkpoint / 'head.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Selected head changed'):
            prepare(self.source, self.out, self.root)
        (self.checkpoint / 'head.pt').write_bytes(original_head)
        (self.benchmark / 'head.pt').write_bytes(b'changed benchmark')
        with self.assertRaisesRegex(ValueError, 'original benchmark changed'):
            prepare(self.source, self.out, self.root)

    def test_changed_frozen_adapter_is_rejected(self):
        (self.checkpoint / 'adapter_model.safetensors').write_bytes(b'different adapter')
        with self.assertRaisesRegex(ValueError, 'frozen input differs'):
            prepare(self.source, self.out, self.root)

    def test_test_streams_cannot_overlap_training(self):
        (self.source / 'training-games-stage-01.json').write_text(json.dumps([{'seed': 3202000}]))
        with self.assertRaisesRegex(ValueError, 'streams overlap'):
            prepare(self.source, self.out, self.root)

    def test_existing_confirmation_is_never_overwritten(self):
        prepare(self.source, self.out, self.root)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            prepare(self.source, self.out, self.root)


if __name__ == '__main__':
    unittest.main()
