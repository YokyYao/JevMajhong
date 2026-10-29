import copy
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch

from scripts import audit_score_confirmation as auditor
from scripts.audit_score_confirmation import audit_rows, compare


class ConfirmationAuditTest(unittest.TestCase):
    def setUp(self):
        self.config = {'confirmation_seed_start': 10, 'confirmation_seeds': 3,
                       'reward_scale': 1000.0, 'target_mean_net_points': 0,
                       'target_net_points_gain': 500}
        self.reference = [dict(seed=seed, engine_seed=seed << 32, seed_scheme='spaced-v1',
                               seat=seat, rank=2, score=25000, game_mode='4p-red-east',
                               reward_mode='net_points', reward_scale=1000.0, reward=0.0,
                               sampling_seed=None, trainable_decisions=0, old_opponent=False)
                          for seed in range(10, 13) for seat in range(4)]

    def shifted(self, gains):
        rows = copy.deepcopy(self.reference)
        for row in rows:
            row['score'] += gains[row['seed'] - 10]
            row['reward'] = (row['score'] - 25000) / 1000
        return rows

    def test_clustered_interval_uses_three_streams_not_twelve_seats(self):
        from scipy.stats import t
        result = compare(self.shifted([100, 700, 1000]), self.reference, self.config)
        self.assertEqual(result['primary_delta'], 600)
        half = float(t.ppf(.975, 2)) * statistics.stdev([100, 700, 1000]) / 3**.5
        self.assertAlmostEqual(result['primary_delta_ci95'][0], 600 - half)
        self.assertFalse(result['qualified'])  # Positive mean alone is insufficient.

    def test_improvement_requires_effect_size_and_positive_points(self):
        self.assertTrue(compare(self.shifted([600, 700, 800]), self.reference, self.config)['qualified'])
        self.assertFalse(compare(self.shifted([100, 100, 100]), self.reference, self.config)['qualified'])
        self.config['target_mean_net_points'] = 1000
        self.assertFalse(compare(self.shifted([600, 700, 800]), self.reference, self.config)['qualified'])

    def test_duplicate_missing_and_foreign_streams_rejected(self):
        for rows in (self.reference[:-1], self.reference + [self.reference[0]],
                     self.reference[:-1] + [self.reference[0]]):
            with self.subTest(records=len(rows)), self.assertRaises(ValueError):
                audit_rows(rows, self.config)

    def test_changed_engine_reward_or_sampling_rejected(self):
        for key, value in [('engine_seed', 10), ('reward', 10), ('game_mode', '4p-red-half'),
                           ('sampling_seed', 20), ('trainable_decisions', 1), ('old_opponent', True)]:
            rows = copy.deepcopy(self.reference)
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit_rows(rows, self.config)

    def test_full_audit_verifies_weights_code_protocol_and_recorded_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / 'run'
            run.mkdir()
            config = {**self.config, 'reference': 'reference', 'game_mode': '4p-red-east',
                      'reward_mode': 'net_points', 'seed_scheme': 'spaced-v1',
                      'development_seed_start': 100, 'development_seeds': 2}
            for directory in ['reference', 'candidate']:
                (root / directory).mkdir()
                (root / directory / 'head.pt').write_bytes(directory.encode())
                (root / directory / 'adapter_model.safetensors').write_bytes(b'frozen')
            code = root / 'model.py'
            code.write_text('frozen inference code')
            protocol = {'config': config,
                        'sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                        'code_sha256': {'model.py': auditor.sha256(code)},
                        'reference_adapter_sha256': auditor.sha256(root / 'reference/adapter_model.safetensors'),
                        'reference_head_sha256': auditor.sha256(root / 'reference/head.pt')}
            selected = {'config': config, 'checkpoint': 'candidate'}
            (run / 'protocol.json').write_text(json.dumps(protocol))
            (run / 'confirmation-selection.json').write_text(json.dumps(selected))
            (run / 'training-games-stage-01.json').write_text(json.dumps([{'seed': 1}]))
            candidate = self.shifted([600, 700, 800])
            result = compare(candidate, self.reference, config)
            recorded = copy.deepcopy({**result, 'status': 'complete', 'checkpoint': 'candidate',
                                      'default_policy_changed': False, 'primary_metric': 'mean_net_points',
                                      'paired': {'primary_delta': result['primary_delta']}})
            result_file = run / 'confirmation-result.json'
            result_file.write_text(json.dumps(recorded))
            for name, rows in [('reference', self.reference), ('candidate', candidate)]:
                directory = run / f'confirmation-{name}'
                directory.mkdir()
                (directory / 'games.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            head_sha = auditor.sha256(root / 'candidate/head.pt')
            with patch.object(auditor, 'ROOT', root):
                self.assertTrue(auditor.audit(run, head_sha)['qualified'])
                recorded['candidate']['mean_net_points'] += 1000
                result_file.write_text(json.dumps(recorded))
                with self.assertRaisesRegex(ValueError, 'differs from raw outcomes'):
                    auditor.audit(run, head_sha)
                result_file.write_text(json.dumps({**recorded, 'candidate': result['candidate']}))
                code.write_text('changed inference code')
                with self.assertRaisesRegex(ValueError, 'Numerical code changed'):
                    auditor.audit(run, head_sha)


if __name__ == '__main__':
    unittest.main()
