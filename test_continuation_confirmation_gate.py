import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from call_policy import digest
from continuation_confirmation_gate import choose_confirmation, confirmation_decision


class ContinuationConfirmationTests(unittest.TestCase):
    def audits(self, old_gain=1., parent_gain=.1):
        old = {'status': 'passed_complete', 'stage': 'development', 'games': 768,
            'independent_wall_streams': 32,
            'overall_paired_gain': {'gain': old_gain, 'ci95': [-100., 100.],
                                  'independent_wall_streams': 32}}
        parent = {'status': 'passed_complete', 'games': 768, 'new_parent_games': 384,
            'reused_candidate_games': 384, 'candidate_games_source_verified': True,
            'source_identity_passed': True, 'native_parity_passed': True,
            'summary_recomputed': True,
            'overall_parent_paired_gain': {'gain': parent_gain, 'ci95': [-100., 100.],
                                          'independent_wall_streams': 32}}
        return old, parent

    def test_small_positive_crossing_zero_is_preserved(self):
        old, parent = self.audits(parent_gain=.0001)
        self.assertEqual(choose_confirmation(old, parent, seeds=32)['decision'], 'confirm')

    def test_oldB_gain_does_not_override_parent_regression(self):
        for gain in (0., -1.):
            old, parent = self.audits(old_gain=10., parent_gain=gain)
            self.assertEqual(choose_confirmation(old, parent, seeds=32)['decision'],
                             'skip_nonpositive_parent')

    def test_nonpositive_oldB_skips_without_parent_games(self):
        old, _ = self.audits(old_gain=0.)
        self.assertEqual(choose_confirmation(old, None, seeds=32)['decision'],
                         'skip_nonpositive_oldB')

    def test_positive_requires_parent_completion(self):
        old, parent = self.audits()
        with self.assertRaises(ValueError):
            choose_confirmation(old, None, seeds=32)
        for key, value in [('games', 767), ('new_parent_games', 383),
                           ('candidate_games_source_verified', False),
                           ('source_identity_passed', False), ('status', 'running')]:
            modified = {**parent, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                choose_confirmation(old, modified, seeds=32)

    def test_invalid_statistics_and_wall_count_fail(self):
        for key in ('old', 'parent'):
            for value in (math.nan, math.inf, True):
                old, parent = self.audits()
                audit, field = (old, 'overall_paired_gain') if key == 'old' else (parent, 'overall_parent_paired_gain')
                audit[field]['gain'] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    choose_confirmation(old, parent, seeds=32)
        old, parent = self.audits()
        with self.assertRaises(ValueError):
            choose_confirmation(old, parent, seeds=31)
        parent['overall_parent_paired_gain']['independent_wall_streams'] = 31
        with self.assertRaises(ValueError):
            choose_confirmation(old, parent, seeds=32)

    def test_check_does_not_mutate_original_audits(self):
        old, parent = self.audits()
        before = copy.deepcopy((old, parent))
        choose_confirmation(old, parent, seeds=32)
        self.assertEqual((old, parent), before)

    def test_source_protocol_and_parent_records_must_remain_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'dev').mkdir()
            (root / 'parent').mkdir()
            source = {'development_seed_start': 100, 'development_seeds': 32,
                      'confirmation_seed_start': 1000, 'confirmation_seeds': 128}
            config = {'stage': 'independent_confirmation', 'source_config': 'train.json',
                      'out': 'confirm', 'development_out': 'dev', 'seed_start': 1000, 'seeds': 128}
            dev_config = {'source_config': 'train.json', 'out': 'dev', 'stage': 'development',
                          'seed_start': 100, 'seeds': 32, 'paired_parent_out': 'parent'}
            old, parent = self.audits()
            games = root / 'parent/games-private.jsonl'
            games.write_text('original audited bytes\n')
            parent['raw_parent_games_sha256'] = digest(games)
            expected = {'candidate_files': {'head': 'frozen-candidate'},
                        'parent_files': {'head': 'frozen-parent'}}
            documents = {'train.json': source, 'dev/protocol.json': {'config': dev_config},
                         'dev/evaluation-audit.json': old, 'parent/evaluation-audit.json': parent,
                         'parent/protocol.json': expected,
                         'parent/summary.json': {'overall': {
                             'candidate_minus_reference': parent['overall_parent_paired_gain']}}}
            for name, value in documents.items():
                (root / name).write_text(json.dumps(value))
            with patch('scripts.evaluate_feature_parent.registration',
                       return_value=(expected, {}, {})) as verify:
                self.assertEqual(confirmation_decision(root, config)['decision'], 'confirm')
                verify.assert_called_once_with(dev_config)
                games.write_text('changed records\n')
                with self.assertRaisesRegex(ValueError, 'parent records changed'):
                    confirmation_decision(root, config)
                games.write_text('original audited bytes\n')
                (root / 'parent/protocol.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'source or weights changed'):
                    confirmation_decision(root, config)
                with self.assertRaisesRegex(ValueError, 'walls changed'):
                    confirmation_decision(root, {**config, 'seed_start': 1001})


if __name__ == '__main__':
    unittest.main()
