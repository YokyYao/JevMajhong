import json
from pathlib import Path
import tempfile
import unittest

from tools.audit_endgame_labels import audit
from jevmahjong.endgame.collect import digest, write
from jevmahjong.endgame.context import placements, terminal_utility
from jevmahjong.endgame.coverage import public_stratum
from jevmahjong.endgame.win_value import settlement_request


class LabelAuditTests(unittest.TestCase):
    def fixture(self, directory):
        payload = settlement_request({'state': {'seat': 1, 'dealer': 3, 'round_wind': 0,
            'hand_number': 4, 'honba': 0, 'sticks': 0, 'scores': [32000, 22000, 25000, 21000],
            'hand': '2m 3m 4p 5p 6p 6s 7s 8s 2p 2p 5z 5z 5z', 'dora_indicators': '9m', 'riichi': [],
            'rivers': ['', '', '', ''], 'melds': [[], [], [], []],
            'last_discard': {'seat': 2, 'tile': '4m'}},
            'questions': {'action': {'type': 'choice', 'criteria': {'win': {'type': 'hora'}, 'pass': {'type': 'none'}}}}})
        stratum = public_stratum(payload)
        rows = []
        for seed, continuation in ((1, [30000, 26000, 23000, 21000]), (2, [32000, 20000, 27000, 21000])):
            row = {'seed': seed, 'step': 7, 'seat': 1, 'request_sha256': digest(payload),
                   'group_id': f'source:{seed}', 'split': 'train', 'coverage_stratum': stratum,
                   'public_request': payload}
            rows.append(row)
            def outcome(scores):
                ranks = placements(scores)
                return {'scores': scores, 'ranks': ranks, 'utility': terminal_utility(ranks[1]),
                        'ledger_passed': True, 'hand_history': {'final_scores': scores}}
            label = {'status': 'complete_terminal_menu', 'source_request_sha256': row['request_sha256'],
                     'continued_until_native_match_done': True, 'worlds_per_action': 16,
                     'actual_actions': 2, 'record': payload,
                     'outcomes': [[outcome([32000, 23300, 23700, 21000])] * 16, [outcome(continuation)] * 16]}
            rid = digest([seed, 7, 1, row['request_sha256']])
            write(Path(directory) / 'labels' / (rid + '.json'), label)
        write(Path(directory) / 'selected-positions-private.json', rows)
        write(Path(directory) / 'manifest-status-private.json', {'positions_sha256': digest(rows)})
        return {'out': directory, 'future_worlds': 16}, rows

    def test_paired_audit_retains_both_beneficial_and_harmful_continuations(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _ = self.fixture(directory)
            result = audit(config)
            self.assertEqual(result['training_complete'], 2)
            self.assertEqual(result['counts']['positive_best_finite_mean'], 1)
            self.assertEqual(result['counts']['negative_best_finite_mean'], 1)
            self.assertEqual(result['counts']['complete_terminal_rollouts_audited'], 64)

    def test_missing_training_label_cannot_be_reported_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            config, rows = self.fixture(directory)
            row = rows[0]
            rid = digest([row['seed'], row['step'], row['seat'], row['request_sha256']])
            (Path(directory) / 'labels' / (rid + '.json')).unlink()
            with self.assertRaises(ValueError):
                audit(config)
            self.assertEqual(audit(config, partial=True)['status'], 'partial_train_label_audit')


if __name__ == '__main__':
    unittest.main()
