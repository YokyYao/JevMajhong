import copy
import unittest

from scripts.audit_single_decision import audit_records
from scripts.train_single_decision import intervention_advantages
from test_single_decision import RuleTeacher


class SingleDecisionAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = RuleTeacher()
        cls.policy.config.update(training_seed_start=3400002, training_seeds_per_stage=1, training_replicas=4,
                                 advantage_scale=10.)
        cls.controls, cls.games, cls.rows = [], [], []
        for seat in range(4):
            control, _ = cls.policy.game(3400002, seat, frozen=True, replica=-1)
            cls.controls.append(control)
            for replica in range(4):
                game, rows = cls.policy.game(3400002, seat, training=True, replica=replica)
                cls.games.append(game); cls.rows.extend(rows)
        intervention_advantages(cls.controls, cls.games, cls.rows, 4, 10.)

    def audit(self, games=None, rows=None):
        return audit_records(self.games if games is None else games, self.controls,
                             self.rows if rows is None else rows, self.policy.config, self.policy.reference)

    def test_genuine_simulation_records_recompute(self):
        report = self.audit()
        self.assertEqual(report['verified_sampling_replays'], 16)
        self.assertEqual(report['controls'], 4)
        self.assertEqual(report['positive_point_deltas'] + report['negative_point_deltas'] + report['zero_point_deltas'], 16)
        self.assertFalse(report['performance_claim'])

    def test_duplicate_training_decision_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'only one training decision'):
            self.audit(rows=self.rows + [self.rows[0]])

    def test_terminal_return_misattribution_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['advantage'] += .1
        with self.assertRaisesRegex(ValueError, 'advantage differs'):
            self.audit(rows=rows)

    def test_sampling_and_reuse_corruption_are_rejected(self):
        games = copy.deepcopy(self.games)
        games[0]['sampling_seed'] += 1
        with self.assertRaisesRegex(ValueError, 'sampling seeds'):
            self.audit(games=games)
        games = copy.deepcopy(self.games)
        game = next(g for g in games if g['sampled_action'] != g['teacher_action'])
        game['reused_control_outcome'] = True
        with self.assertRaisesRegex(ValueError, 'control reuse'):
            self.audit(games=games)


if __name__ == '__main__':
    unittest.main()
