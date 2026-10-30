import copy
import unittest

from scripts.collect_confirmed_diagonal_data import validate_confirmation


class ConfirmedCollectionTests(unittest.TestCase):
    def setUp(self):
        self.config = {'frozen_teacher_head_sha256': 'preserved', 'actor_lr': 0, 'updates_per_stage': 0}
        self.evidence = {'audit_status': 'passed', 'qualified': True,
                         'primary_delta_ci95': [0.001, 1.999],
                         'candidate': {'mean_net_points': 1.0},
                         'candidate_head_sha256': 'preserved',
                         'training_and_evaluation_streams_disjoint': True}

    def test_small_confirmed_gain_has_no_250_or_500_point_veto(self):
        validate_confirmation(self.config, self.evidence, 'preserved')

    def test_inconclusive_or_unaudited_source_cannot_be_called_confirmed(self):
        for change in [{'qualified': False}, {'audit_status': 'failed'},
                       {'primary_delta_ci95': [-0.001, 2.0]},
                       {'training_and_evaluation_streams_disjoint': False}]:
            evidence = {**copy.deepcopy(self.evidence), **change}
            with self.assertRaises(ValueError):
                validate_confirmation(self.config, evidence, 'preserved')

    def test_wrong_teacher_or_actor_updates_rejected(self):
        with self.assertRaises(ValueError):
            validate_confirmation(self.config, self.evidence, 'different')
        with self.assertRaises(ValueError):
            validate_confirmation({**self.config, 'actor_lr': 0.1}, self.evidence, 'preserved')


if __name__ == '__main__':
    unittest.main()
