import copy
import unittest
from unittest.mock import patch

from jevmahjong.endgame.prepare import balance_training_sources, loss_recipe
from jevmahjong.endgame.reference import apply_reference


class ImprovementWeightingTests(unittest.TestCase):
    def row(self, rid, better):
        return {'state': {'seat': 3}, '_meta': {'id': rid, 'bucket': 'fourth',
            'source_initial_placement': 4, 'expected_terminal_utilities': {'old': -.05, 'better': better},
            'terminal_escape4_probabilities': {'old': 0., 'better': 1.}, 'target_basis': 'complete'},
            'questions': {'action': {'src': 'endgame_terminal', 'criteria':
                {'old': {'type': 'dahai'}, 'better': {'type': 'dahai'}},
                'label': 'better', 'target': {'old': .5, 'better': .5}}}}

    def test_flag_comes_from_strict_safe_terminal_improvement_not_action_type(self):
        rows = [self.row('improve', 0.), self.row('tie', -.05)]
        cache = {'identity_sha256': 'v8', 'input_version': 'endgame-trailing-public-v1',
            'rows': {rid: {'probabilities': {'old': .99, 'better': .01}} for rid in ('improve', 'tie')}}
        config = {'reference_mode': 'released_original_policy', 'input_version': 'endgame-trailing-public-v1',
            'reference_temperature': .1, 'reference_exploration': .1, 'fourth_escape_constraint': True,
            'ordinary_fourth_target_style': 'greedy_improvement', 'split_ordinary_fourth_improvements': True}
        with patch('jevmahjong.endgame.reference.load_reference', return_value=cache):
            apply_reference(config, {'train': rows, 'validation': []})
        self.assertTrue(rows[0]['_meta']['strict_original_policy_improvement'])
        self.assertFalse(rows[1]['_meta']['strict_original_policy_improvement'])
        self.assertEqual(rows[0]['questions']['action']['target'], {'old': 0., 'better': 1.})
        self.assertEqual(rows[1]['questions']['action']['target'], {'old': .99, 'better': .01})

    def test_source_split_retains_targets_and_refuses_unverified_markers(self):
        rows = [self.row('a', 0.), self.row('b', -.05)]
        for row, flag in zip(rows, (True, False)): row['_meta']['strict_original_policy_improvement'] = flag
        targets = [copy.deepcopy(r['questions']['action']['target']) for r in rows]
        config = {'balance_endgame_buckets': True, 'reference_regularization': True,
                  'split_ordinary_fourth_improvements': True}
        balance_training_sources(rows, config)
        self.assertEqual([r['questions']['action']['src'] for r in rows],
            ['endgame_terminal_fourth_improve', 'endgame_terminal_fourth'])
        self.assertEqual([r['questions']['action']['target'] for r in rows], targets)
        with self.assertRaises(ValueError): balance_training_sources([self.row('unverified', 0.)], config)
        staged = self.row('pending-reference', 0.)
        balance_training_sources([staged], dict(config, reference_regularization=False))
        self.assertEqual(staged['questions']['action']['src'], 'endgame_terminal_fourth')

    def test_registered_nineteen_source_mass_is_bounded(self):
        counts = {'replay_reference': 544, 'replay_reference_leading': 64, 'replay_fixed_win': 79,
            'endgame_terminal_win_third': 17, 'endgame_terminal_win_fourth': 94,
            'endgame_terminal_third': 29, 'endgame_terminal_fourth': 36,
            'endgame_terminal_fourth_improve': 19, 'endgame_terminal_hold': 23}
        mass = {'replay_reference': .35, 'replay_reference_leading': .15, 'replay_fixed_win': 0.,
            'endgame_terminal_win_third': .05, 'endgame_terminal_win_fourth': .15,
            'endgame_terminal_third': .05, 'endgame_terminal_fourth': .05,
            'endgame_terminal_fourth_improve': .15, 'endgame_terminal_hold': .05}
        _, weights = loss_recipe(counts, source_mass=mass)
        self.assertAlmostEqual(weights['endgame_terminal_fourth_improve'] * 19 / 905, .15)
        self.assertLess(max(weights.values()), 8.)


if __name__ == '__main__': unittest.main()
