import unittest
from unittest.mock import patch
from jevmahjong.endgame.reference import greedy_win_target, apply_reference, inference_request
from jevmahjong.endgame.prepare import loss_recipe


class DeterministicWinTargetTests(unittest.TestCase):
    def test_positive_best_choice_is_not_suppressed_by_menu_size(self):
        for count in (1, 4, 14):
            criteria = {'win': {'type': 'hora'}, 'best': {'type': 'none'}}
            criteria.update({f'other{i}': {'type': 'dahai'} for i in range(count)})
            values = {k: -.05 for k in criteria}; values.update(win=0., best=.01)
            target = greedy_win_target(criteria, values, list(criteria))
            self.assertEqual(max(target, key=target.get), 'best')
            self.assertEqual(target['win'], 0)

    def test_equally_good_continuations_stay_above_win_individually(self):
        criteria = {'win': {'type': 'hora'}, **{f'a{i}': {'type': 'dahai'} for i in range(12)}}
        target = greedy_win_target(criteria, {k: 0. if k == 'win' else .05 for k in criteria}, list(criteria))
        self.assertAlmostEqual(sum(target.values()), 1)
        self.assertTrue(all(target[k] > target['win'] for k in criteria if k != 'win'))

    def test_ties_losses_and_unsupported_better_actions_keep_the_win(self):
        criteria = {'win': {'type': 'hora'}, 'pass': {'type': 'none'}, 'unsupported': {'type': 'none'}}
        for value in (0., -.05):
            target = greedy_win_target(criteria, {'win': 0., 'pass': value, 'unsupported': 1.1}, ['win', 'pass'])
            self.assertEqual(target, {'win': 1., 'pass': 0., 'unsupported': 0.})

    def test_runtime_fixed_win_controls_do_not_displace_replay_mass(self):
        counts = {'replay_reference': 512, 'replay_fixed_win': 79, 'endgame_terminal_win': 47,
                  'endgame_terminal_third': 29, 'endgame_terminal_fourth': 23, 'endgame_terminal_hold': 23}
        mass = {'replay_reference': .7, 'replay_fixed_win': 0., 'endgame_terminal_win': .15,
                'endgame_terminal_third': .05, 'endgame_terminal_fourth': .05, 'endgame_terminal_hold': .05}
        actual, weights = loss_recipe(counts, .3, True, True, mass)
        self.assertEqual(weights['replay_fixed_win'], 0)
        self.assertEqual(actual['replay_reference'], .7)
        self.assertAlmostEqual(sum(counts[k] * weights[k] for k in counts) / sum(counts.values()), 1)
        invalid = dict(mass, replay_reference=0., endgame_terminal_win=.85)
        with self.assertRaises(ValueError):
            loss_recipe(counts, .3, True, True, invalid)

    def test_new_targets_and_zero_loss_controls_do_not_change_model_input(self):
        def record(rid, enough):
            return {'state': {'endgame_context': {'late_match_window': True, 'placement': 3,
                'current_win_settlement': {'status': 'public_single_winner_baseline_scenarios',
                    'winner_is_dealer': False, 'top2_in_any_baseline_scenario': enough}}},
                '_meta': {'id': rid, 'bucket': 'win', 'target_basis': 'complete_terminal_MC_soft_target',
                          'expected_terminal_utilities': {'win': 0., 'pass': .01}},
                'questions': {'action': {'type': 'choice', 'src': 'endgame_terminal',
                    'criteria': {'win': {'type': 'hora'}, 'pass': {'type': 'none'}},
                    'target': {'win': .5, 'pass': .5}, 'label': 'win'}}}
        rows = [record('fixed', True), record('active', False)]
        before = [inference_request(r) for r in rows]
        cache = {'identity_sha256': 'test', 'input_version': 'endgame-public-v3', 'rows': {
            r['_meta']['id']: {'probabilities': {'win': 1., 'pass': 0.}} for r in rows}}
        config = {'reference_mode': 'released_original_policy', 'reference_temperature': .1,
                  'input_version': 'endgame-public-v3',
                  'reference_exploration': .1, 'split_fixed_win_reference': True,
                  'win_target_style': 'greedy_terminal_best'}
        with patch('jevmahjong.endgame.reference.load_reference', return_value=cache):
            apply_reference(config, {'train': rows, 'validation': []})
        self.assertEqual(rows[0]['questions']['action']['src'], 'replay_fixed_win')
        self.assertEqual(rows[1]['questions']['action']['target'], {'win': 0., 'pass': 1.})
        self.assertEqual([inference_request(r) for r in rows], before)


if __name__ == '__main__':
    unittest.main()
