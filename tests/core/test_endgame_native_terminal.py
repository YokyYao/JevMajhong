import copy
import unittest

from jevmahjong.endgame.context import target_from_utilities
from jevmahjong.endgame.native_terminal import terminal_native_record, is_native_terminal
from jevmahjong.endgame.reference import greedy_improvement_target, improvement_target
from jevmahjong.endgame.fit import reference_kl_group, fourth_cohort_floor_guards


class NativeTerminalAdmissionTests(unittest.TestCase):
    def example(self):
        row = {'id': 'native-context:example', 'group_id': 'endgame-native:2026116900',
            'seed': 2026116900, 'seat': 3, 'rank': 4, 'split': 'train', 'phase': 'discard',
            'teacher_request_sha256': 'teacher', 'student_request_sha256': 'student',
            'student_request': {'state': {'seat': 3, 'scores': [40000, 30000, 20000, 10000]},
                'questions': {'action': {'type': 'choice', 'criteria':
                    {'old': {'type': 'dahai'}, 'better': {'type': 'dahai'}}}}}}
        fourth = {'ledger_passed': True, 'scores': [40000, 30000, 20000, 10000],
            'ranks': [1, 2, 3, 4], 'utility': -1., 'hand_history': {'final_scores': [40000, 30000, 20000, 10000]}}
        third = {'ledger_passed': True, 'scores': [40000, 30000, 10000, 20000],
            'ranks': [1, 2, 4, 3], 'utility': -.5, 'hand_history': {'final_scores': [40000, 30000, 10000, 20000]}}
        record = copy.deepcopy(row['student_request'])
        record['questions']['action']['target'] = dict(zip(('old', 'better'), target_from_utilities([-1., -.5], .1)))
        record['_meta'] = {'expected_terminal_utilities': {'old': -1., 'better': -.5}}
        label = {'status': 'complete_terminal_menu', 'source': row, 'source_request_sha256': 'student',
            'baseline_identity_sha256': 'v8', 'worlds_per_action': 16, 'continued_until_native_match_done': True,
            'actual_actions': 2, 'record': record,
            'outcomes': [[copy.deepcopy(fourth) for _ in range(16)], [copy.deepcopy(third) for _ in range(16)]]}
        return row, label

    def test_native_input_and_identity_remain_while_labels_become_terminal(self):
        row, label = self.example(); before = copy.deepcopy(row)
        actual = terminal_native_record(row, label, 'sourcehash', {'identity_sha256': 'v8', 'target_temperature': .25})
        self.assertEqual(row, before)
        self.assertEqual(actual['_meta']['id'], row['id'])
        self.assertEqual(actual['_meta']['group_id'], row['group_id'])
        self.assertEqual(actual['state'], row['student_request']['state'])
        self.assertEqual(actual['questions']['action']['criteria'], row['student_request']['questions']['action']['criteria'])
        self.assertEqual(actual['questions']['action']['src'], 'endgame_terminal')
        self.assertAlmostEqual(actual['_meta']['expected_terminal_utilities']['old'], -.05)
        self.assertEqual(actual['_meta']['expected_terminal_utilities']['better'], 0.)
        self.assertEqual(actual['_meta']['terminal_escape4_probabilities'], {'old': 0., 'better': 1.})
        self.assertNotIn('native_reference_pending', actual['_meta'])
        self.assertTrue(is_native_terminal(actual['_meta']))
        self.assertEqual(reference_kl_group('endgame_terminal', actual['_meta']), 'trailing')
        with self.assertRaises(ValueError):
            terminal_native_record(row, label, 'sourcehash', {'identity_sha256': 'failed-checkpoint', 'target_temperature': .25})

    def test_strict_greedy_improvement_is_not_hidden_by_the_old_prior(self):
        prior = {'old': .99, 'better': .01}; values = {'old': -.05, 'better': 0.}
        tilted = improvement_target(prior, values, .1, .1)
        self.assertEqual(max(tilted, key=tilted.get), 'old')
        target = greedy_improvement_target(prior, values, list(prior))
        self.assertEqual(target, {'old': 0., 'better': 1.})
        # A current best/tied old action should not be unnecessarily hardened.
        self.assertEqual(greedy_improvement_target(prior, {'old': .2, 'better': .2}, list(prior)), prior)
        self.assertEqual(greedy_improvement_target(prior, values, ['old']), {'old': 1., 'better': 0.})
        with self.assertRaises(ValueError): greedy_improvement_target(prior, {'old': 0., 'better': float('nan')}, list(prior))

    def test_eight_new_windows_cannot_mask_a_legacy43_escape_regression(self):
        current = {'existing': [.4] * 43, 'native_fourth': [1.] * 8}
        original = {'existing': [.5] * 43, 'native_fourth': [0.] * 8}
        initial = {'existing': .5, 'native_fourth': 0.}
        passed, audits = fourth_cohort_floor_guards(8, current, original, initial, True)
        self.assertFalse(passed['existing']); self.assertTrue(passed['native_fourth'])
        self.assertEqual(audits['existing']['positions'], 43)
        bad = copy.deepcopy(current); bad['native_fourth'].pop()
        with self.assertRaises(ValueError): fourth_cohort_floor_guards(8, bad, original, initial, True)
        passed, _ = fourth_cohort_floor_guards(0, current, original, {'existing': None, 'native_fourth': None}, True)
        self.assertTrue(all(passed.values()))


if __name__ == '__main__': unittest.main()
