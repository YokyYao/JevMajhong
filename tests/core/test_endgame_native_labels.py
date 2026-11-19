import copy
import unittest

from tools.label_native_endgame import select_rows, partition, validate_config, validate_label


class NativeTerminalLabelsTests(unittest.TestCase):
    def rows(self):
        rows = []
        for i in range(80):
            seed = 2026116900 + i
            rank = 4 if i < 40 else 3
            # 40 consecutive original groups alone have 10 validation groups;
            # construct the required fixed32/8 split without changing its rule.
            if i < 40:
                seed = 2026116900 + (i // 4) * 4 + (i % 4) if i < 32 else 2026117000 + 4 * (i - 32)
            rows.append({'id': 'native-context:' + str(i), 'seed': seed, 'rank': rank,
                'split': 'validation' if (seed - 2026116900) % 4 == 3 else 'train',
                'group_id': 'endgame-native:' + str(seed),
                'teacher_request': {'questions': {'action': {'criteria': {'a': {'type': 'dahai'}, 'b': {'type': 'dahai'}}}}}})
        return rows

    def test_all_fourth_rows_reused_without_outcome_selection_or_split_change(self):
        rows = self.rows(); expected = select_rows(rows)
        changed = copy.deepcopy(rows)
        for row in changed: row['future_score'] = 100000 - row['seed']
        self.assertEqual([r['id'] for r in expected], [r['id'] for r in select_rows(changed)])
        self.assertEqual({r['id'] for p in partition(expected) for r in p}, {r['id'] for r in expected})
        bad = copy.deepcopy(rows); bad[0]['split'] = 'validation'
        with self.assertRaises(ValueError): select_rows(bad)
        bad = copy.deepcopy(rows); bad[0]['teacher_request']['questions']['action']['criteria']['a']['type'] = 'hora'
        with self.assertRaises(ValueError): select_rows(bad)

    def test_failed_teacher_or_unregistered_stage_rejected_before_sources(self):
        with self.assertRaises(ValueError): validate_config({'continuation_policy': 'head_trial11'})

    def test_complete_terminal_history_and_original_student_input_required(self):
        row = {'seat': 3, 'student_request_sha256': 'input', 'student_request':
            {'state': {'seat': 3}, 'questions': {'action': {'criteria': {'a': {'type': 'dahai'}, 'b': {'type': 'dahai'}}}}}}
        outcome = {'scores': [40000, 30000, 20000, 10000], 'ranks': [1, 2, 3, 4],
                   'utility': -1., 'ledger_passed': True, 'hand_history': {'final_scores': [40000, 30000, 20000, 10000]}}
        record = copy.deepcopy(row['student_request'])
        record['questions']['action']['target'] = {'a': .5, 'b': .5}
        record['_meta'] = {'expected_terminal_utilities': {'a': -1., 'b': -1.}}
        label = {'status': 'complete_terminal_menu', 'source': row, 'source_request_sha256': 'input',
            'baseline_identity_sha256': 'v8', 'worlds_per_action': 16, 'continued_until_native_match_done': True,
            'actual_actions': 2, 'record': record,
            'outcomes': [[copy.deepcopy(outcome) for _ in range(16)] for _ in range(2)]}
        self.assertEqual(validate_label(label, row, {'target_temperature': .1}, 'v8'), 32)
        for mutate in (lambda x: x['outcomes'][0].pop(),
                       lambda x: x['outcomes'][1][0]['hand_history']['final_scores'].__setitem__(3, 9999),
                       lambda x: x['record']['state'].__setitem__('future_rank', 1)):
            bad = copy.deepcopy(label); mutate(bad)
            with self.assertRaises(ValueError): validate_label(bad, row, {'target_temperature': .1}, 'v8')


if __name__ == '__main__': unittest.main()
