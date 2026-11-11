import unittest
import json
import tempfile
from pathlib import Path
from reaction_selection import reaction_subset, validate_recipe, verify_benchmark_contract


def record(identity, kind, phase='reaction', split='validation'):
    return {'_meta': {'id': identity, 'phase': phase, 'split': split, 'view': 'native',
                      'group_id': 'match-a'},
            'questions': {'action': {'criteria': {'0': {'type': kind}}, 'label': '0'}}}


class ReactionSelectionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [record('chi', 'chi'), record('pon', 'pon'), record('pass', 'none'),
                     record('ordinary', 'dahai', 'closed_discard')]
        self.config = {'data': 'data/new-stage', 'selection_scope': 'reaction'}
        self.recipe = {'eval_data': 'data/new-stage/validation.jsonl',
                       'eval_transfer_suite': 'data/new-stage/reaction-validation',
                       'early_stop_metric': 'transfer_nll', 'eval_records': 0}

    def test_subset_preserves_labels_options_and_order(self):
        rows, actions = reaction_subset(self.rows)
        self.assertEqual(rows, self.rows[:3])
        self.assertEqual(actions, {'chi': 1, 'pon': 1, 'none': 1})

    def test_test_and_train_records_rejected(self):
        for split in ('test', 'train'):
            with self.subTest(split=split), self.assertRaises(ValueError):
                reaction_subset(self.rows + [record('foreign', 'chi', split=split)])

    def test_missing_call_class_rejected(self):
        with self.assertRaises(ValueError):
            reaction_subset(self.rows[1:])

    def test_duplicate_and_foreign_reaction_rejected(self):
        for extra in (self.rows[0], record('kan', 'daiminkan')):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                reaction_subset(self.rows + [extra])

    def test_actual_metric_and_full_diagnostics_required(self):
        validate_recipe(self.config, self.recipe)
        for key, value in [('early_stop_metric', 'development_nll'),
                           ('eval_transfer_suite', 'other-suite'),
                           ('eval_data', 'reaction-only.jsonl'), ('eval_records', 128)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_recipe(self.config, dict(self.recipe, **{key: value}))

    def test_sdk_normalization_supports_real_benchmark_rows(self):
        from jevany.data import load_records, api_request
        original = self.rows[:3]
        for row in original:
            row['state'] = 'Public mahjong state'
            row['questions']['action'].update(type='choice', instructions='Choose an action')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'validation.jsonl'
            path.write_text(''.join(json.dumps(row) + '\n' for row in original))
            normalized = load_records(path)
        self.assertEqual([api_request(r) for r in original], [api_request(r) for r in normalized])
        self.assertEqual(verify_benchmark_contract(normalized), 3)
        with self.assertRaises(KeyError):
            verify_benchmark_contract(original)


if __name__ == '__main__':
    unittest.main()
