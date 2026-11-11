import copy
import math
import unittest
from call_fit_diagnostic import reaction_records, summarize_fit


def example(identity, label='c'):
    return {'_meta': {'id': identity, 'split': 'train', 'view': 'native', 'phase': 'reaction'},
            'questions': {'action': {'label': label, 'criteria': {
                'c': {'type': 'chi', 'call_features': {'loses_closed_status': True, 'riichi_threat_count': 0}},
                'p': {'type': 'none'}}}}}


class FitDiagnosticTests(unittest.TestCase):
    def test_closed_chi_and_false_call_are_distinct(self):
        records = [example('a'), example('b', 'p')]
        rows = [{'id': 'a', 'keys': ['c', 'p'], 'label': 0, 'p': [.4, .6]},
                {'id': 'b', 'keys': ['c', 'p'], 'label': 1, 'p': [.8, .2]}]
        result = summarize_fit(records, rows)
        self.assertEqual(result['closed_chi']['n'], 1)
        self.assertEqual(result['closed_chi']['exact'], 0)
        self.assertEqual(result['all']['false_calls'], 1)
        self.assertAlmostEqual(result['all']['nll'], -(math.log(.4) + math.log(.2)) / 2)

    def test_same_call_type_wrong_combination_is_not_exact(self):
        record = example('a')
        record['questions']['action']['criteria']['d'] = copy.deepcopy(record['questions']['action']['criteria']['c'])
        result = summarize_fit([record], [{'id': 'a', 'keys': ['c', 'p', 'd'], 'label': 0, 'p': [.1, .1, .8]}])
        self.assertEqual(result['closed_chi']['exact'], 0)
        self.assertEqual(result['closed_chi']['predicted'], {'chi': 1})

    def test_rejects_duplicate_foreign_and_bad_alignment(self):
        record = example('a')
        row = {'id': 'a', 'keys': ['c', 'p'], 'label': 0, 'p': [.5, .5]}
        for key, value in [('id', 'foreign'), ('keys', ['p', 'c']), ('label', 1)]:
            bad = {**row, key: value}
            with self.assertRaises(ValueError):
                summarize_fit([record], [bad])
        with self.assertRaises(ValueError):
            summarize_fit([record, record], [row, row])

    def test_rejects_bad_probabilities(self):
        for values in ([float('nan'), .5], [-.1, 1.1], [.2, .2], [1.]):
            with self.assertRaises(ValueError):
                summarize_fit([example('a')], [{'id': 'a', 'keys': ['c', 'p'], 'label': 0, 'p': values}])

    def test_closed_status_must_be_known(self):
        record = example('a')
        del record['questions']['action']['criteria']['c']['call_features']['loses_closed_status']
        with self.assertRaises(ValueError):
            summarize_fit([record], [{'id': 'a', 'keys': ['c', 'p'], 'label': 0, 'p': [.7, .3]}])

    def test_original_split_only_no_sealed_test(self):
        self.assertEqual(len(reaction_records([example('a')], 'train')), 1)
        with self.assertRaises(ValueError):
            reaction_records([], 'test')
        for key, value in [('split', 'test'), ('view', 'screen_unknown')]:
            record = example('a');record['_meta'][key] = value
            with self.assertRaises(ValueError):
                reaction_records([record], 'train')
        with self.assertRaises(ValueError):
            reaction_records([example('a'), example('a')], 'train')


if __name__ == '__main__':
    unittest.main()
