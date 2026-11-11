import copy
import unittest
from call_fit_control import QUOTAS, choose_control


def pool():
    result = []
    for kind, count in QUOTAS.items():
        for i in range(count + 5):
            phase = 'closed_discard' if kind == 'ordinary' else 'reaction'
            action = 'dahai' if kind == 'ordinary' else kind.split('_')[-1]
            result.append({'_meta': {'id': kind + str(i), 'split': 'train', 'view': 'native', 'phase': phase},
                'questions': {'action': {'label': 'a', 'criteria': {'a': {'type': action,
                    'call_features': {'loses_closed_status': kind.startswith('closed_')}}}}}})
    return result


class FitControlTests(unittest.TestCase):
    def test_balanced_exact_quotas_original_records_unchanged(self):
        records = pool();before = copy.deepcopy(records)
        selected, counts = choose_control(records, 'balanced', 7)
        self.assertEqual(counts, QUOTAS)
        self.assertEqual(len({r['_meta']['id'] for r in selected}), 512)
        self.assertEqual(records, before)

    def test_natural_phase_budget_and_determinism(self):
        records = pool()
        first, counts = choose_control(records, 'natural', 7)
        self.assertEqual(len(first), 512)
        self.assertEqual(counts['ordinary'], 64)
        self.assertEqual(first, choose_control(records, 'natural', 7)[0])
        self.assertNotEqual(first, choose_control(records, 'natural', 8)[0])

    def test_rejects_heldout_and_duplicates(self):
        for split in ('validation', 'test'):
            records = pool();records[0]['_meta']['split'] = split
            with self.assertRaises(ValueError):choose_control(records, 'balanced', 7)
        records = pool();records.append(records[0])
        with self.assertRaises(ValueError):choose_control(records, 'natural', 7)

    def test_rejects_insufficient_positive_support(self):
        records = [r for r in pool() if not r['_meta']['id'].startswith('closed_chi')]
        with self.assertRaises(ValueError):choose_control(records, 'balanced', 7)

    def test_rejects_unknown_status_and_arm(self):
        records = pool();del records[0]['questions']['action']['criteria']['a']['call_features']['loses_closed_status']
        with self.assertRaises(ValueError):choose_control(records, 'balanced', 7)
        with self.assertRaises(ValueError):choose_control(pool(), 'unknown', 7)


if __name__ == '__main__':
    unittest.main()
