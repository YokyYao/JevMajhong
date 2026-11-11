import copy
import unittest
from targeted_reaction_sampling import choose_requests, public_stratum, STRATA


def record(identity, stratum):
    criteria = {'pass': {'type': 'none'}}
    if stratum.startswith('closed_chi'):
        criteria['call'] = {'type': 'chi', 'call_features': {
            'loses_closed_status': True, 'riichi_threat_count': int(stratum.endswith('threat') and not stratum.endswith('no_threat'))}}
    return {'_meta': {'id': identity, 'split': 'train', 'view': 'native',
                     'phase': 'closed_discard' if stratum == 'ordinary' else 'reaction'},
            'questions': {'action': {'criteria': criteria, 'label': 'pass'}}}


class TargetedReactionSamplingTests(unittest.TestCase):
    def setUp(self):
        self.records = [record(f'{kind}-{i}', kind) for kind in STRATA for i in range(5)]
        self.quotas = dict.fromkeys(STRATA, 2)

    def test_labels_do_not_affect_sampling_and_records_are_unchanged(self):
        before = copy.deepcopy(self.records)
        first, report = choose_requests(self.records, set(), self.quotas, 17)
        changed = copy.deepcopy(self.records)
        for row in changed:
            row['questions']['action']['label'] = next(reversed(row['questions']['action']['criteria']))
        second, _ = choose_requests(changed, set(), self.quotas, 17)
        self.assertEqual([r['_meta']['id'] for r in first], [r['_meta']['id'] for r in second])
        self.assertEqual(self.records, before)
        self.assertEqual(report['selected'], self.quotas)

    def test_executed_ids_excluded_without_replacement(self):
        excluded = {r['_meta']['id'] for r in self.records[::5]}
        selected, report = choose_requests(self.records, excluded, self.quotas, 2)
        ids = [r['_meta']['id'] for r in selected]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertFalse(set(ids) & excluded)
        self.assertEqual(report['excluded_requests'], 4)

    def test_unknown_exposure_and_insufficient_support_rejected(self):
        with self.assertRaises(ValueError):
            choose_requests(self.records, {'unknown'}, self.quotas, 1)
        with self.assertRaises(ValueError):
            choose_requests(self.records, set(), dict.fromkeys(STRATA, 6), 1)

    def test_eval_or_duplicate_records_rejected(self):
        for split in ('validation', 'test'):
            rows = copy.deepcopy(self.records)
            rows[0]['_meta']['split'] = split
            with self.subTest(split=split), self.assertRaises(ValueError):
                choose_requests(rows, set(), self.quotas, 1)
        with self.assertRaises(ValueError):
            choose_requests(self.records + self.records[:1], set(), self.quotas, 1)

    def test_public_threat_and_closed_opportunity_are_separate(self):
        for kind in STRATA:
            self.assertEqual(public_stratum(record('x', kind)), kind)
        row = record('x', 'closed_chi_threat')
        row['questions']['action']['criteria']['call']['call_features']['riichi_threat_count'] = None
        with self.assertRaises(ValueError):
            public_stratum(row)


if __name__ == '__main__':
    unittest.main()
