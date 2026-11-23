import copy
import json
import unittest
from pathlib import Path

from tools import curriculum_calls_post_reach_data_v1 as data


def row(identifier, group, split='train'):
    return dict(state=dict(hand=[identifier]), questions=dict(action=dict(
        type='choice', instructions='choose', criteria={'d': {'type': 'dahai'}},
        label='d', src='curriculum1:teacher:post_reach_discard')),
        _meta=dict(id=identifier, group_id=group, split=split,
                   curriculum_stratum='post_reach_discard', previous_actual_own_action='reach'))


class CallsPostReachDataTests(unittest.TestCase):
    def counts(self):
        return dict(zip(data.OLD_MASS, [8826, 3106, 194, 272, 1075, 812,
                                      5590, 3569, 1786, 9352, 13399, 729]),
                    **{data.POST: 312})

    def test_configuration_and_fixed_mass_do_not_drop_replay(self):
        path = Path(__file__).resolve().parents[2]/'configs/training/curriculum-calls-post-reach-data-v1.json'
        self.assertEqual(json.loads(path.read_text()), data.CONFIG)
        self.assertAlmostEqual(sum(data.MASS.values()), 1.)
        for key, mass in data.OLD_MASS.items():
            self.assertEqual(data.MASS[key], .98*mass)
        self.assertEqual(data.MASS[data.POST], .02)

    def test_count_only_layout_preserves_real_full_epochs(self):
        counts = self.counts(); result = data.layout(counts)
        self.assertEqual(sum(counts.values()), 49022)
        self.assertEqual(result['record_exposures'], 98044)
        self.assertEqual(result['final_step'], 2*result['updates_per_epoch'])
        self.assertLessEqual(result['updates_per_epoch'], min(counts.values()))
        self.assertGreaterEqual(result['final_update_records'], .75*result['accumulation'])
        self.assertEqual(data.orders(counts), data.orders(counts))
        for order in data.orders(counts):
            self.assertEqual(order['records'], 49022)
            self.assertTrue(order['all_updates_have_all_thirteen_strata'])
        for broken in (dict(counts, **{data.POST: 0}), {k: v for k, v in counts.items() if k != data.POST}):
            with self.assertRaises(ValueError): data.layout(broken)

    def test_retag_keeps_public_input_labels_metadata_and_original_intact(self):
        original = row('p', 'g'); saved = copy.deepcopy(original)
        actual = data.retag_post(original, 'train')
        self.assertEqual(original, saved)
        self.assertEqual(actual['_meta'], original['_meta'])
        self.assertEqual(data.fingerprint(actual), data.fingerprint(original))
        actual['questions']['action']['src'] = original['questions']['action']['src']
        self.assertEqual(actual, original)
        for field, value in (('previous_actual_own_action', 'dahai'), ('split', 'validation')):
            altered = copy.deepcopy(original); altered['_meta'][field] = value
            with self.assertRaises(ValueError): data.retag_post(altered, 'train')
        altered = copy.deepcopy(original); altered['questions']['action']['criteria']['d']['type'] = 'reach'
        with self.assertRaises(ValueError): data.retag_post(altered, 'train')

    def test_merged_replay_rejects_reserved_groups_duplicates_and_cross_split_inputs(self):
        old = {'train': [row('a', 'a')], 'validation': [row('b', 'b', 'validation')]}
        pools = {'train': old['train']+[row('c', 'c')],
                 'validation': old['validation']+[row('d', 'd', 'validation')]}
        self.assertEqual(data.check_splits(pools, old, {'reserved'})['new_duplicate_inputs'], 0)
        with self.assertRaises(ValueError): data.check_splits(pools, old, {'c'})
        duplicate = copy.deepcopy(pools); duplicate['train'][-1]['state'] = old['train'][0]['state']
        with self.assertRaises(ValueError): data.check_splits(duplicate, old, set())
        leak = copy.deepcopy(pools); leak['train'][-1]['state'] = old['validation'][0]['state']
        with self.assertRaises(ValueError): data.check_splits(leak, old, set())


if __name__ == '__main__': unittest.main()
