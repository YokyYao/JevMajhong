import copy
import unittest
from unittest.mock import patch

from tools import curriculum_placement_partition_v1 as partition
from tools.curriculum_placement_exposures_v1 import REPEAT_FIELD
from tools.curriculum_placement_mix_v1 import layout, REPLAY_MASS, NEW_SOURCES, RECORDS


def small():
    return dict(train=[dict(_meta=dict(id='t', group_id='gt'), state={'x': 1})],
        validation=[dict(_meta=dict(id='v', group_id='gv', split='train'), state={'x': 2})])


class PlacementPartitionTests(unittest.TestCase):
    def test_source_tags_do_not_redefine_current_partition_or_mutate_originals(self):
        pools = small(); before = copy.deepcopy(pools); manifest = partition.membership(pools)
        views = partition.partition_views(pools['validation'], manifest, 'validation')
        self.assertEqual(views[0]['_meta']['split'], 'validation')
        self.assertEqual(views[0]['_meta'][partition.DERIVED]['original_split_value'], 'train')
        self.assertEqual(pools, before)
        self.assertEqual(manifest['partitions']['train']['legacy_split_counts'], {'<absent>': 1})
        with self.assertRaises(ValueError): partition.require_partition(pools['validation'], manifest, 'train')
        with self.assertRaises(ValueError): partition.require_partition(views, manifest, 'validation')

    def test_overlap_and_post_binding_record_changes_are_rejected(self):
        pools = small(); bad = copy.deepcopy(pools)
        bad['validation'][0]['_meta']['group_id'] = 'gt'
        with self.assertRaises(ValueError): partition.membership(bad)
        bad = copy.deepcopy(pools); bad['validation'][0]['_meta']['id'] = 't'
        with self.assertRaises(ValueError): partition.membership(bad)
        manifest = partition.membership(pools); pools['validation'][0]['state']['x'] = 3
        with self.assertRaises(ValueError): partition.require_partition(pools['validation'], manifest, 'validation')

    def test_full_validation_wrapper_uses_views_and_preserves_source_metadata(self):
        pools = small(); manifest = partition.membership(pools)
        with patch.object(partition.metrics, 'profile', return_value={'loss': .4}) as profile:
            result = partition.profile(pools['validation'], [], {}, None, manifest)
        self.assertEqual(profile.call_args.args[0][0]['_meta']['split'], 'validation')
        self.assertEqual(pools['validation'][0]['_meta']['split'], 'train')
        self.assertTrue(result['validation_current_membership_verified'])

    def test_expansion_restores_absent_or_stale_source_tags_on_originals_and_copies(self):
        counts = {s: max(1, int(v*RECORDS)) for s, v in REPLAY_MASS.items()}
        counts[max(counts, key=counts.get)] += RECORDS-sum(counts.values())
        counts.update({s: 8 for s in NEW_SOURCES})
        rows = [dict(_meta=dict(id=f'{s}:{i}', group_id=f'{s}:g{i % 4}'),
            questions=dict(action=dict(src=s, label='a', criteria={'a': {'type': 'dahai'}})))
            for s, n in counts.items() for i in range(n)]
        for row in rows:
            if row['questions']['action']['src'] in NEW_SOURCES:
                row['_meta']['split'] = 'validation'  # Inherited source tag, not current membership.
        heldout = dict(_meta=dict(id='v', group_id='isolated-validation', split='train'))
        manifest = partition.membership(dict(train=rows, validation=[heldout]))
        plan = layout(counts, {s: 4 for s in counts})
        expanded = partition.expand_train(rows, plan, manifest)
        originals = {r['_meta']['id']: r for r in rows}
        self.assertTrue(all(a is b for a, b in zip(rows, expanded)))
        self.assertNotIn('split', expanded[0]['_meta'])
        self.assertTrue(all(partition.DERIVED not in r['_meta'] for r in expanded))
        for replica in expanded[len(rows):]:
            provenance = replica['_meta'][REPEAT_FIELD]; restored = copy.deepcopy(replica)
            restored['_meta']['id'] = provenance['original_id']; restored['_meta'].pop(REPEAT_FIELD)
            self.assertEqual(restored, originals[provenance['original_id']])
        with self.assertRaises(ValueError): partition.expand_train([heldout], plan, manifest)
