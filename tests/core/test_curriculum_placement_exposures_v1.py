from collections import Counter
import copy
import random
import unittest

from tools.curriculum_placement_exposures_v1 import expand_train, epoch_order, choose_accumulation, REPEAT_FIELD
from tools.curriculum_placement_mix_v1 import layout, REPLAY_MASS, NEW_SOURCES, RECORDS


def fixture():
    counts = {s: max(1, int(v*RECORDS)) for s, v in REPLAY_MASS.items()}
    counts[max(counts, key=counts.get)] += RECORDS-sum(counts.values())
    counts.update({s: 8+(i % 3)*8 for i, s in enumerate(NEW_SOURCES)})
    rows = [dict(_meta=dict(id=f'{source}:{i}', split='train', group_id=f'{source}:g{i % 4}'),
                 state=dict(visible='synthetic'), questions=dict(action=dict(src=source,
                 label='a', criteria={'a': {'type': 'dahai'}, 'b': {'type': 'reach'}}, target={'a': .6, 'b': .4})))
            for source, count in counts.items() for i in range(count)]
    return rows, layout(counts, {s: 4 for s in counts})


class PlacementExposureTests(unittest.TestCase):
    def test_replica_provenance_preserves_originals_and_all_visible_supervision(self):
        rows, plan = fixture(); expanded = expand_train(rows, plan)
        self.assertEqual(len(expanded), plan['virtual_exposures_per_pass'])
        self.assertTrue(all(a is b for a, b in zip(rows, expanded)))
        originals = {r['_meta']['id']: r for r in rows}
        self.assertEqual(len({r['_meta']['id'] for r in expanded}), len(expanded))
        for row in expanded[len(rows):]:
            meta = row['_meta']; provenance = meta[REPEAT_FIELD]
            original = originals[provenance['original_id']]
            restored = copy.deepcopy(row)
            restored['_meta']['id'] = provenance['original_id']; restored['_meta'].pop(REPEAT_FIELD)
            self.assertEqual(restored, original)
            self.assertFalse(provenance['adds_independent_data'])
        self.assertTrue(all(REPEAT_FIELD not in r['_meta'] for r in rows))

    def test_deterministic_epoch_has_exact_exposures_and_preserves_replay_order(self):
        rows, plan = fixture(); expanded = expand_train(rows, plan)
        categories = [r['questions']['action']['src'] for r in expanded]
        accum = choose_accumulation(len(expanded))
        a, receipt = epoch_order(categories, random.Random(83279), accum)
        b, _ = epoch_order(categories, random.Random(83279), accum)
        c, _ = epoch_order(categories, random.Random(83280), accum)
        self.assertEqual(a, b); self.assertNotEqual(a, c)
        self.assertEqual(sorted(a), list(range(len(expanded))))
        actual = Counter()
        for counts in receipt['source_exposure_counts_by_update']: actual.update(counts)
        self.assertEqual(dict(actual), plan['virtual_exposures_by_source'])
        old = [i for i in a if categories[i] in REPLAY_MASS]
        self.assertEqual(len(old), RECORDS)
        for start in range(0, len(old), 511):
            self.assertEqual({categories[i] for i in old[start:start+511]}, set(REPLAY_MASS))
        self.assertFalse(receipt['new_sources_each_optimizer_update_claimed'])

    def test_validation_and_preexpanded_inputs_cannot_be_expanded(self):
        rows, plan = fixture()
        rows[0]['_meta']['split'] = 'validation'
        with self.assertRaises(ValueError): expand_train(rows, plan)
        rows[0]['_meta']['split'] = 'train'; rows[0]['_meta'][REPEAT_FIELD] = {}
        with self.assertRaises(ValueError): expand_train(rows, plan)

    def test_changed_proposal_and_unknown_or_missing_sources_fail(self):
        rows, plan = fixture(); bad = copy.deepcopy(plan)
        bad['virtual_exposures_per_pass'] += 1
        with self.assertRaises(ValueError): expand_train(rows, bad)
        expanded = expand_train(rows, plan); categories = [r['questions']['action']['src'] for r in expanded]
        with self.assertRaises(ValueError): epoch_order(categories+['unknown'], random.Random(1), 73)
        with self.assertRaises(ValueError):
            epoch_order([s for s in categories if s != NEW_SOURCES[0]], random.Random(1), 73)

