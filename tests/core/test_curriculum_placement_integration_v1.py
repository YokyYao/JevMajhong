import copy
import unittest

from tools.preflight_curriculum_placement_v1 import control_rows, FilteredControlRandom
from tools.curriculum_placement_mix_v1 import masses
from tools.curriculum_placement_exposures_v1 import REPEAT_FIELD


def fixture():
    # Source metadata can differ from the current file membership.
    return [dict(_meta=dict(id=f'{source}:{i}', split='validation'),
        state=dict(source=source, sample=i), questions=dict(action=dict(src=source)))
        for source in masses() for i in range(2)]


class PlacementIntegrationTests(unittest.TestCase):
    def test_all_sources_use_distinct_pristine_train_file_inputs_without_rewriting_tags(self):
        rows = fixture(); before = copy.deepcopy(rows)
        copy_row = copy.deepcopy(rows[0]); copy_row['_meta'].update(id='virtual', **{REPEAT_FIELD: {}})
        rows.insert(0, copy_row)
        a,b = control_rows(rows, lambda r: tuple(r['state'].values()))
        self.assertEqual(len(a), 57); self.assertEqual(len(b), 57)
        self.assertEqual(rows[1:], before)
        self.assertTrue(all(index != 0 and row is rows[index] for index,row in a+b))
        self.assertFalse({r['_meta']['id'] for _,r in a} & {r['_meta']['id'] for _,r in b})
        self.assertEqual({r['questions']['action']['src'] for _,r in a}, set(masses()))

    def test_semantically_duplicate_second_pool_cannot_pass_by_changing_id(self):
        rows = fixture(); rows[1]['state'] = copy.deepcopy(rows[0]['state'])
        with self.assertRaises(ValueError): control_rows(rows, lambda r: tuple(r['state'].values()))

    def test_actual_full_order_is_filtered_into_control_indices(self):
        class Full:
            calls = 0
            def shuffle(self, order):
                if order != [0,1,2,3,4]: raise AssertionError('Not a complete full-order adapter call')
                order[:] = [4,1,3,0,2] if self.calls == 0 else [3,4,2,1,0]
                self.calls += 1
            def getstate(self): return ('actual_rng_state', self.calls)
        full = Full(); rng = FilteredControlRandom(full, 5, [3,1]); order = [0,1]
        rng.shuffle(order); self.assertEqual(order, [1,0])
        rng.shuffle(order); self.assertEqual(order, [0,1])
        self.assertEqual(rng.getstate(), ('actual_rng_state', 2))
        with self.assertRaises(ValueError): rng.shuffle([0,0])
        self.assertEqual(full.calls, 2)
