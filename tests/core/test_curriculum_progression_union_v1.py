import copy
import unittest

from tools.prepare_curriculum_progression_union_v1 import mc_support, restore_source_tag


class ProgressionUnionTests(unittest.TestCase):
    def row(self):
        return dict(_meta=dict(label_type='paired_future_masked'), questions=dict(action=dict(
            criteria={'a': {}, 'b': {}, 'unobserved': {}}, target={'a': .4, 'b': .6}, src='paired_future_masked')))

    def test_mask_support_is_not_full_legal_menu_and_singletons_are_structural(self):
        row = self.row(); self.assertEqual(mc_support(row), 2)
        row['questions']['action']['target'] = {'a': 1.}; self.assertEqual(mc_support(row), 1)
        for target in ({'a': .5}, {'unknown': 1.}, {'a': float('nan')}, {'a': -1., 'b': 2.}):
            row['questions']['action']['target'] = target
            with self.assertRaises(ValueError): mc_support(row)

    def test_only_private_source_tag_may_differ(self):
        original = self.row(); row = copy.deepcopy(original)
        row['questions']['action']['src'] = 'curriculum3:mc:closed_discard'
        self.assertEqual(restore_source_tag(row, original), original)
        self.assertEqual(row['questions']['action']['src'], 'curriculum3:mc:closed_discard')
        row['questions']['action']['target'] = {'a': .6, 'b': .4}
        with self.assertRaises(ValueError): restore_source_tag(row, original)


if __name__ == '__main__': unittest.main()
