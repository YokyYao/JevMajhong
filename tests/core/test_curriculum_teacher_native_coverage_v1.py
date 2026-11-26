import unittest
from types import SimpleNamespace
from tools.audit_curriculum_teacher_native_coverage_v1 import own_discard_number, target_diagnostic


class NativeTeacherCoverageTests(unittest.TestCase):
    def test_real_tile_counts_and_text_length_disagree_but_tokens_match(self):
        public = dict(seat=0, rivers=['1m 0p E', '', '', ''])
        self.assertEqual(own_discard_number(dict(discards=[[0, 52, 108], [], [], []]), public), 4)
        self.assertEqual(own_discard_number(dict(discards=[[], [], [], []]), dict(seat=1, rivers=['', '', '', ''])), 1)
        with self.assertRaises(ValueError):
            own_discard_number(dict(discards=[[0, 52], [], [], []]), public)

    def request(self):
        return dict(state=dict(seat=0, riichi=[1]), questions={'action':{'criteria':{
            'a':{'type':'dahai'}, 'b':{'type':'dahai'}, 'c':{'type':'reach'}}}})

    def test_special_reach_survives_the_actual_serving_union(self):
        features = [dict(sh=1, safe=[1]), dict(sh=2, safe=[1]), {}]
        result = target_diagnostic(SimpleNamespace(), [0,1,2], features, self.request(), 'c', lambda *a:[0], lambda *a:0)
        self.assertTrue(result['teacher_target_allowed'])
        self.assertEqual(result['eligible_keys'], ['a','c'])

    def test_safe_fold_can_be_blocked_for_increasing_shanten(self):
        features = [dict(sh=0, safe=[]), dict(sh=1, safe=[1]), {}]
        result = target_diagnostic(SimpleNamespace(), [0,1,2], features, self.request(), 'b', lambda *a:[0], lambda *a:0)
        self.assertEqual(result['reason'], 'blocked_shanten_increase')
        self.assertFalse(result['fold_rule_active'])

    def test_unexplained_exclusion_is_rejected(self):
        features = [dict(sh=1, safe=[1]), dict(sh=1, safe=[1]), {}]
        with self.assertRaises(ValueError):
            target_diagnostic(SimpleNamespace(), [0,1,2], features, self.request(), 'b', lambda *a:[0], lambda *a:0)


if __name__ == '__main__':
    unittest.main()
