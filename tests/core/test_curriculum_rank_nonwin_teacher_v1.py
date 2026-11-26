import unittest
from tools.curriculum_rank_nonwin_teacher_v1 import candidate_keys, soft_target


def result(rank, net=0): return dict(full_match_completed=True, rank=rank, top2=rank <= 2, net_from_anchor=net)


class NonWinTeacherTests(unittest.TestCase):
    def test_available_calls_and_native_pass_all_have_current_candidates(self):
        request = dict(state=dict(seat=0, riichi=[]), questions=dict(action=dict(criteria={
            'pass':dict(type='none'), 'chi1':dict(type='chi'), 'chi2':dict(type='chi'),
            'pon':dict(type='pon'), 'kan':dict(type='daiminkan')})))
        got = candidate_keys(request, {'pass':.5,'chi1':.2,'chi2':.1,'pon':.15,'kan':.05}, 'pass')
        self.assertEqual(got, ['pass','chi1','pon','kan'])

    def test_fourth_can_prefer_top2_risk_to_safer_third_place(self):
        worlds = [{'safe':result(3), 'pursue':result(2 if i % 4 == 0 else 4)} for i in range(32)]
        got = soft_target(worlds, ['safe','pursue'], 4, 17)
        self.assertEqual(got['label'], 'pursue')
        self.assertGreater(got['target']['pursue'], .9)

    def test_leading_equal_placement_preserves_reference_not_extra_net(self):
        worlds = [{'base':result(2, 1000), 'other':result(2, 12000)} for _ in range(32)]
        got = soft_target(worlds, ['base','other'], 2, 17)
        self.assertEqual(got['label'], 'base')
        self.assertLess(got['target']['other'], .01)
