import copy
import unittest
from tools.curriculum_post_reach_teacher_v1 import category


class PostReachTeacherTests(unittest.TestCase):
    def test_only_actual_immediate_post_reach_choices(self):
        state=dict(seat=0,riichi=[0,1],melds=[[],[],[],[]])
        self.assertEqual(category({'dahai'},state,previous_reach=True),'post_reach_discard')
        self.assertIsNone(category({'dahai'},state,previous_reach=False))
        self.assertIsNone(category({'dahai'},state,previous_reach=True,existing=True))
        self.assertIsNone(category({'dahai'},dict(state,riichi=[1]),previous_reach=True))

    def test_a_later_own_riichi_discard_is_not_an_immediate_followup(self):
        state=dict(seat=2,riichi=[2],melds=[[],[],[],[]])
        for types in ({'dahai','ankan'},{'reach','dahai'},{'hora'},{'chi','none'}):
            self.assertIsNone(category(types,state,previous_reach=True))
        self.assertIsNone(category({'dahai'},state))

    def test_prior_closed_kan_and_outcome_do_not_filter_the_window(self):
        state=dict(seat=0,riichi=[0],melds=[['closed_kan'],[],[],[]])
        before=copy.deepcopy(state)
        for outcome in ('win','dealin','other_win'):
            self.assertEqual(category({'dahai'},dict(state,future_outcome=outcome),previous_reach=True),'post_reach_discard')
        self.assertEqual(state,before)


if __name__=='__main__':unittest.main()
