import unittest
from tools.curriculum_post_reach_teacher_v1 import category as flawed_accepted_only
from tools.curriculum_post_reach_teacher_v2 import category


class PendingReachTeacherTests(unittest.TestCase):
    def test_actual_native_pending_reach_has_not_yet_set_the_acceptance_flag(self):
        state=dict(seat=0,riichi=[1],melds=[[],[],[],[]])
        self.assertIsNone(flawed_accepted_only({'dahai'},state,previous_reach=True))
        self.assertEqual(category({'dahai'},state,previous_reach=True),'post_reach_discard')

    def test_a_later_locked_draw_or_other_players_reach_is_not_a_followup(self):
        state=dict(seat=0,riichi=[0,1],melds=[[],[],[],[]])
        self.assertIsNone(category({'dahai'},state,previous_reach=False))
        self.assertIsNone(category({'dahai'},state))
        self.assertIsNone(category({'dahai'},state,previous_reach=True,existing=True))

    def test_only_the_native_discard_menu_is_selected(self):
        state=dict(seat=0,riichi=[],melds=[[],[],[],[]])
        for types in ({'dahai','ankan'},{'reach','dahai'},{'hora'},{'chi','none'}):
            self.assertIsNone(category(types,state,previous_reach=True))

    def test_no_filter_on_wins_safety_or_prior_concealed_kan(self):
        for won in (True,False):
            state=dict(seat=0,riichi=[],melds=[['closed_kan'],[],[],[]],future_win=won)
            self.assertEqual(category({'dahai'},state,previous_reach=True),'post_reach_discard')


if __name__=='__main__':unittest.main()
