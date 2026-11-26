import unittest
from tools.curriculum_placement_full_win_policy_v1 import select


class FullWinSelectionTests(unittest.TestCase):
    def request(self,entries):
        return dict(questions=dict(action=dict(criteria={k:dict(type=t) for k,t in entries})))

    def test_legal_riichi_kan_or_call_can_compete_with_the_win(self):
        request=self.request([('win','hora'),('discard','dahai'),('ri','reach'),('kan','ankan')])
        self.assertEqual(select(request,dict(win=.1,discard=.1,ri=.1,kan=.7)),'kan')
        self.assertEqual(select(request,dict(win=.1,discard=.1,ri=.7,kan=.1)),'ri')
        request=self.request([('win','hora'),('pass','none'),('pon','pon')])
        self.assertEqual(select(request,{'win':.1,'pass':.1,'pon':.8}),'pon')

    def test_ties_take_win_independently_of_menu_order(self):
        for entries in [[('win','hora'),('kan','ankan')],[('kan','ankan'),('win','hora')]]:
            self.assertEqual(select(self.request(entries),{'win':.5,'kan':.5}),'win')

    def test_complete_distribution_is_required(self):
        request=self.request([('win','hora'),('discard','dahai'),('ri','reach')])
        for values in [dict(win=.5,discard=.5),dict(win=.4,discard=.4,ri=float('nan')),
            dict(win=.5,discard=.5,ri=-.1),dict(win=.2,discard=.2,ri=.2)]:
            with self.assertRaises(ValueError):select(request,values)

    def test_complete_win_menu_uses_action_preferences_instead_of_nonwin_RI_class_rule(self):
        request=self.request([('win','hora'),('discard','dahai'),('ri','reach')])
        self.assertEqual(select(request,dict(win=.4,discard=.3,ri=.3)),'win')

