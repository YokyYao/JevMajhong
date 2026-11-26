import copy
import unittest
from types import SimpleNamespace
from tools.curriculum_risk_teacher import category,guard_diagnostic


class RiskTeacherTests(unittest.TestCase):
    def state(self):return dict(seat=0,riichi=[1],melds=[[],[],[],[]])

    def test_only_novel_closed_not_own_riichi_discard_windows_are_selected(self):
        state=self.state();self.assertEqual(category({'dahai'},state),'closed_discard_under_riichi')
        self.assertIsNone(category({'dahai'},state,existing=True))
        for types in ({'dahai','reach'},{'chi','none'},{'dahai','ankan'},{'hora'}):self.assertIsNone(category(types,state))
        for field,value in (('riichi',[]),('riichi',[0,1]),('melds',[['pon'],[],[],[]])):
            changed=copy.deepcopy(state);changed[field]=value;self.assertIsNone(category({'dahai'},changed))

    def request(self):return dict(state=self.state(),questions={'action':{'criteria':{'a':{'type':'dahai'},'b':{'type':'dahai'}}}})

    def test_safe_teacher_discard_can_be_excluded_by_the_shanten_ceiling(self):
        features=[{'sh':0,'safe':[]},{'sh':1,'safe':[1]}]
        result=guard_diagnostic(SimpleNamespace(),[0,1],features,self.request(),'b',lambda *a:[0],lambda *a:0)
        self.assertFalse(result['teacher_target_allowed']);self.assertEqual(result['reason'],'blocked_shanten_increase')
        self.assertTrue(result['feature_safe_to_all_threats']);self.assertFalse(result['fold_rule_active'])

    def test_rule_risk_and_allowed_are_reported_without_changing_eligibility(self):
        features=[{'sh':1,'safe':[1]},{'sh':1,'safe':[]}]
        result=guard_diagnostic(SimpleNamespace(),[0,1],features,self.request(),'b',lambda *a:[0],lambda *a:0)
        self.assertEqual(result['reason'],'blocked_rule_risk');self.assertTrue(result['fold_rule_active'])
        other=guard_diagnostic(SimpleNamespace(),[0,1],features,self.request(),'b',lambda *a:[0,1],lambda *a:0)
        self.assertEqual(other['reason'],'allowed');self.assertTrue(other['teacher_target_allowed'])

    def test_unexplained_exclusion_is_rejected(self):
        features=[{'sh':1,'safe':[1]},{'sh':1,'safe':[1]}]
        with self.assertRaises(ValueError):guard_diagnostic(SimpleNamespace(),[0,1],features,self.request(),'b',lambda *a:[0],lambda *a:0)

    def test_selection_is_independent_of_outcome_and_teacher_action(self):
        for outcome in ('win','dealin','other_win'):
            state=dict(self.state(),future_outcome=outcome,teacher_chosen_action='unsafe',scores=[1,99999,-1000,0])
            self.assertEqual(category({'dahai'},state),'closed_discard_under_riichi')


if __name__=='__main__':unittest.main()
