import unittest
from tools.build_mortal_base import scope,family_weights,select_validation


class FreshBaseDataTests(unittest.TestCase):
    def test_riichi_decline_gets_explicit_opportunity_scope(self):
        r={'state':{'seat':0,'riichi':[]},'questions':{'action':{'criteria':{
            'a':{'type':'reach'},'b':{'type':'dahai'}}}},'_meta':{'teacher_action':'b'}}
        self.assertEqual(scope(r),'teacher_riichi_no')
        r['_meta']['teacher_action']='a';self.assertEqual(scope(r),'teacher_riichi_yes')

    def test_call_decline_is_not_hidden_in_ordinary_discard_sampling(self):
        r={'state':{'seat':0,'riichi':[]},'questions':{'action':{'criteria':{
            'a':{'type':'chi'},'b':{'type':'none'}}}},'_meta':{'teacher_action':'b'}}
        self.assertEqual(scope(r),'teacher_call_no')
        r['_meta']['teacher_action']='a';self.assertEqual(scope(r),'teacher_call_yes')

    def test_balanced_sampling_does_not_force_a_fifty_fifty_action_prior(self):
        counts={'teacher_plain':100,'teacher_defense':20,'teacher_kan':10,
                'teacher_riichi_yes':40,'teacher_riichi_no':60,'teacher_call_yes':10,'teacher_call_no':90}
        w=family_weights(counts,{'teacher_plain':.45,'teacher_defense':.15,'teacher_kan':.05,'riichi':.15,'call':.20})
        self.assertAlmostEqual(w['teacher_call_yes'],.02)
        self.assertAlmostEqual(w['teacher_call_no'],.18)
        self.assertAlmostEqual(sum(w.values()),1.)


if __name__=='__main__':unittest.main()
