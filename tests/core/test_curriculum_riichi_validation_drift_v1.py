import copy
import unittest

from tools.diagnostics.riichi_validation_drift_v1 import transitions


class DecisionTransitions(unittest.TestCase):
    def rows(self):
        return [dict(id=str(i),group='g'+str(i),source=source,truth=truth,
            raw_argmax=choice,discard_choices=12) for i,(source,truth,choice) in enumerate([
                ('legacy',True,True),('legacy',False,True),
                ('teacher',True,False),('teacher',False,False)])]

    def test_gains_do_not_hide_regressions(self):
        before=self.rows();after=copy.deepcopy(before)
        after[0]['raw_argmax']=False;after[1]['raw_argmax']=False;after[2]['raw_argmax']=True
        result,detail=transitions(before,list(reversed(after)))
        self.assertEqual(result['pooled'],dict(correct_to_correct=1,correct_to_wrong=1,
            wrong_to_correct=2,wrong_to_wrong=0,windows=4,changed=3,net_correct_change=1,
            correct_before=2,correct_after=3))
        self.assertEqual(result['legacy']['net_correct_change'],0)
        self.assertEqual(len(detail),4)

    def test_reject_changed_population_or_labels(self):
        before=self.rows()
        for key,value in [('id','other'),('group','other'),('source','teacher'),('truth',False),('discard_choices',11)]:
            after=copy.deepcopy(before);after[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):transitions(before,after)
        with self.assertRaises(ValueError):transitions(before,before+before[:1])
        with self.assertRaises(ValueError):transitions([],[])

    def test_unchanged_actions_are_retained(self):
        rows=self.rows();result,_=transitions(rows,copy.deepcopy(rows))
        self.assertEqual(result['pooled']['changed'],0)
        self.assertEqual(result['pooled']['correct_to_correct'],2)
        self.assertEqual(result['pooled']['wrong_to_wrong'],2)


if __name__=='__main__':unittest.main()
