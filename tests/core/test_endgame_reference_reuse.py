import copy
import unittest
from jevmahjong.endgame.collect import digest
from jevmahjong.endgame.reference import reused_reference_prior


class ReferenceReuseTests(unittest.TestCase):
    def test_exact_cached_values_can_be_reused_but_changed_input_or_support_cannot(self):
        payload={'state':{'scores':[25000]*4},'questions':{}}
        teacher={'state':{'original':True},'questions':{}}
        row={'request_sha256':digest(payload),'teacher_request_sha256':digest(teacher),
             'teacher_eligible_keys':['a','b'],'probabilities':{'a':.7,'b':.3,'c':0.}}
        self.assertEqual(reused_reference_prior(row,payload,teacher,['a','b'],['a','b','c']),row['probabilities'])
        for kwargs in ({'payload':{'different':True}},{'teacher':{'different':True}},
                       {'eligible':['a']},{'keys':['b','a','c']}):
            args=dict(row=row,payload=payload,teacher=teacher,eligible=['a','b'],keys=['a','b','c']);args.update(kwargs)
            with self.assertRaises(ValueError):reused_reference_prior(**args)
        bad=copy.deepcopy(row);bad['probabilities']={'a':.6,'b':.3,'c':.1}
        with self.assertRaises(ValueError):reused_reference_prior(bad,payload,teacher,['a','b'],['a','b','c'])


if __name__=='__main__':unittest.main()
