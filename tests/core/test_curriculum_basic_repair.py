from collections import Counter
import math
import random
import unittest
from tools.curriculum_basic_repair import MASS,weights,accumulation,profile,grade
from tools.curriculum_course_order import course_order


class BasicRepairTests(unittest.TestCase):
    def test_six_source_epoch_keeps_each_example_and_loss_mass(self):
        counts={'legacy:plain':8826,'legacy:riichi_yes':194,'legacy:riichi_no':272,
                'teacher:plain':3106,'teacher:riichi_yes':1075,'teacher:riichi_no':812}
        scaling=weights(counts);size,updates=accumulation(counts)
        categories=[key for key,n in counts.items() for _ in range(n)]
        order=course_order(categories,random.Random(2026112002),size)
        self.assertEqual(sorted(order),list(range(len(categories))))
        self.assertEqual(updates%2,0)
        self.assertGreaterEqual(len(order)-(updates-1)*size,.75*size)
        for start in range(0,len(order),size):self.assertEqual({categories[i] for i in order[start:start+size]},set(MASS))
        for key in MASS:self.assertAlmostEqual(counts[key]*scaling[key]/len(categories),MASS[key])
        self.assertEqual(Counter(categories[i] for i in order),Counter(counts))

    def test_source_profile_does_not_turn_plain_rows_into_riichi_negatives(self):
        records=[];predictions=[]
        for key in MASS:
            category=key.split(':')[1];reach=category=='riichi_yes'
            criteria={'d':{'type':'dahai'},'x':{'type':'dahai'}} if category=='plain' else {'d':{'type':'dahai'},'r':{'type':'reach'}}
            label='r' if reach else 'd';keys=list(criteria);prob=[.9 if k==label else .1 for k in keys]
            records.append({'_meta':{'id':key},'questions':{'action':{'src':'curriculum1:'+key,'criteria':criteria,'label':label}}})
            predictions.append({'id':key,'keys':keys,'p':prob})
        ce=lambda q,keys,p:-math.log(p[keys.index(q['label'])])
        result=profile(records,predictions,MASS,{'weighted_target_ce':123},ce)
        self.assertEqual(result['riichi_type_counts'],{'tp':2,'tn':2})
        self.assertEqual(result['riichi_by_source']['legacy'],{'tp':1,'tn':1})
        self.assertEqual(result['stratum_counts'],{'plain':2,'riichi_yes':2,'riichi_no':2})
        self.assertTrue(grade(result)['all_source_quality_passed'])

    def test_pooled_good_results_cannot_hide_a_failed_legacy_decline_class(self):
        value={'riichi_type_counts':{'tp':190,'fn':10,'tn':190,'fp':10},
            'riichi_by_source':{'legacy':{'tp':10,'fn':0,'tn':1,'fp':9},'teacher':{'tp':180,'fn':10,'tn':189,'fp':1}},
            'stratum_accuracy':{'plain':.8},'stratum_ce':{'plain':.5},
            'source_stratum_accuracy':{'legacy:plain':.8},'source_stratum_ce':{'legacy:plain':.5}}
        result=grade(value)
        self.assertTrue(result['original_aggregate_gate']['passed'])
        self.assertTrue(result['independent_teacher_source_gate']['passed'])
        self.assertFalse(result['unchanged_legacy_gate']['passed'])
        self.assertFalse(result['all_source_quality_passed'])

    def test_missing_sources_and_extreme_reweighting_are_rejected(self):
        with self.assertRaises(ValueError):weights({'legacy:plain':10})
        counts=dict.fromkeys(MASS,100);counts['legacy:riichi_yes']=1
        with self.assertRaises(ValueError):weights(counts)


if __name__=='__main__':unittest.main()
