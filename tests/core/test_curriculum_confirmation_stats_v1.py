import copy
import math
import statistics
import unittest
from tools.curriculum_confirmation_stats_v1 import compare,interval,matrix,T_CRITICAL_DF31_975


def rows():
    result=[]
    for index in range(32):
        for profile in ('efficiency','calling'):
            for seat in range(4):
                scores=[30000,26000,24000,20000]
                rank=sorted(range(4),key=lambda p:(-scores[p],p)).index(seat)+1
                result.append(dict(seed=910000000+index,profile=profile,seat=seat,completed=True,
                    ledger_passed=True,final_scores=scores,rank=rank,net_points=scores[seat]-25000,
                    hands=4+index%2,wins=1,dealin_hands=0))
    return result


class ConfirmationStatisticsTests(unittest.TestCase):
    def test_same_recorded_outcomes_are_not_automatically_a_parity_claim(self):
        data=rows();result=compare(data,copy.deepcopy(data),910000000)
        self.assertTrue(result['joint_primary_lower_bounds_nonnegative'])
        self.assertTrue(result['degenerate_zero_difference_not_a_parity_proof'])
        self.assertFalse(result['task_completion_or_release_authorized'])
        self.assertEqual(result['independent_seed_groups'],32)

    def test_missing_duplicate_truncated_or_inconsistent_games_are_rejected(self):
        data=rows()
        for bad in (data[:-1],data[:-1]+[data[0]]):
            with self.assertRaises(ValueError):matrix(bad,910000000)
        for key,value in (('completed',False),('ledger_passed',False),('net_points',123),
            ('rank',4),('hands',0),('wins',99)):
            bad=copy.deepcopy(data);bad[0][key]=value
            with self.assertRaises(ValueError):matrix(bad,910000000)

    def test_t_interval_uses32_whole_seed_means_instead_of256_games(self):
        values=[float(i-16) for i in range(32)]
        result=interval(values)
        radius=T_CRITICAL_DF31_975*statistics.stdev(values)/math.sqrt(32)
        self.assertAlmostEqual(result['ci95'][0],statistics.mean(values)-radius)
        self.assertEqual(result['df'],31)
        with self.assertRaises(ValueError):interval(values*8)

    def test_cluster_weighting_and_match_hand_denominators_stay_explicit(self):
        base=rows();candidate=copy.deepcopy(base)
        for row in candidate:
            row['wins']=2 if row['seed']%2==0 else 1
        result=compare(candidate,base,910000000)
        expected=statistics.mean((c['wins']/c['hands'])-(r['wins']/r['hands']) for c,r in zip(candidate,base))
        self.assertAlmostEqual(result['metrics']['hand_win_rate']['gain'],expected)
        self.assertEqual(result['metrics']['net_points']['gain'],0.)

