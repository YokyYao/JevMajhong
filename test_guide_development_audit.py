import copy
import statistics
import unittest

from guide_development_audit import aggregate,SUITES,POLICIES
from scripts.audit_guide_calls import compare


class GuideDevelopmentAuditTests(unittest.TestCase):
    def setUp(self):
        self.cfg={'development_seed_start':10,'development_seeds':3}
        self.rows=[]
        for suite in SUITES:
            for seed,gain in zip(range(10,13),(100,700,1000)):
                for seat in range(4):
                    for policy in POLICIES:
                        score=25000+(gain if policy=='candidate' else 0)
                        self.rows.append({'suite':suite,'seed':seed,'seat':seat,'policy':policy,
                            'score':score,'rank':2,'hands':5,'hand_wins':1,'dealins':0,'calls':1,
                            'settlement_ledger_passed':True,'point_components':{'fixture_net':score-25000}})

    def test_primary_clusters_shared_suites_and_seats_by_whole_wall(self):
        from scipy.stats import t
        result=aggregate(self.rows,self.cfg);primary=result['overall']['candidate_minus_reference']
        half=float(t.ppf(.975,2))*statistics.stdev([100,700,1000])/3**.5
        self.assertEqual(primary['gain'],600)
        self.assertEqual(primary['independent_wall_streams'],3)
        self.assertAlmostEqual(primary['ci95'][0],600-half)
        self.assertTrue(result['development_positive_signal'])
        self.assertFalse(result['meets_development_stability_condition'])

    def test_incomplete_duplicate_or_foreign_game_is_rejected(self):
        for rows in (self.rows[:-1],self.rows+[self.rows[0]],self.rows[:-1]+[self.rows[0]]):
            with self.assertRaises(ValueError):aggregate(rows,self.cfg)

    def test_score_and_summary_tampering_are_rejected(self):
        rows=copy.deepcopy(self.rows);rows[0]['score']+=1000
        with self.assertRaisesRegex(ValueError,'ledger mismatch'):aggregate(rows,self.cfg)
        result=aggregate(self.rows,self.cfg);changed=copy.deepcopy(result)
        changed['overall']['candidate_minus_reference']['ci95']=[500,700]
        with self.assertRaisesRegex(ValueError,'differs from raw'):compare(changed,result)


if __name__=='__main__':unittest.main()
