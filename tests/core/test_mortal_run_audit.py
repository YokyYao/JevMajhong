import copy
import unittest
from tools.audit_mortal_teacher_run import account,audit_games


def draw_log():
    return [{'type':'start_game'},{'type':'start_kyoku','scores':[25000]*4,'kyotaku':0},
            {'type':'ryukyoku','deltas':[0,0,0,0]},{'type':'end_kyoku'},{'type':'end_game'}]


class IndependentAuditTests(unittest.TestCase):
    def test_terminal_score_and_tie_order_are_recounted(self):
        self.assertEqual(account(draw_log(),[25000]*4)['ranks'],[1,2,3,4])
        with self.assertRaises(ValueError): account(draw_log()[:-1],[25000]*4)
        with self.assertRaises(ValueError): account(draw_log(),[26000,24000,25000,25000])

    def test_final_unclaimed_deposit_is_awarded_without_fabricated_score(self):
        log=draw_log();log.insert(2,{'type':'reach_accepted','actor':3})
        self.assertEqual(account(log,[26000,25000,25000,24000])['final_scores'],[26000,25000,25000,24000])

    def test_double_ron_counts_events_and_hands_separately(self):
        log=draw_log();log[2:3]=[{'type':'hora','actor':1,'target':0,'deltas':[-1000,1000,0,0]},
                                   {'type':'hora','actor':2,'target':0,'deltas':[-1000,0,1000,0]}]
        result=account(log,[23000,26000,26000,25000])
        self.assertEqual(result['deal_events'][0],2);self.assertEqual(result['deal_hands'][0],1)
        bad=copy.deepcopy(log);bad[2]['deltas'][0]=0
        with self.assertRaises(ValueError): account(bad,[23000,26000,26000,25000])

    def test_duplicate_grid_cannot_pass_using_completed_flags(self):
        rows=[]
        for arm in ('parent','candidate'):
            for seed in range(8):
                for profile in ('calling','mortal'):
                    for seat in range(4):
                        rows.append(dict(arm=arm,seed=seed,profile=profile,seat=seat,completed=True,ledger_passed=True,
                            mjai_log=draw_log(),final_scores=[25000]*4,rank=seat+1,net_points=0,
                            hands=1,wins=0,dealins=0,choices={}))
        result=audit_games(rows,list(range(8)))
        self.assertEqual(result['checked_complete_games'],128);self.assertEqual(result['top2_gain'],0.)
        bad=copy.deepcopy(rows);bad[-1]=copy.deepcopy(bad[0])
        with self.assertRaises(ValueError): audit_games(bad,list(range(8)))


if __name__=='__main__':unittest.main()
