import copy
import unittest
from tools.train_mortal_teacher import research_screen, progress_state, paired_summary


def metrics(ce=1.,top2=.25,holding=.9):
    return {'teacher':{'macro_ce':ce},'endgame_primary':top2,
            'old':{'legacy':{'utility_by_stratum':{'hold':holding}}}}


class MortalTrainingTests(unittest.TestCase):
    def test_teacher_agreement_alone_does_not_qualify(self):
        self.assertFalse(research_screen(metrics(.8,.1),metrics())['research_eligible'])
        self.assertTrue(research_screen(metrics(.95,.25),metrics())['research_eligible'])

    def test_observation_and_exposure_prevent_two_update_stop(self):
        c={'observation_new_updates':192,'early_stop_patience':4}
        flags={'research_eligible':False};m=metrics()
        state=progress_state(0,m,flags,None,c,False)
        for i in range(32,193,32):
            state=progress_state(i,m,flags,state,c,True)
            self.assertFalse(state['stop']);self.assertEqual(state['stale'],0)
        for i in range(224,320,32):
            state=progress_state(i,m,flags,state,c,True)
            self.assertFalse(state['stop'])
        self.assertTrue(progress_state(320,m,flags,state,c,True)['stop'])
        self.assertFalse(progress_state(320,m,flags,state,c,False)['stop'])

    def test_distribution_progress_extends_observation_even_if_argmax_unchanged(self):
        c={'observation_new_updates':192,'early_stop_patience':4}
        old=dict(best_ce=1.,selected_update=None,selected_ce=None,stale=3,stop=False)
        new=progress_state(320,metrics(.99),{'research_eligible':False},old,c,True)
        self.assertEqual(new['stale'],0);self.assertFalse(new['stop'])

    def test_complete_game_grid_and_whole_seed_interval(self):
        rows=[]
        for arm in ('parent','candidate'):
            for seed in range(8):
                for profile in ('calling','mortal'):
                    for seat in range(4):
                        rows.append(dict(arm=arm,seed=seed,profile=profile,seat=seat,
                            rank=2 if arm=='candidate' else 3,net_points=0,hands=4,wins=1,dealins=0,
                            choices={},completed=True,ledger_passed=True))
        report=paired_summary(rows,list(range(8)))
        self.assertEqual(report['top2_gain'],1.)
        self.assertEqual(report['whole_seed_ci95'],[1.,1.])
        self.assertFalse(report['independent_confirmation'])
        with self.assertRaises(ValueError): paired_summary(rows[:-1],list(range(8)))
        altered=copy.deepcopy(rows);altered[0]['completed']=False
        with self.assertRaises(ValueError): paired_summary(altered,list(range(8)))


if __name__=='__main__': unittest.main()
