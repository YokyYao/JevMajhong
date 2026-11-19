import copy
import json
from pathlib import Path
import unittest

from tools.train_targeted_endgame import validate_training_config,screen,progress_state,validate_world_trace


def metrics(primary=.25,old=.225,new=.275):
    return dict(primary=primary,soft_primary=.2,old=dict(fourth=dict(top2=old),legacy=dict(
        third_pursuit=dict(candidate=.45),utility_by_stratum=dict(third=.11,hold=.97),original_expert_ce=1.)),
        fresh=dict(fourth=dict(top2=new),third=dict(top2=.4),hold=dict(top2=.8)))


class TargetedTrainingTests(unittest.TestCase):
    def test_actual_outcome_guards_require_gain_and_preserve_new_and_old_monitoring(self):
        initial=metrics();current=metrics(primary=.27,old=.23,new=.31)
        self.assertTrue(screen(current,initial)['research_eligible'])
        for kind in ('old_fourth','new_fourth','third','hold'):
            bad=copy.deepcopy(current)
            if kind=='old_fourth':bad['old']['fourth']['top2']=.20
            elif kind=='new_fourth':bad['fresh']['fourth']['top2']=.1
            elif kind=='third':bad['old']['legacy']['third_pursuit']['candidate']=.44
            else:bad['fresh']['hold']['top2']=.7
            self.assertFalse(screen(bad,initial)['research_eligible'])
        self.assertFalse(screen(current,initial)['release_qualified'])

    def test_soft_progress_extends_learning_without_selecting_weights(self):
        c=json.loads((Path(__file__).resolve().parents[2]/'configs/training/endgame-targeted-refresh-v1.json').read_text())
        validate_training_config(c);m=metrics();state=progress_state(0,m,{},None,c)
        for update in range(16,193,16):
            m['soft_primary']+=.001;state=progress_state(update,m,screen(m,metrics()),state,c)
            self.assertFalse(state['stop']);self.assertIsNone(state['selected_update'])

    def test_observe64_then_three_later_checks(self):
        c=json.loads((Path(__file__).resolve().parents[2]/'configs/training/endgame-targeted-refresh-v1.json').read_text())
        m=metrics();state=progress_state(0,m,{},None,c)
        for update in range(16,112,16):
            state=progress_state(update,m,screen(m,m),state,c);self.assertFalse(state['stop'])
        self.assertTrue(progress_state(112,m,screen(m,m),state,c)['stop'])

    def test_counterfactual_world_policy_pairing_cannot_be_misaligned(self):
        from jevmahjong.endgame.targeted_refresh import world_design
        values=[dict(world=i,**world_design(i),scores=[10000,20000,30000,40000],ranks=[4,3,2,1],
                     completed=True,ledger_passed=True) for i in range(64)]
        validate_world_trace(values)
        bad=copy.deepcopy(values);bad[33]['policy']=1-bad[33]['policy']
        with self.assertRaises(ValueError):validate_world_trace(bad)
        bad=copy.deepcopy(values);bad[4]['ranks']=[1,2,3,4]
        with self.assertRaises(ValueError):validate_world_trace(bad)


if __name__=='__main__':unittest.main()
