import copy
import json
from pathlib import Path
import unittest

from tools.train_endgame_fourth_adamw import validate_config, progress_state, scheduled_lr, combine_source_gradients


def config():
    return json.loads((Path(__file__).resolve().parents[2]/'configs/training/endgame-fourth-adamw-v1.json').read_text())


def metrics(hard=.2, soft=.18, tail=.6):
    return {'fourth': {'top2':hard, 'soft_top2':soft, 'fourth':tail}}


class FourthAdamWTests(unittest.TestCase):
    def test_flat_validation_has_32_step_observation_and_four_later_checks(self):
        cfg=config();current=metrics();guard={'eligible_nonzero':False}
        state=progress_state(0,current,guard,None,cfg)
        for step in range(8,64,8):
            state=progress_state(step,current,guard,state,cfg)
            self.assertFalse(state['stop'])
        state=progress_state(64,current,guard,state,cfg)
        self.assertTrue(state['stop']);self.assertEqual(state['selected_step'],0)

    def test_soft_progress_resets_patience_without_qualifying_a_model(self):
        cfg=config();guard={'eligible_nonzero':False}
        state=progress_state(0,metrics(),guard,None,cfg)
        for step in range(8,129,8):
            state=progress_state(step,metrics(soft=.18+step*.0001),guard,state,cfg)
            self.assertFalse(state['stop']);self.assertEqual(state['selected_step'],0)

    def test_higher_fourth_risk_does_not_veto_an_eligible_top2_gain(self):
        cfg=config();state=progress_state(0,metrics(),{},None,cfg)
        state=progress_state(8,metrics(hard=.25,tail=.9),{'eligible_nonzero':True},state,cfg)
        self.assertEqual(state['selected_step'],8)
        self.assertEqual(state['selected_fourth'],.9)

    def test_failed_common_selection_guards_cannot_promote_soft_or_hard_gain(self):
        cfg=config();state=progress_state(0,metrics(),{},None,cfg)
        state=progress_state(8,metrics(hard=.4),{'eligible_nonzero':False},state,cfg)
        self.assertEqual(state['selected_step'],0)

    def test_fixed_schedule_and_registration(self):
        cfg=config();validate_config(cfg)
        self.assertEqual(scheduled_lr(1,cfg),cfg['learning_rate']/8)
        self.assertEqual(scheduled_lr(8,cfg),cfg['learning_rate'])
        self.assertEqual(scheduled_lr(128,cfg),cfg['learning_rate'])
        for key,value in [('early_stop_patience',2),('per_source_ce_monotonicity',True),
                          ('fourth_escape_constraint','veto'),('max_optimizer_steps',1000)]:
            with self.assertRaises(ValueError):validate_config(dict(cfg,**{key:value}))

    def test_every_source_gradient_contributes_even_when_they_conflict(self):
        import torch
        g={'replay':torch.tensor([1.,0.]),'fourth':torch.tensor([-2.,1.])}
        result=combine_source_gradients(g,{'replay':.5,'fourth':.5})
        self.assertTrue(torch.equal(result,torch.tensor([-.5,.5],dtype=torch.float64)))
        with self.assertRaises(ValueError):combine_source_gradients(g,{'fourth':1.})
        with self.assertRaises(ValueError):combine_source_gradients({'x':torch.tensor([float('nan')])},{'x':1.})


if __name__=='__main__':unittest.main()
