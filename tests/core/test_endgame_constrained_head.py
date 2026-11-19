import copy
import json
from pathlib import Path
import unittest
import torch

from jevmahjong.endgame.constrained_head import (projected_descent,guarded_validation,
    source_loss_values,constrained_update,updated_head_metadata)
from tools.audit_endgame_head_gradients import source_gradients
from tools.train_endgame_constrained_head import validate_config
from test_endgame_head_gradients import TinyPointer


class ConstrainedHeadTests(unittest.TestCase):
    def test_projection_preserves_all_local_descent_constraints(self):
        gradients={'a':torch.tensor([1.,0.]),'b':torch.tensor([-.5,1.])}
        snapshot={k:v.clone() for k,v in gradients.items()}
        direction,proof=projected_descent(gradients,{'a':.9,'b':.1})
        for value in gradients.values():
            self.assertGreaterEqual(float((value.double()/value.double().norm())@direction),.01-2e-8)
        self.assertTrue(all(torch.equal(v,snapshot[k]) for k,v in gradients.items()))
        self.assertLess(proof['cycles'],2048)
        with self.assertRaises(ValueError):projected_descent({'a':torch.tensor([1.,0.]),'b':torch.tensor([-1.,0.])},
            {'a':.5,'b':.5},max_cycles=16)

    def test_exact_source_check_accepts_descent_and_restores_a_bad_step(self):
        head=TinyPointer();value={'decide':torch.tensor([1.,0.]),'options':torch.tensor([[0.,1.],[0.,-1.]]),'keys':['x','y']}
        rows=[{'_meta':{'id':key},'questions':{'action':{'src':key,'target':{'x':1.,'y':0.}}}} for key in ('a','b')]
        features={'a':value,'b':value};gradients,losses=source_gradients(head,rows,features)
        direction,_=projected_descent(gradients,{'a':.5,'b':.5})
        accepted=constrained_update(head,rows,features,direction,losses,relative_step=.05,tolerance=0.)
        self.assertTrue(accepted['accepted']);self.assertTrue(all(accepted['losses'][k]<losses[k] for k in losses))
        self.assertEqual(accepted['before_losses'],losses)
        before={k:v.clone() for k,v in head.state_dict().items()};gradients,losses=source_gradients(head,rows,features)
        direction,_=projected_descent(gradients,{'a':.5,'b':.5})
        rejected=constrained_update(head,rows,features,-direction,losses,relative_step=.01,max_halvings=3,tolerance=0.)
        self.assertFalse(rejected['accepted'])
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in head.state_dict().items()))

    def test_guarded_selection_does_not_compensate_placement_with_expert_ce(self):
        initial={'original_expert_ce':1.,'utility_by_stratum':{'third':.1,'hold':.97},
            'kl':{k:{'mean':v} for k,v in (('original',.005),('trailing',.02),('leading',0.))},
            'fourth_escape':{'existing':{'candidate':.412,'original_policy':.436},
                             'native_fourth':{'candidate':.35,'original_policy':.35}},
            'third_pursuit':{'candidate':.448,'original_policy':.433}}
        config={'enforce_placement_strata_guard':True,'enforce_win_stratum_guard':False,
            'reference_kl_guard_relative_to_start':True,'fourth_escape_constraint':True,
            'reference_kl_guard':.05,'native_reference_kl_guard':.05,'leading_native_reference_kl_guard':.05}
        zero=guarded_validation(0,initial,initial,config)
        self.assertFalse(zero['raw_constraints_met']);self.assertTrue(zero['all_guards_passed'])
        good=copy.deepcopy(initial);good['fourth_escape']['existing']['candidate']=.44
        self.assertTrue(guarded_validation(1,good,initial,config)['eligible_nonzero'])
        self.assertFalse(guarded_validation(1,good,initial,config,exposure_met=False)['eligible_nonzero'])
        bad=copy.deepcopy(good);bad['original_expert_ce']=.8;bad['utility_by_stratum']['hold']=.92
        self.assertFalse(guarded_validation(1,bad,initial,config)['eligible_nonzero'])

    def test_registration_requires_explicit_fit_enable_and_fixed_guards(self):
        path=Path(__file__).resolve().parents[2]/'configs/training/endgame-placement-v15.json'
        config=json.loads(path.read_text());validate_config(config)
        with self.assertRaises(ValueError):validate_config(config,fit=True)
        validate_config(dict(config,fit_registered=True),fit=True)
        for change in ({'max_optimizer_steps':128},{'relative_head_l2_step':.05},
                       {'guard_protocol':'relaxed'},{'independent_confirmation':True}):
            with self.assertRaises(ValueError):validate_config(dict(config,**change))

    def test_checkpoint_metadata_updates_real_step_without_nesting_extra(self):
        parent={'base':'qwen','head':{'old':1},'optimizer_step':2088,'args':{'lora':16}}
        new=updated_head_metadata(parent,{'new':2},3)
        self.assertEqual(new['optimizer_step'],3);self.assertEqual(new['formal_parent_step'],2088)
        self.assertNotIn('extra',new);self.assertEqual(new['args'],parent['args'])
        self.assertEqual(parent['head'],{'old':1});self.assertEqual(parent['optimizer_step'],2088)


if __name__=='__main__':unittest.main()
