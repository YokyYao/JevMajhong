import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from jevmahjong.training.finalization import fixed_budget_selection,initial_snapshot
from jevmahjong.training.recover_fit import recover


class FitFinalizationTests(unittest.TestCase):
    def recipe(self):
        cfg=dict(evaluation_candidate_rule='final_registered_budget_v1',registered_final_step=4)
        args=dict(max_steps=4,early_stop_patience=0,eval_every_steps=2,eval_before_start=True)
        metrics=dict(optimizer_steps=4,early_stopped=False,selected_checkpoint=None)
        curve=[dict(step=s,weighted_target_ce=v) for s,v in [(0,1.),(2,1.1),(4,1.05)]]
        return cfg,args,metrics,curve

    def test_fixed_budget_preserves_zero_ce_minimum_without_faking_upstream_selection(self):
        cfg,args,metrics,curve=self.recipe();before=copy.deepcopy((cfg,args,metrics,curve))
        selected=fixed_budget_selection(cfg,args,metrics,curve)
        self.assertEqual(selected['step'],0)
        self.assertFalse(selected['upstream_early_stopping_used'])
        self.assertEqual(selected['registered_final_step'],4)
        self.assertEqual((cfg,args,metrics,curve),before)
        curve[-1]['weighted_target_ce']=.9
        self.assertEqual(fixed_budget_selection(cfg,args,metrics,curve)['step'],4)

    def test_incomplete_or_changed_registered_fit_is_rejected(self):
        for key,value in [('optimizer_steps',2),('early_stopped',True)]:
            cfg,args,metrics,curve=self.recipe();metrics[key]=value
            with self.assertRaises(ValueError):fixed_budget_selection(cfg,args,metrics,curve)
        cfg,args,metrics,curve=self.recipe()
        for broken in [curve[1:],curve[:1]+curve[2:],list(reversed(curve)),curve+[curve[-1]]]:
            with self.assertRaisesRegex(ValueError,'trace'):fixed_budget_selection(cfg,args,metrics,broken)
        args['early_stop_patience']=5
        with self.assertRaises(ValueError):fixed_budget_selection(cfg,args,metrics,curve)

    def test_true_initial_snapshot_survives_later_parameter_updates(self):
        with tempfile.TemporaryDirectory() as temp:
            head=torch.nn.Linear(2,2);original=head.weight.detach().clone()
            def save_adapter(target,**kwargs):
                (target/'adapter_model.safetensors').write_bytes(b'initial adapter')
            model=SimpleNamespace(lm=SimpleNamespace(save_pretrained=save_adapter),head=head,
                special_embeddings=False,backbone_adapter='text',branch_mode='rows',verbalizers=[])
            tokenizer=SimpleNamespace(save_pretrained=lambda target:(target/'tokenizer.json').write_text('{}'))
            trainer=SimpleNamespace(Meta=lambda **kw:kw,write_meta=lambda path,meta:torch.save(meta,path/'head.pt'))
            args=dict(base='base',base_revision='rev',lora=16,early_stop_patience=0)
            initial_snapshot(model,tokenizer,args,trainer,Path(temp),0)
            target=Path(temp)/'checkpoints/step-000000'
            with torch.no_grad():head.weight.add_(1.)
            saved=torch.load(target/'head.pt',weights_only=True)
            self.assertTrue(torch.equal(saved['head']['weight'],original))
            self.assertEqual(saved['extra']['optimizer_step'],0)
            self.assertEqual(saved['extra']['snapshot_origin'],'wrapper_before_first_optimizer_update')
            initial_snapshot(model,tokenizer,args,trainer,Path(temp),2)
            self.assertFalse((Path(temp)/'checkpoints/step-000002').exists())
            with self.assertRaises(FileExistsError):initial_snapshot(model,tokenizer,args,trainer,Path(temp),0)

    def test_early_stopping_snapshot_stays_owned_by_upstream(self):
        with tempfile.TemporaryDirectory() as temp:
            initial_snapshot(None,None,{'early_stop_patience':1},None,Path(temp),0)
            self.assertEqual(list(Path(temp).iterdir()),[])

    def test_recovery_refuses_unreconstructable_fresh_initialization(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);trainer=root/'training-run'
            (root/'protocol.json').write_text(json.dumps(dict(config={'initialization':'fresh'},trainer_args={'out':str(trainer)})))
            with self.assertRaisesRegex(ValueError,'fresh initialization'):recover(root,root/'log',{})
            self.assertFalse(trainer.exists())


if __name__=='__main__':unittest.main()
