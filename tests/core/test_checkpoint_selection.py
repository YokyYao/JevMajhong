import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from safetensors.torch import save_file
from jevmahjong.core.io import digest
from jevmahjong.evaluation.checkpoint_selection import (METHOD,declare_final_candidate,candidate_for_evaluation)
from jevmahjong.evaluation.unified_contract import bundle
from jevmahjong.training.base_lora import validation_profile


def fixture(root):
    run=root/'runs/fit';trainer=run/'training-run';zero=trainer/'checkpoints/step-000000';final=trainer/'checkpoints/step-000384'
    zero.mkdir(parents=True);final.mkdir(parents=True)
    for directory,value in [(zero,1.),(final,2.)]:
        save_file({'lora_A.weight':torch.ones(2,2),'lora_B.weight':torch.full((2,2),value)},str(directory/'adapter_model.safetensors'))
        torch.save(dict(base='unit-base',base_revision='revision',lora=1,head_dim=2,head_residual_dim=0,
            head={'q.weight':torch.full((2,2),value),'q.bias':torch.ones(2)}),directory/'head.pt')
        for name in ('adapter_config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja'):
            (directory/name).write_text(json.dumps({'r':1,'target_modules':['q_proj','v_proj']}) if name=='adapter_config.json' else 'same-'+name)
    record=dict(_meta=dict(id='held',phase='closed_discard',label_type='expert_replay'),
        questions={'action':dict(criteria={'a0':{'type':'dahai'},'a1':{'type':'dahai'}},label='a0')})
    rows=[dict(id='held',keys=['a0','a1'],p=[.6,.4])];weights={'closed_discard':1.}
    profile=validation_profile([record],rows,weights)
    rowfile=trainer/'training_eval/step-000384/development/rows.json';rowfile.parent.mkdir(parents=True);rowfile.write_text(json.dumps(rows))
    (trainer/'training_metrics.json').write_text(json.dumps({'optimizer_steps':384}))
    # The original CE-selected parent remains zero; the trained checkpoint is
    # a declared diagnostic, not a rewritten training selection.
    (run/'result.json').write_text(json.dumps({'optimizer_steps':384,'selected_step':0}))
    (run/'protocol.json').write_text(json.dumps({'config':{'out':str(run)}}))
    (run/'validation-curve.json').write_text(json.dumps([dict(step=384,**profile)]))
    base=dict(model_id='fit',model=bundle(zero),selected_step=0,
        training_protocol={'trainer_args':{'out':str(trainer)}},training_config={'validation_phase_weights':weights},
        validation_records=[record],receipt={str(run/f):digest(run/f) for f in ('result.json','protocol.json','validation-curve.json')},
        eligible_for_development=False,initialization_mode='continued')
    return run,base,final


class CheckpointSelectionTests(unittest.TestCase):
    def test_target_module_order_is_not_architecture_but_membership_and_tokenizer_are(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run,base,final=fixture(root)
            (final/'adapter_config.json').write_text(json.dumps({'r':1,'target_modules':['v_proj','q_proj']}))
            with patch('jevmahjong.evaluation.checkpoint_selection.checked_candidate',return_value=base):
                self.assertEqual(declare_final_candidate(run,root/'good.json',[root/'runs'])['step'],384)
                (final/'adapter_config.json').write_text(json.dumps({'r':1,'target_modules':['v_proj','k_proj']}))
                with self.assertRaisesRegex(ValueError,'adapter architecture'):
                    declare_final_candidate(run,root/'bad-module.json',[root/'runs'])
                (final/'adapter_config.json').write_text(json.dumps({'r':2,'target_modules':['q_proj','v_proj']}))
                with self.assertRaisesRegex(ValueError,'adapter architecture'):
                    declare_final_candidate(run,root/'bad-rank.json',[root/'runs'])
                (final/'adapter_config.json').write_text(json.dumps({'r':1,'target_modules':['q_proj','v_proj']}))
                (final/'tokenizer.json').write_text('changed')
                with self.assertRaisesRegex(ValueError,'tokenization'):
                    declare_final_candidate(run,root/'bad-tokenizer.json',[root/'runs'])

    def test_exact_predeclared_budget_can_use_final_even_when_ce_selected_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run,base,final=fixture(root);target=root/'decl.json'
            base['training_config'].update(evaluation_candidate_rule='final_registered_budget_v1',registered_final_step=384)
            base['training_protocol']['trainer_args'].update(max_steps=384,early_stop_patience=0)
            base.update(selected_step=384,model=bundle(final))
            with patch('jevmahjong.evaluation.checkpoint_selection.checked_candidate',return_value=base):
                d=declare_final_candidate(run,target,[root/'runs'])
                cfg=dict(candidate_run=str(run),candidate_selection=dict(method=METHOD,manifest=str(target),sha256=digest(target)))
                self.assertEqual(candidate_for_evaluation(cfg)['training_validation_selected_step'],384)
                self.assertEqual(d['step'],384)
                base['training_config']['registered_final_step']=512
                with self.assertRaisesRegex(ValueError,'registered fixed training budget'):candidate_for_evaluation(cfg)

    def test_declared_final_probe_preserves_ce_parent_and_uses_real_final_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run,base,_=fixture(root);target=root/'declaration.json'
            with patch('jevmahjong.evaluation.checkpoint_selection.checked_candidate',return_value=base):
                self.assertIs(candidate_for_evaluation({'candidate_run':str(run)}),base)
                declaration=declare_final_candidate(run,target,[root/'runs'])
                cfg=dict(candidate_run=str(run),candidate_selection=dict(method=METHOD,manifest=str(target),sha256=digest(target)))
                result=candidate_for_evaluation(cfg)
            self.assertEqual(declaration['step'],384);self.assertEqual(result['selected_step'],384)
            self.assertEqual(result['training_validation_selected_step'],0)
            self.assertEqual(result['selected_validation_rows'][0]['p'],[.6,.4])
            self.assertTrue(result['eligible_for_development']);self.assertEqual(base['selected_step'],0)
            self.assertEqual(json.loads((run/'result.json').read_text())['selected_step'],0)
            self.assertIn(str(target.resolve()),result['receipt'])

    def test_cannot_substitute_a_historical_step_even_with_a_new_manifest_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run,base,_=fixture(root);target=root/'declaration.json'
            with patch('jevmahjong.evaluation.checkpoint_selection.checked_candidate',return_value=base):
                declare_final_candidate(run,target,[root/'runs']);d=json.loads(target.read_text());d['step']=128;target.write_text(json.dumps(d))
                cfg=dict(candidate_run=str(run),candidate_selection=dict(method=METHOD,manifest=str(target),sha256=digest(target)))
                with self.assertRaisesRegex(ValueError,'Declared final candidate changed'):candidate_for_evaluation(cfg)

    def test_registration_rejects_prior_full_match_evidence_and_nonzero_ce_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run,base,_=fixture(root);old=root/'runs/old-evaluation';old.mkdir()
            (old/'protocol.json').write_text(json.dumps({'config':{'candidate_run':str(run)}}));(old/'games-private.jsonl').write_text('{}\n')
            with patch('jevmahjong.evaluation.checkpoint_selection.checked_candidate',return_value=base):
                with self.assertRaisesRegex(ValueError,'already has full-match evidence'):declare_final_candidate(run,root/'decl.json',[root/'runs'])
            self.assertFalse((root/'decl.json').exists())
            bad=dict(base,selected_step=128)
            with patch('jevmahjong.evaluation.checkpoint_selection.checked_candidate',return_value=bad):
                with self.assertRaisesRegex(ValueError,'CE-selected zero'):declare_final_candidate(run,root/'decl.json',[root/'runs'])

    def test_older_step_cannot_be_presented_as_final_by_editing_completion_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run,base,_=fixture(root)
            (run/'result.json').write_text(json.dumps({'optimizer_steps':128,'selected_step':0}))
            (run/'training-run/training_metrics.json').write_text(json.dumps({'optimizer_steps':128}))
            with patch('jevmahjong.evaluation.checkpoint_selection.checked_candidate',return_value=base):
                with self.assertRaisesRegex(ValueError,'completed validation trace'):declare_final_candidate(run,root/'decl.json',[root/'runs'])

    def test_changed_final_tensor_after_declaration_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run,base,final=fixture(root);target=root/'declaration.json'
            with patch('jevmahjong.evaluation.checkpoint_selection.checked_candidate',return_value=base):
                declare_final_candidate(run,target,[root/'runs'])
                saved=torch.load(final/'head.pt',weights_only=True);saved['head']['q.weight'][0,0]+=1;torch.save(saved,final/'head.pt')
                cfg=dict(candidate_run=str(run),candidate_selection=dict(method=METHOD,manifest=str(target),sha256=digest(target)))
                with self.assertRaisesRegex(ValueError,'Declared final candidate changed'):candidate_for_evaluation(cfg)

    def test_unchanged_adapter_or_changed_tokenizer_cannot_be_called_new_final_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run,base,final=fixture(root);initial=Path(base['model']['directory'])
            with patch('jevmahjong.evaluation.checkpoint_selection.checked_candidate',return_value=base):
                (final/'tokenizer.json').write_text('different')
                with self.assertRaisesRegex(ValueError,'tokenization'):declare_final_candidate(run,root/'decl.json',[root/'runs'])
                (final/'tokenizer.json').write_bytes((initial/'tokenizer.json').read_bytes())
                (final/'adapter_model.safetensors').write_bytes((initial/'adapter_model.safetensors').read_bytes())
                with self.assertRaisesRegex(ValueError,'adapter is unchanged'):declare_final_candidate(run,root/'decl.json',[root/'runs'])


if __name__=='__main__':unittest.main()
