import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from safetensors.torch import save_file
from jevmahjong.training.base_lora import package_digest
from jevmahjong.training.initialization import (check_groups,exact_loaded_tensors,
    check_initialization,verified_parent,lineage_training_files,bind_parent)
from jevmahjong.core.io import digest
from unittest.mock import patch


class LoraInitializationTests(unittest.TestCase):
    def test_continued_parent_requires_and_uses_explicit_archived_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source';package=source/'jevmahjong';module=package/'evaluation'
            module.mkdir(parents=True);(package/'__init__.py').write_text('');(module/'__init__.py').write_text('')
            parent=dict(model={'identity_sha256':'continued-final'},receipt={},selected_step=512,
                training_config={'initialization':'continued','base':'base','base_revision':'rev'},training_protocol={})
            (module/'unified_contract.py').write_text('def checked_candidate(run):\n    return '+repr(parent)+'\n')
            (module/'checkpoint_selection.py').write_text('def candidate_for_evaluation(cfg):\n'
                "    if cfg['candidate_selection']['sha256']!='bound-declaration': raise ValueError('Declaration changed')\n"
                '    return '+repr(parent)+'\n')
            cfg=dict(out=str(root/'new'),base='base',base_revision='rev',parent_initialization=dict(
                source=str(source),source_sha256=package_digest(package),run=str(root/'parent'),
                identity_sha256='continued-final',selected_step=512))
            with self.assertRaisesRegex(ValueError,'explicit registered checkpoint'):verified_parent(cfg)
            cfg['parent_initialization']['candidate_selection']=dict(method='registered_final_training_checkpoint',manifest='decl.json',sha256='bound-declaration')
            self.assertEqual(verified_parent(cfg),parent)
            wrong=copy.deepcopy(cfg);wrong['parent_initialization']['candidate_selection']['sha256']='wrong'
            with self.assertRaisesRegex(ValueError,'Declaration changed'):verified_parent(wrong)
            wrong=copy.deepcopy(cfg);wrong['parent_initialization']['selected_step']=0
            with self.assertRaisesRegex(ValueError,'identity differs'):verified_parent(wrong)
            wrong=copy.deepcopy(cfg);wrong['parent_initialization']['candidate_selection']['method']='latest'
            with self.assertRaisesRegex(ValueError,'Explicit registered'):verified_parent(wrong)

    def test_changed_holdout_is_checked_against_every_ancestor_not_only_immediate_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            def data(name,groups):
                f=root/name;f.write_text(''.join(json.dumps({'_meta':{'group_id':g}})+'\n' for g in groups));return str(f)
            grand=data('grand.jsonl',['older-match']);parent_file=data('parent.jsonl',['parent-match'])
            current=data('current.jsonl',['new-match']);held=data('held.jsonl',['held-match'])
            protocol=dict(config={'initialization':'continued'},trainer_args={'data':parent_file},
                input_sha256={p:digest(p) for p in [grand,parent_file]},code_sha256={},
                parent_initialization={'parent_training_file':grand})
            parent=dict(model={'directory':str(root/'weights'),'identity_sha256':'parent'},selected_step=512,
                receipt={},training_protocol=protocol)
            args=dict(init_from=str(root/'weights'),data=current,eval_data=held)
            with patch('jevmahjong.training.initialization.verified_parent',return_value=parent):
                bound=bind_parent({},args)
                self.assertEqual(bound['ancestor_training_files'],[parent_file,grand])
                self.assertEqual(bound['group_proof']['ancestor_validation_overlap'],0)
                data('held.jsonl',['older-match'])
                with self.assertRaisesRegex(ValueError,'earlier ancestor'):bind_parent({},args)
            # A third generation must retain both earlier training inputs.
            protocol['parent_initialization']['ancestor_training_files']=[grand,parent_file]
            self.assertEqual(lineage_training_files(protocol),[parent_file,grand])
            protocol['input_sha256'].pop(grand)
            with self.assertRaisesRegex(ValueError,'unbound'):lineage_training_files(protocol)

    def test_every_loaded_parent_tensor_is_checked_not_only_names_and_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)
            adapter={'lora_A.weight':torch.ones(2,3),'lora_B.weight':torch.ones(3,2)}
            head={'q.weight':torch.ones(2,3),'q.bias':torch.ones(2)}
            save_file(adapter,str(p/'adapter_model.safetensors'));torch.save({'head':head},p/'head.pt')
            self.assertTrue(exact_loaded_tensors(adapter,head,p)['parent_adapter_exact'])
            bad={k:v.clone() for k,v in adapter.items()};bad['lora_B.weight'][0,0]+=1
            with self.assertRaisesRegex(ValueError,'tensors differ'):exact_loaded_tensors(bad,head,p)
            bad_head={k:v.clone() for k,v in head.items()};bad_head['q.bias'][0]+=1
            with self.assertRaisesRegex(ValueError,'tensors differ'):exact_loaded_tensors(adapter,bad_head,p)
            with self.assertRaisesRegex(ValueError,'tensors differ'):exact_loaded_tensors({'lora_A.weight':adapter['lora_A.weight']},head,p)

    def test_new_validation_must_exclude_parent_and_current_training_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)
            def data(name,groups):
                f=p/name;f.write_text(''.join(json.dumps({'_meta':{'group_id':g}})+'\n' for g in groups));return f
            ancestor=data('ancestor',['old-match']);train=data('train',['new-match']);held=data('held',['held-match'])
            self.assertEqual(check_groups(ancestor,train,held)['parent_validation_overlap'],0)
            for overlap in ['old-match','new-match']:
                held=data('held',[overlap])
                with self.assertRaisesRegex(ValueError,'overlaps parent or current'):check_groups(ancestor,train,held)

    def test_continuation_cannot_claim_zero_initialized_lora_as_proof_of_updates(self):
        cfg={'initialization':'continued','parent_initialization':{'identity_sha256':'parent'}}
        fit=dict(status='continued_base_LoRA_training_complete',old_adapters_loaded=True,external_residual_networks=0,pretrained_base_frozen=True)
        audit=dict(initialization='continued',old_adapters_loaded=True,external_residual_networks=0,
            pretrained_base_requires_grad=False,parent_adapter_exact=True,parent_head_exact=True,
            parent_identity_sha256='parent',parent_group_proof={'overlap':0},new_optimizer=True)
        parent={'model':{'identity_sha256':'parent'},'group_proof':{'overlap':0}}
        proof=dict(head_parameters_changed=True,continued_LoRA_parameters_changed=True)
        check_initialization(cfg,fit,audit,proof,parent)
        for change in [dict(continued_LoRA_parameters_changed=False),dict(head_parameters_changed=False)]:
            with self.assertRaisesRegex(ValueError,'unverified'):check_initialization(cfg,fit,audit,dict(proof,**change),parent)
        for change in [dict(parent_adapter_exact=False),dict(parent_head_exact=False),dict(new_optimizer=False),dict(parent_identity_sha256='other'),dict(parent_group_proof={'overlap':1})]:
            with self.assertRaisesRegex(ValueError,'unverified'):check_initialization(cfg,fit,dict(audit,**change),proof,parent)
        fresh_proof=dict(head_parameters_changed=True,fresh_LoRA_B_now_nonzero=True)
        with self.assertRaisesRegex(ValueError,'unverified'):check_initialization(cfg,fit,audit,fresh_proof,parent)

    def test_fresh_mode_still_requires_fresh_adapter_and_frozen_base(self):
        fit=dict(status='fresh_base_LoRA_training_complete',old_adapters_loaded=False,external_residual_networks=0,pretrained_base_frozen=True)
        audit=dict(old_adapters_loaded=False,external_residual_networks=0,pretrained_base_requires_grad=False,
            fresh_lora_B_zero=True,fresh_lora_A_nonzero=True)
        proof=dict(head_parameters_changed=True,fresh_LoRA_B_now_nonzero=True)
        check_initialization({},fit,audit,proof)
        with self.assertRaises(ValueError):check_initialization({},fit,dict(audit,old_adapters_loaded=True),proof)

    def test_parent_uses_its_own_hash_bound_source_in_an_isolated_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source';package=source/'jevmahjong';module=package/'evaluation'
            module.mkdir(parents=True);(package/'__init__.py').write_text('');(module/'__init__.py').write_text('')
            parent=dict(model={'identity_sha256':'bound-parent'},receipt={},selected_step=9,
                training_config={'base':'base','base_revision':'rev'},training_protocol={})
            verifier=module/'unified_contract.py'
            verifier.write_text('def checked_candidate(run):\n    return '+repr(parent)+'\n')
            cfg=dict(out=str(root/'new'),base='base',base_revision='rev',parent_initialization=dict(
                source=str(source),source_sha256=package_digest(package),run=str(root/'parent'),
                identity_sha256='bound-parent',selected_step=9))
            self.assertEqual(verified_parent(cfg),parent)
            wrong=copy.deepcopy(cfg);wrong['parent_initialization']['identity_sha256']='wrong'
            with self.assertRaisesRegex(ValueError,'identity differs'):verified_parent(wrong)
            verifier.write_text(verifier.read_text()+'# modified\n')
            with self.assertRaisesRegex(ValueError,'source changed'):verified_parent(cfg)


if __name__=='__main__':unittest.main()
