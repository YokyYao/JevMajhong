import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from safetensors.torch import save_file
from jevmahjong.training.base_lora import package_digest
from jevmahjong.training.initialization import (check_groups,exact_loaded_tensors,
    check_initialization,verified_parent)


class LoraInitializationTests(unittest.TestCase):
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
