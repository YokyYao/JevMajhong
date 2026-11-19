import unittest
import torch
from jevmahjong.endgame.fit import (configure_parameter_scope,validate_parameter_updates,
    training_compute_options,verify_frozen_readout)


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base=torch.nn.Linear(3,3)
        self.lora_A=torch.nn.Linear(3,3,bias=False)
        self.dropout=torch.nn.Dropout(.5)
        self.base.requires_grad_(False)

    def forward(self,x):return self.base(x)+self.dropout(self.lora_A(x))

    def enable_test_input_grads(self):
        self.input_hook=self.base.register_forward_hook(lambda module,args,out:out.requires_grad_(True))
        self.gradient_checkpointing=True

    def disable_input_require_grads(self):self.input_hook.remove()

    def gradient_checkpointing_disable(self):self.gradient_checkpointing=False


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.lm=Encoder();self.head=torch.nn.Linear(3,2)

    def forward(self,x):return self.head(self.lm(x))


class ParameterScopeTests(unittest.TestCase):
    def test_head_only_optimizer_changes_readout_without_changing_encoder(self):
        torch.manual_seed(20261119);model=TinyPolicy().train()
        before={k:v.clone() for k,v in model.state_dict().items()}
        proof=configure_parameter_scope(model,'head_only')
        self.assertFalse(model.lm.training);self.assertTrue(model.head.training)
        self.assertEqual(proof['trainable_encoder_parameters'],0)
        self.assertEqual(proof['trainable_parameters'],proof['head_parameters'])
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        loss=model(torch.ones(2,3)).square().mean();loss.backward();optimizer.step()
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.state_dict().items() if k.startswith('lm.')))
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in model.state_dict().items() if k.startswith('head.')))

    def test_default_joint_scope_keeps_existing_lora_trainability(self):
        model=TinyPolicy().train();before={k:p.requires_grad for k,p in model.named_parameters()}
        proof=configure_parameter_scope(model,'lora_and_head')
        self.assertEqual(before,{k:p.requires_grad for k,p in model.named_parameters()})
        self.assertGreater(proof['trainable_encoder_parameters'],0)
        with self.assertRaises(ValueError):configure_parameter_scope(model,'unknown')

    def test_update_proof_rejects_the_wrong_parameter_group(self):
        validate_parameter_updates('head_only',False,True)
        validate_parameter_updates('lora_and_head',True,True)
        for scope,adapter,head in (('head_only',True,True),('head_only',False,False),
                                   ('lora_and_head',False,True),('lora_and_head',True,False)):
            with self.assertRaises(ValueError):validate_parameter_updates(scope,adapter,head)

    def test_frozen_encoder_removes_input_grad_hook_but_head_still_learns(self):
        model=TinyPolicy().train();model.lm.enable_test_input_grads()
        configure_parameter_scope(model,'head_only',True)
        features=model.lm(torch.ones(3));options=model.lm(torch.ones(2,3))
        self.assertFalse(model.lm.gradient_checkpointing)
        proof=verify_frozen_readout(model,(features,options))
        self.assertFalse(proof['encoder_features_require_grad'])
        model.head(features).square().sum().backward()
        self.assertTrue(all(p.grad is None for p in model.lm.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.head.parameters()))

    def test_real_readout_audit_rejects_precision_and_gradient_drift(self):
        model=TinyPolicy().train();configure_parameter_scope(model,'head_only')
        features=(torch.ones(3),torch.ones(2,3))
        verify_frozen_readout(model,features)
        with self.assertRaises(ValueError):verify_frozen_readout(model,(features[0].requires_grad_(),features[1]))
        with self.assertRaises(ValueError):verify_frozen_readout(model,tuple(x.detach().bfloat16() for x in features))
        with torch.autocast('cpu',dtype=torch.bfloat16):
            with self.assertRaises(ValueError):verify_frozen_readout(model,tuple(x.detach() for x in features))
        model.lm.train()
        with self.assertRaises(ValueError):verify_frozen_readout(model,tuple(x.detach() for x in features))

    def test_compute_configuration_preserves_old_recipe_and_rejects_joint_fp32(self):
        self.assertEqual(training_compute_options({}),{'dtype':'bf16','checkpointing':1,
            'disable_frozen_encoder_input_grads':False})
        valid={'trainable_scope':'head_only','training_compute_dtype':'fp32',
               'disable_frozen_encoder_input_grads':True}
        self.assertEqual(training_compute_options(valid)['checkpointing'],0)
        for invalid in ({'training_compute_dtype':'fp32'},
                        dict(valid,trainable_scope='lora_and_head'),
                        dict(valid,training_compute_dtype='bf16'),
                        dict(valid,disable_frozen_encoder_input_grads=1)):
            with self.assertRaises(ValueError):training_compute_options(invalid)


if __name__=='__main__':unittest.main()
