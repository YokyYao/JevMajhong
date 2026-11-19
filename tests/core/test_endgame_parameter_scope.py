import unittest
import torch
from jevmahjong.endgame.fit import configure_parameter_scope,validate_parameter_updates


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base=torch.nn.Linear(3,3)
        self.lora_A=torch.nn.Linear(3,3,bias=False)
        self.dropout=torch.nn.Dropout(.5)
        self.base.requires_grad_(False)

    def forward(self,x):return self.base(x)+self.dropout(self.lora_A(x))


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


if __name__=='__main__':unittest.main()
