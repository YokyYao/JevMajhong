import json
from pathlib import Path
import unittest
import torch

from tools.audit_endgame_head_gradients import (validate_config,cosine,gradient_directions,
                                              source_gradients,audit_order)


class TinyPointer(torch.nn.Module):
    def __init__(self):
        super().__init__();self.q=torch.nn.Linear(2,2,bias=False);self.k=torch.nn.Linear(2,2,bias=False)
        with torch.no_grad():self.q.weight.copy_(torch.eye(2));self.k.weight.copy_(torch.eye(2))

    def forward(self,decide,options):return self.k(options)@self.q(decide)


class HeadGradientAuditTests(unittest.TestCase):
    def test_registration_forbids_optimization_and_validation(self):
        path=Path(__file__).resolve().parents[2]/'configs/training/endgame-head-gradient-diagnostic-v1.json'
        config=json.loads(path.read_text());validate_config(config)
        for change in ({'optimizer_updates':1},{'validation_data_used':True},
                       {'candidate_training_registered':True},{'new_encoder_queries':826}):
            with self.assertRaises(ValueError):validate_config(dict(config,**change))

    def test_opposing_source_gradients_and_weighted_objective_agree(self):
        head=TinyPointer();snapshot={k:v.clone() for k,v in head.state_dict().items()}
        feature={'decide':torch.tensor([1.,0.]),'options':torch.tensor([[0.,1.],[0.,-1.]]),'keys':['x','y']}
        features={'a':feature,'b':feature}
        rows=[{'_meta':{'id':key},'questions':{'action':{'src':key,'target':target}}} for key,target in
              (('a',{'x':1.,'y':0.}),('b',{'x':0.,'y':1.}))]
        gradients,losses=source_gradients(head,rows,features)
        self.assertAlmostEqual(cosine(gradients['a'],gradients['b']),-1.)
        parameters=torch.cat([p.detach().reshape(-1) for p in head.parameters()])
        summary,total=gradient_directions(gradients,{'a':.25,'b':.75},parameters)
        self.assertGreater(summary['sources']['a']['sgd_first_order_loss_derivative_per_unit_lr'],0)
        self.assertLess(summary['sources']['b']['sgd_first_order_loss_derivative_per_unit_lr'],0)
        logits=head(feature['decide'],feature['options']);loss=-.25*logits.log_softmax(-1)[0]-.75*logits.log_softmax(-1)[1]
        actual=torch.cat([v.reshape(-1) for v in torch.autograd.grad(loss,tuple(head.parameters()))])
        self.assertTrue(torch.allclose(total,actual.double(),atol=1e-8))
        self.assertTrue(all(p.grad is None for p in head.parameters()))
        self.assertTrue(all(torch.equal(v,snapshot[k]) for k,v in head.state_dict().items()))

    def test_zero_gradient_is_undefined_not_an_invented_conflict(self):
        self.assertIsNone(cosine(torch.zeros(3),torch.ones(3)))
        with self.assertRaises(ValueError):gradient_directions({'a':torch.ones(3)},{'a':.8},torch.ones(3))

    def test_saved_order_must_reproduce_exposure_and_distinguish_weights(self):
        rows=[{'_meta':{'id':str(i)},'questions':{'action':{'src':'strict' if i>=128 else 'replay'}}} for i in range(160)]
        order=list(range(160));seen=[str(i) for i in order[:128]]
        blocks,_=audit_order(rows,order,seen,{'strict':7.,'replay':.5},steps=8)
        self.assertEqual(blocks[0]['counts'],{'replay':128})
        self.assertEqual(blocks[0]['weighted_loss_coefficient_sum_by_source'],{'replay':.5})
        with self.assertRaises(ValueError):audit_order(rows,list(reversed(order)),seen,{'strict':7.,'replay':.5},steps=8)


if __name__=='__main__':unittest.main()
