import unittest

import torch
from jevany.model import PointerHead
from point_bias import folded_bias,value_features


class ContextualValueExportTests(unittest.TestCase):
    def test_bias_export_equals_additive_value_without_changing_backbone_head_maps(self):
        torch.manual_seed(31);head=PointerHead(4,3).eval().requires_grad_(False)
        before={name:value.detach().clone() for name,value in head.state_dict().items()}
        state,options=torch.randn(4),torch.randn(7,4);coefficient=torch.tensor([.4,-.2,.7])
        folded=folded_bias(head,coefficient,.5)
        expected=head(state,options)+.5*(value_features(head,state,options)@coefficient)
        self.assertTrue(torch.allclose(folded(state,options),expected,atol=1e-6))
        for name,value in before.items():
            self.assertTrue(torch.equal(value,head.state_dict()[name]))
            if name!='q.bias':self.assertTrue(torch.equal(value,folded.state_dict()[name]))

    def test_same_action_difference_is_zero_without_an_intercept(self):
        head=PointerHead(4,3).eval().requires_grad_(False);state,options=torch.randn(4),torch.randn(5,4)
        values=value_features(head,state,options);coef=torch.randn(3)
        self.assertEqual(float((values[2]-values[2])@coef),0.)
        unchanged=folded_bias(head,coef,0.)
        self.assertTrue(all(torch.equal(value,unchanged.state_dict()[name]) for name,value in head.state_dict().items()))

    def test_nonfinite_or_wrong_projection_cannot_be_exported(self):
        head=PointerHead(4,3).eval()
        for coefficient,strength in [(torch.ones(2),1.),(torch.full((3,),float('nan')),1.),(torch.ones(3),-1.)]:
            with self.assertRaises(ValueError):folded_bias(head,coefficient,strength)


if __name__=='__main__':unittest.main()
