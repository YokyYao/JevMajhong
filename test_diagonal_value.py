import unittest

import torch
from jevany.model import PointerHead
from diagonal_value import cross_validate,folded_head,pointer_products,ridge_fit,wall_folds


class DiagonalValueTests(unittest.TestCase):
    def test_fold_matches_additive_point_value_and_preserves_source(self):
        torch.manual_seed(17);head=PointerHead(4,3).eval();state={k:v.detach().clone() for k,v in head.state_dict().items()}
        decide,options=torch.randn(4),torch.randn(5,4);coef=torch.tensor([.2,-.3,1.])
        folded=folded_head(head,coef,.5)
        expected=head(decide,options)+.5*(pointer_products(head,decide,options)@coef)
        self.assertTrue(torch.allclose(folded(decide,options),expected,atol=1e-6))
        self.assertTrue(all(torch.equal(v,head.state_dict()[k]) for k,v in state.items()))
        self.assertTrue(torch.equal(folded.k.weight,head.k.weight))
        identity=folded_head(head,coef,0.)
        self.assertTrue(all(torch.equal(v,identity.state_dict()[k]) for k,v in state.items()))

    def test_regression_has_no_same_action_intercept(self):
        x=torch.tensor([[1.,0.],[0.,1.],[1.,1.],[-1.,2.]],dtype=torch.float64)
        truth=torch.tensor([2.,-3.],dtype=torch.float64);y=x@truth
        fitted=ridge_fit(x,y,1e-8)
        self.assertTrue(torch.allclose(fitted,truth,atol=1e-6))
        self.assertEqual(float(torch.zeros(2,dtype=torch.float64)@fitted),0.)

    def test_all_replicas_of_each_wall_are_held_out_together(self):
        seeds=[s for s in range(10) for _ in range(4)]
        folds=wall_folds(seeds,5)
        self.assertEqual(sorted(i for fold in folds for i in fold),list(range(40)))
        for fold in folds:
            held={seeds[i] for i in fold}
            self.assertFalse(held&{s for i,s in enumerate(seeds) if i not in fold})
        torch.manual_seed(1);x=torch.randn(40,2,dtype=torch.float64);y=x@torch.tensor([2.,-1.],dtype=torch.float64)
        report=cross_validate(x,y,seeds,[.001,1.])
        self.assertTrue(report['improves_over_zero'])
        self.assertEqual(report['best_alpha'],.001)


if __name__=='__main__':unittest.main()
