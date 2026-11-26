import copy
import unittest
import torch
from tools import curriculum_ri_parent_anchor_v1 as anchor
from tools import curriculum_ri_parent_anchor_weighted_v1 as weighted
from tools import curriculum_ri_inverse_repeat_loss_v1 as inverse
from tools import curriculum_progression_parent_loss_v1 as retained
from tests.core import test_curriculum_ri_parent_anchor_v1 as fixtures


class WeightedAnchorTests(unittest.TestCase):
    def annotated(self):
        rows,encoded,predictions=fixtures.ParentAnchorTests().population();targets=anchor.make_targets(rows,predictions)
        result=[]
        for row in rows:
            source=row['questions']['action']['src']
            if ':mc:' in source:
                row,rec=fixtures.fixture(source,['dahai']*3,2,row['_meta']['id'],[.3,0.,.7])
            else:rec=encoded[row['_meta']['id']]
            work=anchor.annotate(rec,row,targets);weighted.attach(work,1.25,row['_meta']['id'])
            result.append(work['questions'][0])
        return result

    def test_weighted_value_and_gradient_match_independent_class_KL_reference(self):
        q=next(q for q in self.annotated() if q['src'] in anchor.RI_SOURCES)
        a=torch.tensor([-.3,.8,.2],dtype=torch.float64,requires_grad=True)
        b=a.detach().clone().requires_grad_(True)
        actual=weighted.weighted_loss(a,q,'cpu')
        reach=q[anchor.FIELD]['reach'];probability=q[anchor.FIELD]['probability']
        probability2=torch.stack((b[[i for i in range(len(b)) if i!=reach]].logsumexp(0),b[reach])).log_softmax(0)
        target=torch.tensor([1-probability,probability],dtype=torch.float64)
        kl=(target*(target.log()-probability2)).sum()
        expected=(retained.parts(b,q,'cpu')['objective']+4*kl)*1.25
        self.assertTrue(torch.allclose(actual,expected,atol=1e-12,rtol=0))
        actual.backward();expected.backward();self.assertTrue(torch.allclose(a.grad,b.grad,atol=1e-12,rtol=0))

    def test_all_other_sources_keep_exact_old_weighted_objective_and_gradient(self):
        for q in self.annotated():
            if q['src'] in anchor.RI_SOURCES:continue
            a=torch.tensor([-.3,.8,.2],dtype=torch.float64,requires_grad=True)
            b=a.detach().clone().requires_grad_(True)
            actual=weighted.weighted_loss(a,q,'cpu');expected=inverse.weighted_loss(b,q,'cpu')
            self.assertTrue(torch.equal(actual,expected));actual.backward();expected.backward()
            self.assertTrue(torch.equal(a.grad,b.grad))

    def test_actual_observer_reconstructs_total_anchor_and_retained_objective(self):
        pending=[];losses=[]
        for q in self.annotated():
            logits=torch.tensor([-.3,.8,.2],dtype=torch.float64,requires_grad=True)
            losses.append(weighted.weighted_loss(logits,q,'cpu',pending))
        summary=weighted.summarize(pending)
        self.assertAlmostEqual(summary['loss'],float(torch.stack(losses).mean().detach()),places=12)
        self.assertAlmostEqual(summary['loss'],summary['retained_objective_without_anchor']+summary['parent_RI_anchor_adjustment'],places=12)
        self.assertAlmostEqual(summary['parent_RI_anchor_adjustment'],4*summary['parent_RI_KL'],places=12)
        self.assertEqual(set(summary['source_counts']),set(retained.MASS))
        self.assertEqual(summary['loss_method'],weighted.METHOD)

    def test_invalid_weight_missing_anchor_and_corrupted_observations_rejected(self):
        q=self.annotated()[0];z=torch.tensor([-.3,.8,.2],requires_grad=True)
        bad=copy.deepcopy(q);bad[inverse.FIELD]=0.
        with self.assertRaises(ValueError):weighted.weighted_loss(z,bad,'cpu')
        bad=copy.deepcopy(q);bad.pop(anchor.MARKER)
        with self.assertRaises(ValueError):weighted.weighted_loss(z,bad,'cpu')
        pending=[];weighted.weighted_loss(z,q,'cpu',pending);pending[0]['objective']+=1.
        with self.assertRaises(ValueError):weighted.summarize(pending)


if __name__=='__main__':unittest.main()
