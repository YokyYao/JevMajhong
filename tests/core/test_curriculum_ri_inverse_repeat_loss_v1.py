import copy
from collections import Counter
import unittest
import torch
from tools import curriculum_ri_inverse_repeat_loss_v1 as weighted
from tools import curriculum_progression_parent_loss_v1 as retained
from tools.curriculum_ri_every_update_replay_v1 import build,RI,MASS
from tests.core.test_curriculum_progression_parent_loss_v1 import fixture


def example(source):
    if source.endswith(':riichi_yes'):kinds,label=['dahai','reach','dahai'],1
    elif source.endswith(':riichi_no'):kinds,label=['dahai','reach','dahai'],2
    elif source.endswith((':call',':kan')):kinds,label=['none','pon','daiminkan'],0
    else:kinds,label=['dahai','dahai','dahai'],1
    return fixture(source,kinds,label,[.3,0.,.7] if ':mc:' in source else None)


class InverseRepeatLossTests(unittest.TestCase):
    def test_unequal_repeat_whole_empirical_objective_and_gradient(self):
        categories=[];records=[]
        for source in MASS:
            for _ in range(RI.index(source)+1 if source in RI else 2):
                raw,rec=example(source);records.append(retained.annotate(rec,raw));categories.append(source)
        counts=Counter(categories);plan=build(categories,list(range(len(categories))),17,batch=9)
        za=torch.linspace(-1,1,len(categories)*3,dtype=torch.float64,requires_grad=True)
        zb=za.detach().clone().requires_grad_(True)
        expected=sum(retained.parts(za[i*3:i*3+3],r['questions'][0],'cpu')['objective']*MASS[categories[i]]/counts[categories[i]] for i,r in enumerate(records))
        observed=[]
        for i in plan['indices']:
            q=weighted.attach(copy.deepcopy(records[i]),plan['weights_by_original_index'][str(i)],str(i))['questions'][0]
            observed.append(weighted.weighted_loss(zb[i*3:i*3+3],q,'cpu'))
        actual=torch.stack(observed).mean()
        self.assertAlmostEqual(float(expected.detach()),float(actual.detach()),places=12)
        a,=torch.autograd.grad(expected,za);b,=torch.autograd.grad(actual,zb)
        self.assertTrue(torch.allclose(a,b,atol=1e-14,rtol=1e-13))

    def test_journal_reports_actual_per_example_weighting(self):
        pending=[];losses=[];ces=[]
        for source,weight in ((RI[0],.3),(RI[1],2.7),('curriculum3:mc:closed_discard',1.2)):
            raw,rec=example(source);retained.annotate(rec,raw)
            q=weighted.attach(rec,weight,source)['questions'][0]
            z=torch.tensor([.2,-.4,.7],dtype=torch.float64,requires_grad=True)
            ces.append(float(retained.parts(z,q,'cpu')['standard_target_ce'].detach())*weight)
            losses.append(weighted.weighted_loss(z,q,'cpu',pending))
        summary=weighted.summarize(pending)
        self.assertAlmostEqual(summary['loss'],float(torch.stack(losses).mean().detach()),places=12)
        self.assertAlmostEqual(summary['ce'],sum(ces)/3,places=12)
        self.assertAlmostEqual(sum(summary['effective_source_mass'].values()),4.2/3,places=12)
        self.assertAlmostEqual(summary['loss']-summary['ce'],summary['signed_objective_adjustment'],places=12)

    def test_annotations_preserve_request_and_unweighted_objective(self):
        raw,rec=example(RI[0]);original=copy.deepcopy(raw);retained.annotate(rec,raw)
        z=torch.tensor([.2,-.4,.7],dtype=torch.float64)
        before=retained.parts(z,rec['questions'][0],'cpu')['objective']
        weighted.attach(rec,1.3,'original-id')
        self.assertTrue(torch.equal(before,retained.parts(z,rec['questions'][0],'cpu')['objective']))
        self.assertEqual(raw,original)
        with self.assertRaises(ValueError):weighted.attach(rec,1.3,'original-id')

    def test_invalid_or_missing_weights_and_corrupt_observations_rejected(self):
        raw,rec=example(RI[0]);retained.annotate(rec,raw)
        for bad in (True,0,-1,17,float('nan'),float('inf')):
            with self.assertRaises(ValueError):weighted.attach(copy.deepcopy(rec),bad,'id')
        with self.assertRaises(ValueError):weighted.weighted_loss(torch.zeros(3),rec['questions'][0],'cpu')
        with self.assertRaises(ValueError):weighted.summarize([])


if __name__=='__main__':unittest.main()
