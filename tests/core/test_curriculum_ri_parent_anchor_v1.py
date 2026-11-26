import copy
import math
import unittest
import torch
from tools import curriculum_ri_parent_anchor_v1 as anchor
from tools import curriculum_progression_parent_loss_v1 as retained


def fixture(source, kinds, label, identifier, target=None):
    keys = ['k'+str(i) for i in range(len(kinds))]
    raw = dict(src=source, label=keys[label], criteria={k:dict(type=t) for k,t in zip(keys,kinds)})
    q = dict(qid='action', src=source, keys=keys, label=label, target=target)
    if target is not None: raw['target'] = dict(zip(keys,target))
    return dict(state={},_meta=dict(id=identifier),questions=dict(action=raw)), dict(questions=[q])


class ParentAnchorTests(unittest.TestCase):
    def population(self):
        rows=[]; enc={}; predictions=[]
        for i,source in enumerate(retained.MASS):
            kinds=['dahai','reach','dahai'] if source in anchor.RI_SOURCES else ['dahai']*3
            label=1 if source.endswith(':riichi_yes') else 2
            row,q=fixture(source,kinds,label,str(i));rows.append(row);enc[str(i)]=q
            if source in anchor.RI_SOURCES:
                predictions.append(dict(id=str(i),keys=['k0','k1','k2'],p=[.15,.6,.25]))
        return rows,enc,predictions

    def test_zero_anchor_and_zero_anchor_gradient_at_parent_distribution(self):
        rows,encoded,predictions=self.population();targets=anchor.make_targets(rows,predictions)
        for row in rows:
            if row['questions']['action']['src'] not in anchor.RI_SOURCES:continue
            before=copy.deepcopy(row)
            q=anchor.annotate(encoded[row['_meta']['id']],row,targets)['questions'][0]
            logits=torch.tensor([math.log(v) for v in (.15,.6,.25)],dtype=torch.float64,requires_grad=True)
            result=anchor.parts(logits,q,'cpu')
            self.assertAlmostEqual(float(result['parent_RI_KL'].detach()),0.,places=12)
            gradient=torch.autograd.grad(result['parent_RI_anchor_adjustment'],logits)[0]
            self.assertLess(float(gradient.abs().max()),1e-12)
            self.assertEqual(row,before)

    def test_anchor_gradient_pulls_probability_toward_parent_in_both_directions(self):
        rows,encoded,predictions=self.population();targets=anchor.make_targets(rows,predictions)
        row=next(r for r in rows if r['questions']['action']['src'] in anchor.RI_SOURCES)
        q=anchor.annotate(encoded[row['_meta']['id']],row,targets)['questions'][0]
        for current in (.2,.9):
            logits=torch.tensor([math.log((1-current)/2),math.log(current),math.log((1-current)/2)],
                dtype=torch.float64,requires_grad=True)
            result=anchor.parts(logits,q,'cpu');gradient=torch.autograd.grad(result['parent_RI_KL'],logits)[0]
            self.assertAlmostEqual(float(gradient[1]),current-.6,places=12)
            self.assertGreater(float(result['parent_RI_KL'].detach()),0.)

    def test_non_RI_sources_keep_exact_retained_values_and_gradients(self):
        for i,source in enumerate(set(retained.MASS)-anchor.RI_SOURCES):
            target=[.3,0.,.7] if ':mc:' in source else None
            row,encoded=fixture(source,['dahai']*3,2,str(i),target)
            old=retained.annotate(copy.deepcopy(encoded),row)['questions'][0]
            new=anchor.annotate(encoded,row,{})['questions'][0]
            a=torch.tensor([-.3,.4,.8],dtype=torch.float64,requires_grad=True)
            b=a.detach().clone().requires_grad_(True)
            expected=retained.parts(a,old,'cpu')['objective'];actual=anchor.parts(b,new,'cpu')['objective']
            self.assertTrue(torch.equal(expected,actual));expected.backward();actual.backward()
            self.assertTrue(torch.equal(a.grad,b.grad))

    def test_candidate_permutation_preserves_anchor_value_and_gradient(self):
        rows,encoded,predictions=self.population();targets=anchor.make_targets(rows,predictions)
        row=next(r for r in rows if r['questions']['action']['src'] in anchor.RI_SOURCES)
        raw=encoded[row['_meta']['id']];q=anchor.annotate(copy.deepcopy(raw),row,targets)['questions'][0]
        permutation=[2,0,1];permuted=copy.deepcopy(raw);pq=permuted['questions'][0]
        pq['keys']=[pq['keys'][i] for i in permutation];pq['label']=permutation.index(pq['label'])
        qp=anchor.annotate(permuted,row,targets)['questions'][0]
        x=torch.tensor([-.4,.8,.2],dtype=torch.float64,requires_grad=True)
        y=x.detach()[permutation].clone().requires_grad_(True)
        a=anchor.parts(x,q,'cpu')['objective'];b=anchor.parts(y,qp,'cpu')['objective']
        self.assertTrue(torch.allclose(a,b,atol=1e-12,rtol=0));a.backward();b.backward()
        self.assertTrue(torch.allclose(x.grad[permutation],y.grad,atol=1e-12,rtol=0))

    def test_missing_rows_menus_targets_and_changed_inputs_rejected(self):
        rows,encoded,predictions=self.population()
        with self.assertRaises(ValueError):anchor.make_targets(rows,predictions[:-1])
        broken=copy.deepcopy(predictions);broken[0]['keys'][0]='unknown'
        with self.assertRaises(ValueError):anchor.make_targets(rows,broken)
        targets=anchor.make_targets(rows,predictions)
        row=next(r for r in rows if r['questions']['action']['src'] in anchor.RI_SOURCES)
        with self.assertRaises(ValueError):anchor.annotate(copy.deepcopy(encoded[row['_meta']['id']]),row,{})
        changed=copy.deepcopy(row);changed['state']['unexpected_context']=1
        with self.assertRaises(ValueError):anchor.annotate(copy.deepcopy(encoded[row['_meta']['id']]),changed,targets)


if __name__=='__main__':unittest.main()
