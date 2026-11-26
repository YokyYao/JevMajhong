import copy
import math
import unittest
import torch
from tools import curriculum_placement_objective_v1 as objective
from tools import curriculum_yaku_objective_v1 as replay


class PlacementSoftObjectiveTests(unittest.TestCase):
    def question(self,target,kinds):
        return dict(src='curriculum4:rank_nonwin:progression_draw:third_gap3000',
            target=target,label=max(range(len(target)),key=target.__getitem__),
            **{objective.MARKER:objective.METHOD,objective.FIELD:kinds})

    def test_full_soft_RI_matches_analytic_group_and_conditional_decomposition(self):
        z=torch.tensor([.2,-.3,.8],dtype=torch.float64,requires_grad=True)
        q=self.question([.6,.1,.3],['reach','dahai','dahai'])
        result=objective.parts(z,q,'cpu')
        standard=-(torch.tensor(q['target'],dtype=z.dtype)*z.log_softmax(0)).sum()
        self.assertAlmostEqual(float((result['standard_target_ce']-standard).detach()),0.,places=12)
        self.assertAlmostEqual(float((standard-result['RI_group_ce']-result['conditional_discard_ce']).detach()),0.,places=12)
        self.assertAlmostEqual(float((result['objective']-standard-result['RI_group_ce']).detach()),0.,places=12)
        one=objective.parts(z,q,'cpu',class_weight=1.)['objective']
        self.assertGreater(float((torch.autograd.grad(result['objective'],z,retain_graph=True)[0]
            -torch.autograd.grad(one,z)[0]).abs().max()),.01)

    def test_soft_target_is_stationary_under_full_RI_rule_and_candidate_permutation(self):
        target=[.4,.1,.5]
        z=torch.tensor([math.log(p) for p in target],dtype=torch.float64,requires_grad=True)
        q=self.question(target,['reach','dahai','dahai'])
        result=objective.parts(z,q,'cpu');gradient=torch.autograd.grad(result['objective'],z)[0]
        self.assertLess(float(gradient.abs().max()),1e-12)
        perm=[2,0,1];p=self.question([target[i] for i in perm],[q[objective.FIELD][i] for i in perm])
        other=objective.parts(z[perm],p,'cpu')
        self.assertAlmostEqual(float((result['objective']-other['objective']).detach()),0.,places=12)

    def test_untested_legal_option_keeps_exact_zero_gradient(self):
        z=torch.tensor([0.,1.,90.],dtype=torch.float64,requires_grad=True)
        q=self.question([.4,.6,0.],['none','pon','chi'])
        result=objective.parts(z,q,'cpu');result['objective'].backward()
        self.assertEqual(float(z.grad[2]),0.)
        self.assertEqual(float(result['RI_group_ce'].detach()),0.)

    def test_partial_RI_is_rejected_instead_of_ignoring_unlabelled_discards(self):
        q=self.question([.4,.6,0.],['reach','dahai','dahai'])
        with self.assertRaises(ValueError):objective.parts(torch.zeros(3),q,'cpu')

    def test_existing_replay_objective_and_gradient_remain_exact(self):
        from tools import curriculum_progression_objective_v1 as old
        q=dict(src='curriculum2:basic:teacher:riichi_no',label=1,
            **{old.MARKER:old.METHOD,old.FIELD:[0],replay.MARKER:replay.METHOD})
        z=torch.tensor([.2,-.3,.8],dtype=torch.float64,requires_grad=True)
        a=objective.parts(z,q,'cpu');b=replay.parts(z,q,'cpu')
        for k in objective.COMPONENTS:self.assertTrue(torch.equal(a[k],b[k]))
        self.assertTrue(torch.equal(torch.autograd.grad(a['objective'],z,retain_graph=True)[0],
            torch.autograd.grad(b['objective'],z)[0]))

    def test_annotation_aligns_materializer_order_without_altering_visible_supervision(self):
        raw=dict(src='curriculum4:rank_nonwin:progression_draw:third_gap3000',
            label='ri',target={'d':.2,'ri':.8},criteria={'d':dict(type='dahai'),'ri':dict(type='reach')})
        encoded=dict(questions=[dict(qid='action',src=raw['src'],keys=['ri','d'],label=0,target=[.8,.2])])
        original=copy.deepcopy(encoded);request=dict(questions=dict(action=copy.deepcopy(raw)))
        objective.annotate(encoded,request)
        restored=copy.deepcopy(encoded)
        for q in restored['questions']:
            q.pop(objective.MARKER);q.pop(objective.FIELD)
        self.assertEqual(restored,original);self.assertEqual(request['questions']['action'],raw)
        with self.assertRaises(ValueError):objective.annotate(encoded,request)

