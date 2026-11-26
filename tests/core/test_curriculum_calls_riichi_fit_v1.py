import copy
import json
import math
from pathlib import Path
import unittest
import torch
from tools import curriculum_calls_riichi_fit_v1 as fit
from tools.curriculum_calls_riichi_retention_v1 import annotate,weighted_loss,METHOD
from tools.curriculum_parent import ParentNotReady
from tests.core import test_curriculum_calls_fit as old_tests
from tests.core import test_curriculum_calls_repair as old_profile_tests

ROOT=Path(__file__).resolve().parents[2]


class CallsRiichiFitTests(unittest.TestCase):
    def fixture(self):return old_profile_tests.CallsRepairTests().fixture()
    def ce(self,q,keys,p):return -math.log(max(p[keys.index(q['label'])],1e-30))
    def recipe(self):
        _,data,counts=old_tests.CallsFitTests().recipe()
        config=json.loads((ROOT/'configs/training/curriculum-calls-riichi-retention-fit-v1.json').read_text())
        return config,data,counts

    def test_full_budget_preserved_and_implicit_objective_switch_rejected(self):
        config,data,counts=self.recipe();result=fit.require_recipe(config,data,counts)
        self.assertEqual(result['final_step'],384);self.assertEqual(result['record_exposures'],97420)
        for field,value in (('calls_loss_method','hard_ce'),('stage','stage2-calls-repair-v1'),
            ('candidate_rule','minimum_loss'),('epochs',1),('automatic_release',True),('accumulation',128)):
            altered=copy.deepcopy(config);altered[field]=value
            with self.assertRaises(ValueError):fit.require_recipe(altered,data,counts)

    def test_parent_kind_does_not_substitute_old_hard_CE(self):
        parent=dict(curriculum=dict(stage='stage1-riichi-class-control-v1'),
            training_config=dict(loss_method='max_discard_class_plus_conditional_v1'))
        fit.require_parent_kind(parent)
        for key,field,value in (('curriculum','stage','stage1-basic-repair-v1'),('training_config','loss_method','hard_ce')):
            altered=copy.deepcopy(parent);altered[key][field]=value
            with self.assertRaises(ParentNotReady):fit.require_parent_kind(altered)

    def test_common_metrics_and_source_quality_are_unchanged(self):
        rows,predictions=self.fixture();original=copy.deepcopy((rows,predictions))
        old=old_profile_tests.repair.profile(rows,predictions,fit.MASS,{'weighted_target_ce':123},self.ce)
        actual=fit.profile(rows,predictions,fit.MASS,{'weighted_target_ce':123},self.ce)
        for key,value in old.items():self.assertEqual(actual[key],value,key)
        self.assertEqual((rows,predictions),original);self.assertEqual(actual['loss_method'],METHOD)

    def test_cached_full_profile_matches_differentiable_loss_and_training_summary(self):
        rows,predictions=self.fixture();_,data,_=self.recipe();weights=data['example_weights'];pending=[]
        for row,p in zip(rows,predictions):
            raw=row['questions']['action'];keys=list(raw['criteria'])
            z=torch.linspace(-.7,.6,len(keys),dtype=torch.float64,requires_grad=True)
            p['p']=z.detach().softmax(0).tolist()
            encoded=dict(questions=[dict(qid='action',keys=keys,label=keys.index(raw['label']),src=raw['src'])])
            annotate(encoded,row);weighted_loss(z,encoded['questions'][0],'cpu',weights,pending)
        actual=fit.profile(rows,predictions,fit.MASS,{'weighted_target_ce':123},self.ce)
        summary=fit.summarize(pending,weights)
        for key,parts in actual['source_loss_components'].items():
            for name,value in parts.items():self.assertAlmostEqual(value,summary['source_loss_components'][key][name],places=12)
        expected_loss=sum(weights[r['source']]*r['objective'] for r in pending)/len(pending)
        self.assertAlmostEqual(summary['loss'],expected_loss,places=12)
        # This fixture offers only one discard beside reach, so max and
        # logsumexp coincide. Multiple-discard separation is checked below.
        self.assertAlmostEqual(summary['loss'],summary['ce'],places=12)

    def test_original_CE_separate_when_multiple_discards_compete_with_RI(self):
        rows,predictions=self.fixture()
        for row,p in zip(rows,predictions):
            if row['questions']['action']['src'].endswith('riichi_yes'):
                row['questions']['action']['criteria']['extra']={'type':'dahai'}
                p['keys'].append('extra');p['p']=[.25,.5,.25]
        result=fit.profile(rows,predictions,fit.MASS,{'weighted_target_ce':0},self.ce)
        key='basic:teacher:riichi_yes';parts=result['source_loss_components'][key]
        self.assertAlmostEqual(parts['objective'],math.log(1.5),places=12)
        self.assertAlmostEqual(parts['standard_action_ce'],math.log(2),places=12)
        self.assertNotEqual(result['weighted_class_objective'],result['weighted_target_ce'])

    def test_missing_or_nonfinite_update_observations_cannot_make_a_curve(self):
        rows,predictions=self.fixture();weights={key:1. for key in fit.MASS}
        with self.assertRaises(ValueError):fit.summarize([],weights)
        observations=[dict(source=key,**{name:1. for name in fit.COMPONENTS}) for key in fit.MASS]
        fit.summarize(observations,weights)
        observations[0]['objective']=float('nan')
        with self.assertRaises(ValueError):fit.summarize(observations,weights)
        with self.assertRaises(ValueError):fit.profile(rows,predictions[:-1],fit.MASS,{'weighted_target_ce':0},self.ce)


if __name__=='__main__':unittest.main()
