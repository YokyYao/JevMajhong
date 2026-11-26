import copy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.build_curriculum_calls_repair import MASS,layout
from tools.curriculum_calls_fit import require_recipe,retention,balanced,parent_loader,verify_parent
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit
from tools.curriculum_parent import ParentNotReady

ROOT=Path(__file__).resolve().parents[2]


class CallsFitTests(unittest.TestCase):
    def recipe(self):
        c=json.loads((ROOT/'configs/training/curriculum-calls-repair-v1.json').read_text())
        counts={'basic:legacy:plain':8826,'basic:legacy:riichi_yes':194,'basic:legacy:riichi_no':272,
            'basic:teacher:plain':3106,'basic:teacher:riichi_yes':1075,'basic:teacher:riichi_no':812,
            'advanced:legacy:post_meld':5590,'advanced:legacy:call':3569,'advanced:legacy:kan':1786,
            'advanced:teacher:post_meld':9352,'advanced:teacher:call':13399,'advanced:teacher:kan':729}
        proposed=layout(counts)
        data=dict(status='calls_repair12_source_data_prepared_not_parent_or_fit_admitted',
            source_specific_basic_parent_seal_required=True,required_input_schema=None,
            loss_mass=copy.deepcopy(MASS),categories={'train':counts},
            count_only_accumulation=proposed['accumulation'],example_weights=proposed['example_weights'])
        return c,data,counts

    def test_actual_registered_recipe_consumes_every_record_twice(self):
        c,data,counts=self.recipe();result=require_recipe(c,data,counts)
        self.assertEqual(result['accumulation'],254);self.assertEqual(result['final_step'],384)
        self.assertEqual(result['record_exposures'],97420)
        self.assertGreater(result['example_weights']['basic:legacy:riichi_yes'],8)
        self.assertLessEqual(max(result['example_weights'].values()),16)

    def test_changed_budget_source_mass_optimizer_or_cadence_is_rejected(self):
        for field,value in (('epochs',1),('accumulation',253),('learning_rate',2e-5),('head_learning_rate',1e-4),
            ('optimizer_resume',True),('validation_every_steps',96),('game_evaluation_every_steps',96),
            ('candidate_rule','minimum_ce'),('automatic_stage3_launch',True),('automatic_release',True)):
            c,data,counts=self.recipe();c[field]=value
            with self.assertRaises(ValueError,msg=field):require_recipe(c,data,counts)
        for field in ('loss_mass','example_weights','source_specific_basic_parent_seal_required'):
            c,data,counts=self.recipe()
            if field=='source_specific_basic_parent_seal_required':data[field]=False
            else:data[field][next(iter(data[field]))]*=.9
            with self.assertRaises(ValueError,msg=field):require_recipe(c,data,counts)

    def points(self):
        basic=dict(stratum_accuracy={'plain':.8},stratum_ce={'plain':.5},
            source_stratum_accuracy={'legacy:plain':.8,'teacher:plain':.8},
            source_stratum_ce={'legacy:plain':.5,'teacher:plain':.5},
            riichi_type_counts={'tp':160,'fn':40,'tn':160,'fp':40},
            riichi_by_source={s:{'tp':80,'fn':20,'tn':80,'fp':20} for s in ('legacy','teacher')})
        return {'basic_replay':basic}

    def test_retention_does_not_hide_one_source_regression_in_pooled_metric(self):
        first=self.points();last=copy.deepcopy(first);limits=self.recipe()[0]['development_targets']
        self.assertTrue(retention(first,last,limits)['passed'])
        last['basic_replay']['riichi_by_source']['legacy'].update(tp=60,fn=40)
        self.assertTrue(retention(first,last,limits)['basic_source_quality']['all_source_quality_passed'])
        self.assertFalse(retention(first,last,limits)['passed'])
        last=copy.deepcopy(first);last['basic_replay']['source_stratum_accuracy']['teacher:plain']=.7
        self.assertFalse(retention(first,last,limits)['passed'])
        with self.assertRaises(ValueError):balanced({'tn':10,'fp':5})

    def test_terminal_evidence_needs_matching_real_gpu_success_and_entry(self):
        receipt={'job_id':'123','exit_code':0}
        raw='JobId=123 JobState=COMPLETED ExitCode=0:0 NodeList=ws-test AllocTRES=cpu=4,mem=32G,gres/gpu=1 StartTime=2026-11-22T12:00:00 EndTime=2026-11-22T12:01:00 Command=/tmp/task.slurm'
        self.assertEqual(scheduler_exit(receipt,raw,Path('/tmp/task.slurm'))['job_id'],'123')
        for old,new in (('JobId=123','JobId=124'),('COMPLETED','RUNNING'),('0:0','1:0'),('ws-test','lo-01'),
            ('gres/gpu=1','gres/gpu=0'),('gres/gpu=1','billing=4'),('Command=/tmp/task.slurm','Command=/tmp/other.slurm'),
            ('EndTime=2026-11-22T12:01:00','EndTime=Unknown')):
            with self.assertRaises(ParentNotReady):scheduler_exit(receipt,raw.replace(old,new),Path('/tmp/task.slurm'))

    def test_weighted_real_logit_gradient_and_observation_semantics(self):
        import torch
        from tools.train_curriculum_calls_repair import weighted_loss
        _,data,_=self.recipe();weights=data['example_weights']
        for key,weight in weights.items():
            logits=torch.tensor([.2,-.3,.4],dtype=torch.float64,requires_grad=True);observed=[]
            q={'src':'curriculum2:'+key,'label':1}
            loss=weighted_loss(logits,q,'cpu',weights,observed);loss.backward()
            p=logits.detach().softmax(0);expected=p.clone();expected[1]-=1
            self.assertTrue(torch.allclose(logits.grad,expected*weight,atol=1e-12))
            self.assertAlmostEqual(float(loss),-math.log(float(p[1]))*weight)
            self.assertEqual(observed[0][0],key)
            with torch.no_grad():
                ignored=[];weighted_loss(logits,q,'cpu',weights,ignored)
            self.assertEqual(ignored,[])
        with self.assertRaises(ValueError):weighted_loss(logits,{'src':'other','label':1},'cpu',weights)
        with self.assertRaises(ValueError):weighted_loss(logits,dict(q,target={'x':1}),'cpu',weights)

    def test_v2_loader_does_not_accept_old_or_research_seals(self):
        cfg={'parent_initialization':{'manifest':'/tmp/seal','sha256':'h','method':'curriculum_registered_final_with_quality_v1'}}
        for source in ('tools/seal_curriculum_basic_repair.py','unrelated'):
            seal={'method':cfg['parent_initialization']['method'],'source_specific_parent_quality_checked':True,'source_specific_sealer':source}
            with patch('tools.curriculum_calls_fit.sha',return_value='h'),patch('tools.curriculum_calls_fit.read',return_value=seal), \
                patch('tools.curriculum_calls_fit.verify_curriculum_parent') as old:
                with self.assertRaises(ParentNotReady):verify_parent(cfg)
                old.assert_not_called()
        class Init:verified_parent=object()
        original=Init.verified_parent
        with self.assertRaises(RuntimeError):
            with parent_loader(Init):
                self.assertIs(Init.verified_parent,verify_parent);raise RuntimeError('fixture')
        self.assertIs(Init.verified_parent,original)


if __name__=='__main__':unittest.main()
