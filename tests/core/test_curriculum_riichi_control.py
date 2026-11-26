import copy
import json
import math
from pathlib import Path
import unittest

import torch
from tools.curriculum_basic_repair import MASS,grade
from tools.curriculum_riichi_control import (cached_parts,profile,require_recipe,require_initial_predictions,require_launch_failure)
from tools.curriculum_riichi_class_loss import FIELD,parts
from tools.curriculum_parent import ParentNotReady

ROOT=Path(__file__).resolve().parents[2]


class RiichiControlTests(unittest.TestCase):
    def recipe(self):
        c=json.loads((ROOT/'configs/training/curriculum-riichi-class-control-v1.json').read_text())
        old=json.loads((ROOT/'configs/training/curriculum-basic-repair-v1.json').read_text())
        return c,dict(curriculum=old,registered_final_step=608,train_records=14285)

    def test_one_factor_recipe_rejects_data_lr_seed_budget_and_decoder_changes(self):
        c,control=self.recipe();require_recipe(c,control)
        for key,value in (('seed',1),('learning_rate',1e-5),('accumulation',95),('epochs',5),
            ('data_root','other-data'),('seven_pairs_bonus',1),('new_decoder','class05')):
            changed=copy.deepcopy(c);changed[key]=value
            with self.assertRaises(ValueError,msg=key):require_recipe(changed,control)

    def fixture(self):
        records=[];predictions=[]
        for source in MASS:
            category=source.split(':')[1];yes=category=='riichi_yes'
            criteria={'a':{'type':'dahai'},'b':{'type':'dahai'}}
            if category!='plain':criteria['r']={'type':'reach'}
            label='r' if yes else 'b';q={'src':'curriculum1:'+source,'criteria':criteria,'label':label}
            records.append({'_meta':{'id':source},'questions':{'action':q}})
            logits=torch.linspace(-.6,.5,len(criteria),dtype=torch.float64)
            predictions.append({'id':source,'keys':list(criteria),'p':logits.softmax(0).tolist()})
        return records,predictions

    def ce(self,q,keys,p):return -math.log(p[keys.index(q['label'])])

    def test_saved_score_objective_matches_actual_tensor_loss_and_preserves_common_ce(self):
        records,predictions=self.fixture()
        result=profile(records,predictions,MASS,{'weighted_target_ce':1.},self.ce)
        expected=0.;common=0.
        for row,pred in zip(records,predictions):
            q=row['questions']['action'];keys=pred['keys'];source=q['src'].removeprefix('curriculum1:')
            encoded=dict(q,label=keys.index(q['label']),**{FIELD:[i for i,k in enumerate(keys) if q['criteria'][k]['type']=='reach']})
            actual=parts(torch.tensor(pred['p'],dtype=torch.float64).log(),encoded,'cpu')
            cached=cached_parts(q,keys,pred['p'],self.ce)
            for key,value in cached.items():self.assertAlmostEqual(value,float(actual[key]))
            expected+=MASS[source]*float(actual['objective']);common+=MASS[source]*float(actual['standard_action_ce'])
        self.assertAlmostEqual(result['weighted_class_objective'],expected)
        self.assertAlmostEqual(result['weighted_target_ce'],common)
        self.assertNotAlmostEqual(expected,common)
        self.assertEqual(set(result['source_loss_components']),set(MASS))

    def test_all_initial_predictions_required_with_original_key_order(self):
        rows=[{'id':str(i),'keys':['a','b'],'p':[.4,.6]} for i in range(3266)]
        self.assertEqual(require_initial_predictions(rows,copy.deepcopy(rows))['maximum_probability_error'],0)
        for change in ('missing','duplicate','probability','order'):
            bad=copy.deepcopy(rows)
            if change=='missing':bad.pop()
            elif change=='duplicate':bad[-1]=copy.deepcopy(bad[0])
            elif change=='probability':bad[0]['p']=[.5,.5]
            else:bad[0]['keys'].reverse();bad[0]['p'].reverse()
            with self.assertRaises(ValueError,msg=change):require_initial_predictions(rows,bad)

    def final(self):
        _,control=self.recipe()
        counts={key:10 for key in MASS}
        validation=dict(local_step=608,source_stratum_counts=counts,stratum_counts={'plain':20,'riichi_yes':20,'riichi_no':20},
            riichi_by_source={s:{'tp':8,'fn':2,'tn':8,'fp':2} for s in ('legacy','teacher')},
            riichi_type_counts={'tp':16,'fn':4,'tn':16,'fp':4},stratum_accuracy={'plain':.8},stratum_ce={'plain':.5},
            source_stratum_accuracy={'legacy:plain':.8,'teacher:plain':.8},source_stratum_ce={'legacy:plain':.5,'teacher:plain':.5})
        done=dict(status='stage1_fit_and_curves_complete',optimizer_steps=608,actual_records_seen=57140)
        evidence=dict(status='stage1_final_evidence_audit_passed',optimizer_steps=608,real_records_seen=57140,
            source_and_data_hashes_unchanged=True,heldout_groups_excluded_from_all_stages=True,
            final_summary={'matches':64,'seed_groups':8},saved_reloads=[{'step':608,'identity':'actual-test-identity'}],
            source_specific_parent_seal_required=True)
        self.grade(evidence,validation)
        return control,evidence,done,validation

    def grade(self,evidence,validation):
        quality=grade(validation);evidence.update(basic_repair_source_quality=quality,stage1_advancement=quality['original_aggregate_gate'])

    def failed_final(self):
        control,evidence,done,validation=self.final()
        validation['riichi_by_source']['legacy'].update(tn=4,fp=6)
        validation['riichi_type_counts'].update(tn=12,fp=8);self.grade(evidence,validation)
        return control,evidence,done,validation

    def test_actual_completed_source_failure_triggers_fresh_experiment_not_parent_admission(self):
        control,evidence,done,validation=self.failed_final()
        self.assertTrue(evidence['stage1_advancement']['passed'])
        actual=require_launch_failure(control,evidence,done,validation)
        self.assertFalse(actual['failed_parent_admitted']);self.assertEqual(actual['student_initialization'],'fresh_fixed_base')
        with self.assertRaises(ParentNotReady):require_launch_failure(*self.final())

    def test_partial_forged_or_mismatched_failure_cannot_trigger(self):
        for change in ('midpoint','games','audit','reload','confusion','forged_grade'):
            control,evidence,done,validation=self.failed_final()
            if change=='midpoint':done['optimizer_steps']=304
            elif change=='games':evidence['final_summary']['matches']=16
            elif change=='audit':evidence['status']='running'
            elif change=='reload':evidence['saved_reloads']=[]
            elif change=='confusion':validation['source_stratum_counts']['legacy:riichi_no']=11
            else:evidence['basic_repair_source_quality']['all_source_quality_passed']=True
            with self.assertRaises((ParentNotReady,ValueError),msg=change):require_launch_failure(control,evidence,done,validation)


if __name__=='__main__':unittest.main()
