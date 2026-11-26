import copy
import unittest

from tools.curriculum_placement_assessment_v1 import early_profile, grade, CRITERIA, paired_development, OPERATIONS
from tools.audit_curriculum_placement_course_v1 import journal_evidence
from tools.curriculum_placement_objective_v1 import METHOD, COMPONENTS


def fixture():
    records=[]; predictions=[]
    def add(source, kinds, label, probability, late=False):
        rid=str(len(records)); keys=['a','b']
        records.append(dict(_meta=dict(id=rid), state=dict(round_wind=0,hand_number=4 if late else 1),
            questions=dict(action=dict(src=source,label=keys[label],criteria={k:dict(type=v) for k,v in zip(keys,kinds)}))))
        predictions.append(dict(id=rid,keys=keys,p=probability))
    for source in ('legacy','teacher'):
        add('curriculum2:basic:'+source+':plain',['dahai','dahai'],0,[.8,.2])
        add('curriculum2:basic:'+source+':riichi_yes',['reach','dahai'],0,[.8,.2])
        add('curriculum2:basic:'+source+':riichi_no',['reach','dahai'],1,[.2,.8])
    for operation in OPERATIONS:
        add('curriculum2:advanced:kan',['none',operation],1,[.2,.8])
        add('curriculum2:advanced:kan',['none',operation],0,[.8,.2])
    add('curriculum2:basic:teacher:riichi_yes',['reach','dahai'],0,[.1,.9],late=True)
    return records,predictions


def comparison(top2=0., net=-1000., fourth=.25):
    return dict(consumed_development_only=True, independent_strength_or_parity_proof=False,
        metrics=dict(top2_rate=dict(gain=top2), net_points=dict(gain=net), fourth_rate=dict(gain=fourth)))


class PlacementAssessmentTests(unittest.TestCase):
    def test_changed_late_instructions_do_not_dilute_early_label_retention(self):
        records,predictions=fixture(); result=early_profile(records,predictions)
        self.assertEqual(result['late_records_excluded'],1)
        self.assertEqual(result['plain_agreement'],dict(legacy=1.,teacher=1.))
        self.assertEqual(result['RI_balanced_accuracy'],dict(legacy=1.,teacher=1.))
        self.assertEqual(set(result['original_call_balanced_accuracy'].values()),{1.})
        # The fourth-place and net diagnostics do not override the stated top2 objective.
        assessed=grade(result,result,comparison(top2=.03125),CRITERIA)
        self.assertTrue(assessed['passed']); self.assertFalse(assessed['task_completion_or_release_authorized'])
        self.assertTrue(assessed['independent_v8_strength_not_established'])

    def test_early_regression_missing_classes_changed_cases_and_top2_decline_fail(self):
        records,predictions=fixture(); first=early_profile(records,predictions)
        bad=copy.deepcopy(predictions); bad[1]['p']=[.1,.9]
        last=early_profile(records,bad)
        self.assertFalse(grade(first,last,comparison(top2=.1),CRITERIA)['passed'])
        self.assertFalse(grade(first,first,comparison(top2=-1/64),CRITERIA)['passed'])
        records.pop(2); predictions.pop(2); missing=early_profile(records,predictions)
        self.assertFalse(grade(missing,missing,comparison(top2=.1),CRITERIA)['passed'])
        with self.assertRaises(ValueError): grade(first,missing,comparison(),CRITERIA)
        changed=dict(CRITERIA,conditional_fourth_escape_veto=True)
        with self.assertRaises(ValueError): grade(first,first,comparison(),changed)

    def test_complete_paired_seed_matrix_checks_duplicate_cells_and_financial_counters(self):
        zero=[]
        for seed in range(100,108):
            for profile in ('efficiency','calling'):
                for seat in range(4):
                    scores=[40000,30000,20000,10000]
                    zero.append(dict(seed=seed,profile=profile,seat=seat,completed=True,ledger_passed=True,
                        final_scores=scores,rank=seat+1,net_points=scores[seat]-25000,hands=4,wins=1,dealin_hands=0))
        final=copy.deepcopy(zero)
        for row in final:
            seat=row['seat']; row['final_scores'][seat]+=1000; row['final_scores'][(seat+1)%4]-=1000; row['net_points']+=1000
        paired=paired_development(final,zero,100)
        self.assertEqual(paired['metrics']['net_points']['gain'],1000)
        self.assertEqual(paired['metrics']['top2_rate']['gain'],0.)
        self.assertFalse(paired['independent_strength_or_parity_proof'])
        for bad in (final[:-1],final[:-1]+[final[0]], [dict(r,net_points=999) for r in final]):
            with self.assertRaises(ValueError): paired_development(bad,zero,100)

    def test_journal_reconstructs_actual_weighted_objective_and_rejects_partial_exposure(self):
        components={k:0. for k in COMPONENTS}; components.update(objective=1.5,standard_target_ce=1.,extra_RI_group_ce=.5)
        orders=[dict(source_exposure_counts_by_update=[{'s':2}]),dict(source_exposure_counts_by_update=[{'s':2}])]
        reg=dict(curriculum=dict(stage_offset=5896),registered_final_step=2,
            expected_orders=orders,expected_record_exposures=4,course_loss_weights={'s':2.})
        logs=[dict(local_step=step,global_step=5896+step,source_counts={'s':2},source_loss_components={'s':components},
            loss_method=METHOD,reward=None,loss=3.,ce=2.,extra_RI_class_loss=1.,lr=1e-6,grad_norm=1.,records_seen=2*step,
            raw_upstream_metrics=dict(optimizer_step=step,**{'train/records_seen':2*step,'train/objective':3.})) for step in (1,2)]
        proof=journal_evidence(reg,logs,orders)
        self.assertEqual(proof['actual_source_exposures'],{'s':4})
        for field,value in (('loss',2.),('records_seen',3),('reward',0.),('global_step',0)):
            bad=copy.deepcopy(logs); bad[-1][field]=value
            with self.assertRaises(ValueError): journal_evidence(reg,bad,orders)
        with self.assertRaises(ValueError): journal_evidence(reg,logs[:1],orders)
