import copy
import unittest

from tools.register_curriculum_placement_v1 import require_recipe,LOSS,ROUTE,SCHEMA,PARENT_METHOD,STATUS
from tools.curriculum_placement_mix_v1 import layout,REPLAY_MASS,NEW_SOURCES,RECORDS
from tools.prepare_curriculum_placement_exposures_v1 import proposed_schedule


def fixture():
    counts={s:max(1,int(v*RECORDS)) for s,v in REPLAY_MASS.items()}
    counts[max(counts,key=counts.get)]+=RECORDS-sum(counts.values())
    counts.update({s:8 for s in NEW_SOURCES});plan=layout(counts,{s:4 for s in counts});total=plan['virtual_exposures_per_pass']
    data=dict(status='placement_expanded_corpus_and_two_epoch_recipe_prepared_not_parent_or_fit_registered',
        optimizer_updates=0,student_fit_registered=False,order_seed=2026112306,
        source_loss_mass=plan['proposed_loss_mass'],source_exposures=plan['virtual_exposures_by_source'],
        scalar_example_weights=plan['scalar_example_weights'],training_exposures_per_pass=total,
        files=dict(train=dict(records=total),validation=dict(records=100)),unique_validation_records=100,
        proposed_schedule=proposed_schedule(total))
    config=dict(status='planned_not_registered_or_fit_admitted',seed=2026112306,epochs=2,batch=1,
        learning_rate=3e-6,head_learning_rate=1.5e-5,new_optimizer=True,optimizer_resume=False,
        final_candidate_only=True,automatic_release=False,loss_method=LOSS,selection_method=ROUTE,
        placement_input_schema=SCHEMA)
    sealed=dict(status=STATUS,method=PARENT_METHOD,selected_step=2500,original_parent_grade={'passed':True},
        same_execution_policy_as_parent=False,parent_games_not_admitted_as_child_step0_cache=True)
    return config,data,sealed


class PlacementRegistrationTests(unittest.TestCase):
    def test_fixed_full_exposure_budget_and_exact_source_masses_are_retained(self):
        c,d,s=fixture();before=copy.deepcopy((c,d,s));schedule=require_recipe(c,d,s)
        self.assertEqual((c,d,s),before)
        self.assertEqual(schedule['expected_record_exposures'],2*d['files']['train']['records'])
        self.assertEqual(len(schedule['validation_and_game_steps']),5)

    def test_resume_parent_grade_policy_or_hyperparameter_changes_fail(self):
        for field,value in (('optimizer_resume',True),('new_optimizer',False),('epochs',1),
                ('learning_rate',1e-4),('selection_method','old'),('final_candidate_only',False)):
            c,d,s=fixture();c[field]=value
            with self.assertRaises(ValueError):require_recipe(c,d,s)
        c,d,s=fixture();s['original_parent_grade']['passed']=False
        with self.assertRaises(ValueError):require_recipe(c,d,s)
        c,d,s=fixture();s['parent_games_not_admitted_as_child_step0_cache']=False
        with self.assertRaises(ValueError):require_recipe(c,d,s)

    def test_silent_exposure_weight_or_budget_changes_fail(self):
        for kind in ('counts','weight','schedule','validation'):
            c,d,s=fixture();source=NEW_SOURCES[0]
            if kind=='counts':d['source_exposures'][source]+=1
            elif kind=='weight':d['scalar_example_weights'][source]*=.9
            elif kind=='schedule':d['proposed_schedule']['proposed_final_step']-=1
            else:d['files']['validation']['records']+=1
            with self.assertRaises(ValueError):require_recipe(c,d,s)
