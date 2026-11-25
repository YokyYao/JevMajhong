"""Ancestry and report-only admission change; learning math stays identical."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from tools import curriculum_placement_early_course_v1 as course
from tests.core.test_curriculum_placement_parent_registration_v1 import fixture
from tests.core.test_curriculum_placement_runtime_v1 import registration_fixture


def recipe_fixture():
    _,data,sealed=fixture();config=course.read(course.CONFIG)
    sealed.update(status=course.parent.STATUS,method=course.parent.PARENT_METHOD,
        placement_parent_lineage_method=course.parent.METHOD,selected3A_parent_step=1800,
        placement_parent_selected_step=2500,stage3B_grade_is_report_not_admission_veto=True,
        parent_registration=str(course.parent.PARENT),original_parent_grade=dict(passed=False))
    return config,data,sealed


def emitted_fixture():
    reg=registration_fixture()
    reg['curriculum'].update(course.IDENTITY,loss_method=course.runtime.LOSS,stage_offset=5296)
    reg['training_config']=dict(loss_method=course.runtime.LOSS,model_id=course.IDENTITY['run_id'])
    return reg


class EarlyPlacementCourseTests(unittest.TestCase):
    def test_failed_grade_is_an_unchanged_report_but_every_numeric_contract_remains(self):
        c,d,s=recipe_fixture();before=copy.deepcopy((c,d,s))
        result=course.require_recipe(c,d,s)
        self.assertEqual(result,d['proposed_schedule']);self.assertEqual((c,d,s),before)
        for mutation in ('counts','weights','schedule','epochs','cache','method','optimizer'):
            c,d,s=recipe_fixture();source=next(iter(d['source_exposures']))
            if mutation=='counts':d['source_exposures'][source]+=1
            elif mutation=='weights':d['scalar_example_weights'][source]*=.9
            elif mutation=='schedule':d['proposed_schedule']['proposed_final_step']-=1
            elif mutation=='epochs':c['epochs']=3
            elif mutation=='cache':s['parent_games_not_admitted_as_child_step0_cache']=False
            elif mutation=='method':s['method']='old-parent'
            else:c['optimizer_resume']=True
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):course.require_recipe(c,d,s)

    def test_old_recipe_grade_guard_is_not_mutated(self):
        c,d,s=recipe_fixture();c=copy.deepcopy(c);c['loss_method']=course.registration.original.LOSS
        s.update(method=course.registration.original.PARENT_METHOD,status=course.registration.original.STATUS)
        with self.assertRaises(ValueError):course.registration.original.require_recipe(c,d,s)
        with self.assertRaises(ValueError):course.parent.report_grade_only(course.require_config)

    def test_registration_binds_actual_offset_and_preserves_full_runtime(self):
        incoming=emitted_fixture();before=copy.deepcopy(incoming)
        with patch.object(course.registration.original,'write') as output:
            course.registration_function().__globals__['write'](Path('/fixture/registered.json'),incoming)
        actual=output.call_args.args[1];course.require_registration(actual)
        self.assertEqual(incoming,before);self.assertEqual(actual['curriculum']['stage_offset'],5296)
        for key,value in (('stage_offset',5968),('new_policy_zero64_actual_games_required',False)):
            bad=copy.deepcopy(actual);bad['curriculum'][key]=value
            with self.assertRaises(ValueError):course.require_registration(bad)

    def test_actual_training_and_control_use_identical_learning_functions(self):
        self.assertIs(course.registration_function().__code__,course.registration.original.register.__code__)
        self.assertIs(course.train_function().__code__,course.training.train.__code__)
        self.assertIs(course.audit_function().__code__,course.auditing.audit.__code__)
        control=course.integration_function().__globals__;train=course.train_function().__globals__
        for name in ('annotate','weighted_loss','summarize'):self.assertIs(control[name],train[name])
        self.assertIs(control['verify_parent'],course.parent.verify_parent)
        self.assertIs(train['parent_loader'],course.parent.parent_loader)

    def test_integration_metadata_records_real_ancestry_without_becoming_training(self):
        original=course.control.original;value=dict(status='fixture-only',optimizer_steps=2)
        with patch.object(original,'write') as output:
            course.integration_function().__globals__['write'](Path('/fixture/integration-proof-private.json'),value)
        actual=output.call_args.args[1]
        self.assertEqual(actual['global_student_stage_offset'],5296)
        self.assertEqual(actual['placement_parent_lineage_method'],course.parent.METHOD)
        self.assertEqual(value,dict(status='fixture-only',optimizer_steps=2))

    def test_software_binding_is_required_before_model_or_fit_work(self):
        with patch.object(course,'resource'),patch.object(course,'read',side_effect=FileNotFoundError), \
                patch.object(course,'train_function') as trainer:
            with self.assertRaises(FileNotFoundError):course.train(course.REGISTERED)
        trainer.assert_not_called()


if __name__=='__main__':unittest.main()
