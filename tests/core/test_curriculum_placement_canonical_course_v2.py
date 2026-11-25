"""Reject obsolete lineage and preserve actual full placement/control bodies."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from tools import curriculum_placement_canonical_course_v2 as course
from tests.core.test_curriculum_placement_parent_registration_v1 import fixture
from tests.core.test_curriculum_placement_runtime_v1 import registration_fixture


def recipe_fixture():
    _,data,sealed=fixture()
    sealed.update(placement_parent_lineage_method=course.parent.METHOD,
        canonical_parent_selected_step=2472,placement_parent_selected_step=2500,
        parent_registration=str(course.parent.PARENT))
    return course.read(course.CONFIG),data,sealed


def emitted_fixture():
    reg=registration_fixture()
    reg['curriculum'].update(course.IDENTITY,loss_method=course.runtime.LOSS,stage_offset=5968)
    reg['training_config']=dict(loss_method=course.runtime.LOSS,model_id=course.IDENTITY['run_id'])
    return reg


class CanonicalPlacementCourseTests(unittest.TestCase):
    def test_numerical_config_and_original_globals_do_not_change(self):
        actual=course.read(course.CONFIG);old=course.read(course.balanced.CONFIG)
        self.assertEqual(actual,dict(old,**course.IDENTITY));course.require_config(actual)
        self.assertIs(course.balanced.parent,course.balanced.require_recipe.__globals__['parent'])
        for field,value in [('epochs',3),('learning_rate',1e-4),('optimizer_resume',True),('out','runs/production')]:
            with self.subTest(field=field),self.assertRaises(ValueError):
                course.require_config(dict(actual,**{field:value}))

    def test_actual_lineage_full_source_budget_and_weights_are_required(self):
        c,d,s=recipe_fixture();before=copy.deepcopy((c,d,s))
        actual=course.require_recipe(c,d,s)
        self.assertEqual(actual,course.registration.require_recipe(c,d,s));self.assertEqual((c,d,s),before)
        for mutation in ('grade','source','mass','old_lineage','old_parent','old_step'):
            c,d,s=recipe_fixture()
            if mutation=='grade':s['original_parent_grade']['passed']=False
            elif mutation=='source':d['source_exposures'].pop(next(iter(d['source_exposures'])))
            elif mutation=='mass':d['scalar_example_weights'][next(iter(d['scalar_example_weights']))]*=.9
            elif mutation=='old_lineage':s['placement_parent_lineage_method']=course.balanced.parent.METHOD
            elif mutation=='old_parent':s['parent_registration']=str(course.balanced.parent.PARENT)
            else:s['canonical_parent_selected_step']=2400
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):course.require_recipe(c,d,s)

    def test_register_train_audit_and_control_are_original_code_objects(self):
        self.assertIs(course.registration_function().__code__,course.registration.original.register.__code__)
        self.assertIs(course.train_function().__code__,course.training.train.__code__)
        self.assertIs(course.audit_function().__code__,course.auditing.audit.__code__)
        self.assertIs(course.integration_function().__code__,course.control.original.preflight.__code__)

    def test_train_and_control_share_real_losses_and_canonical_parent(self):
        train=course.train_function().__globals__;control=course.integration_function().__globals__
        for name in ('annotate','weighted_loss','summarize'):self.assertIs(train[name],control[name])
        for scope in (train,control,course.audit_function().__globals__):
            self.assertIs(scope['verify_parent'],course.parent.verify_parent)
            self.assertIs(scope['require_registration'],course.require_registration)
        self.assertIs(train['parent_loader'],course.parent.parent_loader)

    def test_emitted_registration_is_canonical_and_keeps_fresh_child_baseline(self):
        incoming=emitted_fixture();before=copy.deepcopy(incoming)
        with patch.object(course.registration.original,'write') as writer:
            course.registration_function().__globals__['write'](Path('/fixture/registered.json'),incoming)
            result=writer.call_args.args[1]
        self.assertEqual(incoming,before);course.require_registration(result)
        self.assertEqual(result['placement_parent_lineage_method'],course.parent.METHOD)
        for mutation in ('lineage','offset','baseline','budget'):
            wrong=copy.deepcopy(result)
            if mutation=='lineage':wrong['placement_parent_lineage_method']=course.balanced.parent.METHOD
            elif mutation=='offset':wrong['curriculum']['stage_offset']=5896
            elif mutation=='baseline':wrong['curriculum']['new_policy_zero64_actual_games_required']=False
            else:wrong['curriculum']['epochs']=1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):course.require_registration(wrong)

    def test_control_metadata_keeps_real_status_and_is_not_course_training(self):
        incoming=dict(status='fixture-only',optimizer_steps=2)
        with patch.object(course.control.original,'write') as writer:
            course.integration_function().__globals__['write'](Path('/fixture/integration-proof-private.json'),incoming)
            proof=writer.call_args.args[1]
        self.assertEqual(incoming,dict(status='fixture-only',optimizer_steps=2))
        self.assertEqual(proof['status'],'fixture-only')
        self.assertEqual(proof['placement_parent_lineage_method'],course.parent.METHOD)
        self.assertEqual(proof['global_student_stage_offset'],5968)
        self.assertEqual(proof['loss_method'],course.control.loss.METHOD)

    def test_actual_registration_and_control_bind_new_software_sources(self):
        for function in (course.registration_function(),course.integration_function()):
            for name in (*course.PUBLIC,*course.parent.PUBLIC,*course.parent.canonical.PUBLIC):
                self.assertIn(name,function.__globals__['PUBLIC_FILES'])

    def test_missing_preflight_cannot_invoke_register_or_model_integration(self):
        with patch.object(course,'resource',return_value='fixture-only'), \
                patch.object(course,'read',side_effect=FileNotFoundError('entry absent')), \
                patch.object(course,'registration_function') as register, \
                patch.object(course,'integration_function') as integrate:
            with self.assertRaises(FileNotFoundError):course.register(Path('/unused'),Path('/unused'))
            with self.assertRaises(FileNotFoundError):course.integrate()
            register.assert_not_called();integrate.assert_not_called()

    def test_no_old_registration_or_student_output_can_be_used_as_control(self):
        with patch.object(course,'require_entry'),patch.object(course,'integration_function') as integrate:
            for registration,out in ((course.balanced.ROOT/course.balanced.IDENTITY['plan_root']/'registered.json',course.INTEGRATION_OUT),
                    (course.REGISTERED,course.ROOT/course.IDENTITY['out'])):
                with self.assertRaises(ValueError):course.integrate(registration,out)
            integrate.assert_not_called()


if __name__=='__main__':unittest.main()
