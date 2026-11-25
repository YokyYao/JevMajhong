"""Alternate placement uses the original full functions and original gates."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from tools import curriculum_placement_balanced_course_v1 as course
from tests.core.test_curriculum_placement_parent_registration_v1 import fixture
from tests.core.test_curriculum_placement_runtime_v1 import registration_fixture


def recipe_fixture():
    _, data, sealed = fixture()
    config = course.read(course.CONFIG)
    sealed.update(placement_parent_lineage_method=course.parent.METHOD,
        balanced_parent_selected_step=2472, placement_parent_selected_step=2500,
        parent_registration=str(course.parent.PARENT))
    return config, data, sealed


def emitted_fixture():
    reg = registration_fixture()
    reg['curriculum'].update(course.IDENTITY, loss_method=course.runtime.LOSS, stage_offset=5968)
    reg['training_config'] = dict(loss_method=course.runtime.LOSS, model_id=course.IDENTITY['run_id'])
    return reg


class BalancedPlacementCourseTests(unittest.TestCase):
    def test_config_only_changes_declared_route_identity(self):
        current = course.read(course.CONFIG)
        course.require_config(current)
        for key, value in [('epochs', 3), ('learning_rate', 1e-4), ('optimizer_resume', True),
                           ('out', 'runs/production')]:
            with self.assertRaises(ValueError): course.require_config(dict(current, **{key:value}))

    def test_recipe_keeps_original_numerical_schedule_and_complete_sources(self):
        c, d, s = recipe_fixture(); before = copy.deepcopy((c,d,s))
        actual = course.require_recipe(c,d,s)
        self.assertEqual((c,d,s), before)
        self.assertEqual(actual, course.registration.require_recipe(c,d,s))
        for mutation in ('grade', 'counts', 'mass', 'lineage'):
            c,d,s = recipe_fixture()
            if mutation == 'grade': s['original_parent_grade']['passed'] = False
            elif mutation == 'counts': d['source_exposures'].pop(next(iter(d['source_exposures'])))
            elif mutation == 'mass': d['scalar_example_weights'][next(iter(d['scalar_example_weights']))] *= .9
            else: s['parent_registration'] = str(course.parent.producer.PARENT)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): course.require_recipe(c,d,s)

    def test_registration_writer_retains_objective_and_adds_explicit_lineage(self):
        incoming = emitted_fixture(); before = copy.deepcopy(incoming)
        with patch.object(course.registration.original, 'write') as output:
            course.registration_function().__globals__['write'](Path('/fixture/registered.json'), incoming)
            actual = output.call_args.args[1]
            course.require_registration(actual)
            self.assertEqual(actual['placement_parent_lineage_method'], course.parent.METHOD)
            self.assertEqual(actual['curriculum']['stage_offset'],5968)
            self.assertEqual(incoming,before)
        for field, value in [('stage_offset',5896),('new_policy_zero64_actual_games_required',False)]:
            wrong=copy.deepcopy(actual);wrong['curriculum'][field]=value
            with self.assertRaises(ValueError):course.require_registration(wrong)

    def test_register_train_and_audit_bodies_are_exact_original_code_objects(self):
        self.assertIs(course.registration_function().__code__,course.registration.original.register.__code__)
        self.assertIs(course.train_function().__code__,course.training.train.__code__)
        self.assertIs(course.audit_function().__code__,course.auditing.audit.__code__)
        self.assertIs(course.train_function().__globals__['parent_loader'],course.parent.parent_loader)
        self.assertIs(course.audit_function().__globals__['verify_parent'],course.parent.verify_parent)

    def test_all_new_sources_are_in_actual_registration_public_bindings(self):
        public=course.registration_function().__globals__['PUBLIC_FILES']
        for name in (*course.PUBLIC,*course.parent.PUBLIC):self.assertIn(name,public)

    def test_missing_bound_preflight_refuses_before_any_registration(self):
        with patch.object(course,'resource',return_value='fixture-only'), \
                patch.object(course,'read',side_effect=FileNotFoundError('bound entry absent')), \
                patch.object(course,'registration_function') as function:
            with self.assertRaises(FileNotFoundError):course.register(Path('/unused'),Path('/unused'))
            function.assert_not_called()


if __name__=='__main__':unittest.main()
