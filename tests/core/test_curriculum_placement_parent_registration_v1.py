"""Retained placement registration preserves full original numeric contracts."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from tools import register_curriculum_placement_parent_v1 as registration
from tests.core.test_curriculum_placement_registration_v1 import fixture as original_fixture
from tests.core.test_curriculum_placement_runtime_v1 import registration_fixture


def fixture():
    config, data, seal = original_fixture()
    config.update(loss_method=registration.runtime.LOSS, parent_loss_method=registration.runtime.PARENT_LOSS,
        retained_parent_replay_objective_explicit=True, upstream_normalized_terminal_alignment_required=True)
    seal.update(parent_loss_method=registration.runtime.PARENT_LOSS, child_loss_method=registration.runtime.LOSS,
        exact_retained_3B_replay_objective=True)
    return config, data, seal


class RetainedPlacementRegistrationTests(unittest.TestCase):
    def test_full_recipe_budget_and_masses_pass_without_mutating_inputs(self):
        config, data, seal = fixture(); before = copy.deepcopy((config, data, seal))
        schedule = registration.require_recipe(config, data, seal)
        self.assertEqual((config, data, seal), before)
        old, _, _ = original_fixture()
        self.assertEqual(schedule, registration.original.require_recipe(old, data, seal))

    def test_old_loss_seal_or_missing_explicit_markers_are_rejected(self):
        for mutation in ('config_loss', 'parent_loss', 'child_loss', 'retained', 'normalized'):
            config, data, seal = fixture()
            if mutation == 'config_loss': config['loss_method'] = registration.original.LOSS
            elif mutation == 'parent_loss': seal['parent_loss_method'] = 'old'
            elif mutation == 'child_loss': seal['child_loss_method'] = 'old'
            elif mutation == 'retained': config['retained_parent_replay_objective_explicit'] = False
            else: config['upstream_normalized_terminal_alignment_required'] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                registration.require_recipe(config, data, seal)

    def test_full_source_counts_weights_schedule_and_parent_grade_are_not_relaxed(self):
        for mutation in ('counts', 'weights', 'schedule', 'grade', 'epochs', 'resume'):
            config, data, seal = fixture(); source = next(iter(data['source_exposures']))
            if mutation == 'counts': data['source_exposures'][source] += 1
            elif mutation == 'weights': data['scalar_example_weights'][source] *= .9
            elif mutation == 'schedule': data['proposed_schedule']['proposed_final_step'] -= 1
            elif mutation == 'grade': seal['original_parent_grade']['passed'] = False
            elif mutation == 'epochs': config['epochs'] = 1
            else: config['optimizer_resume'] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                registration.require_recipe(config, data, seal)

    def test_original_register_code_object_has_isolated_globals(self):
        before = registration.original.LOSS
        function = registration.registration_function()
        self.assertIs(function.__code__, registration.original.register.__code__)
        self.assertIsNot(function.__globals__, registration.original.register.__globals__)
        self.assertEqual(function.__globals__['LOSS'], registration.runtime.LOSS)
        self.assertIs(function.__globals__['verify_parent'], registration.verify_parent)
        self.assertEqual(registration.original.LOSS, before)

    def test_final_writer_preserves_actual_loss_and_validates_the_full_runtime(self):
        result = registration_fixture(); result['curriculum']['loss_method'] = registration.runtime.LOSS
        result['training_config'] = {'loss_method': registration.runtime.LOSS}; before = copy.deepcopy(result)
        with patch.object(registration.original, 'write') as writer:
            hook = registration.registration_function().__globals__['write']
            hook(Path('/fixture/registered.json'), result)
            actual = writer.call_args.args[1]
            registration.runtime.require_registration(actual)
            self.assertTrue(actual['retained_parent_replay_objective_explicit'])
            self.assertTrue(actual['upstream_normalized_terminal_alignment_required'])
            self.assertEqual(result, before)
            for mutation in ('loss', 'baseline'):
                bad = copy.deepcopy(result)
                if mutation == 'loss': bad['training_config']['loss_method'] = registration.original.LOSS
                else: bad['curriculum']['stage_zero_seed_groups'] = 2
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    hook(Path('/fixture/registered.json'), bad)

    def test_prospective_config_changes_only_explicit_route_identity_and_output(self):
        new = registration.read(registration.ROOT/'configs/training/curriculum-placement-parent-fit-v1.json')
        old = registration.read(registration.ROOT/'configs/training/curriculum-placement-fit-v1.json')
        removed = {'run_id', 'stage', 'plan_root', 'out', 'loss_method', 'parent_loss_method',
            'retained_parent_replay_objective_explicit', 'upstream_normalized_terminal_alignment_required'}
        self.assertEqual({k:v for k,v in new.items() if k not in removed}, {k:v for k,v in old.items() if k not in removed})
        self.assertNotEqual(new['out'], old['out'])

    def test_missing_bound_entry_cannot_start_registration(self):
        with patch.object(registration, 'resource', return_value='fixture'), \
                patch.object(registration, 'read', side_effect=FileNotFoundError('entry missing')), \
                patch.object(registration, 'registration_function') as function:
            with self.assertRaises(FileNotFoundError): registration.register(Path('/unused'), Path('/unused'))
            function.assert_not_called()


if __name__ == '__main__': unittest.main()
