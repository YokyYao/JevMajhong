"""Reject accidental recipe changes in the isolated retention control."""
import copy
import unittest
from tools.curriculum_progression_parent_control_v1 import matched_recipe, RUN, ROOT
from tools.curriculum_progression_parent_loss_v1 import METHOD


class MatchedRecipeTests(unittest.TestCase):
    def fixture(self):
        control = dict(curriculum=dict(run_id='old', stage='old', out='old', plan_root='old',
            loss_method='old', development_targets={'unchanged': .6}, learning_rate=3e-6),
            training_config=dict(model_id='old', out='old', trainer_config='old',
                                 trainer_config_sha256='old', loss_method='old'))
        for k in ('source', 'source_sha256', 'parent_model', 'course_loss_weights',
                  'course_loss_mass', 'expected_record_exposures', 'train_records',
                  'validation_records', 'registered_final_step', 'expected_orders'):
            control[k] = {'unchanged': k}
        reg = copy.deepcopy(control)
        reg['curriculum'].update(run_id='jev-curriculum-progression-parent-retention-v1',
            stage='stage3-parent-loss-retained-v1', out=str(RUN.relative_to(ROOT)/'stage3-fit'),
            plan_root=str(RUN.relative_to(ROOT)/'fit-plan'), loss_method=METHOD)
        reg['training_config'].update(model_id=reg['curriculum']['run_id'],
            out=reg['curriculum']['out'], trainer_config=str(RUN/'fit-plan/trainer.toml'),
            trainer_config_sha256='new', loss_method=METHOD)
        reg['both_failed_3A_parameter_updates_excluded'] = True
        previous = {'out': 'old', 'data': 'same', 'lr': 3e-6}
        args = dict(previous, out=str(RUN/'stage3-fit/training-run'))
        return reg, control, args, previous

    def test_exact_isolated_change(self):
        matched_recipe(*self.fixture())

    def test_reject_data_parent_order_threshold_or_budget_changes(self):
        for key in ('parent_model', 'course_loss_mass', 'expected_orders', 'train_records',
                    'registered_final_step'):
            with self.subTest(key=key):
                values = self.fixture(); values[0][key] = 'changed'
                with self.assertRaises(ValueError): matched_recipe(*values)
        values = self.fixture(); values[0]['curriculum']['development_targets']['unchanged'] = .4
        with self.assertRaises(ValueError): matched_recipe(*values)

    def test_reject_hidden_numerical_change(self):
        values = self.fixture(); values[2]['lr'] = 1e-5
        with self.assertRaises(ValueError): matched_recipe(*values)
        values = self.fixture(); values[0]['both_failed_3A_parameter_updates_excluded'] = False
        with self.assertRaises(ValueError): matched_recipe(*values)


if __name__ == '__main__': unittest.main()
