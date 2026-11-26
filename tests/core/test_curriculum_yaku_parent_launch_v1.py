"""Prospective guard contracts; synthetic fixtures never admit a real parent."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.curriculum_yaku_retained_parent_v1 import require_qualified_parent, PARENT_LOSS
from tools.curriculum_progression_parent_loss_v1 import MASS as PARENT_MASS
from tools.curriculum_yaku_parent_runtime_v1 import require_registration
from tools.curriculum_yaku_parent_loss_v1 import MASS, METHOD
from tools.curriculum_yaku_parent_launch_v1 import require_same_recipe, CONFIG, ROOT
from tools.curriculum_complete_legal_class_policy_v1 import METHOD as ROUTE
from tools.train_curriculum_yaku_parent_v1 import write_conditional_config


def parent_fixture():
    parent = dict(status='progression2400_registered_not_started', registered_final_step=2400,
        train_records=87595, validation_records=23688, course_loss_mass=PARENT_MASS,
        training_config={'loss_method': PARENT_LOSS}, both_failed_3A_parameter_updates_excluded=True,
        curriculum=dict(epochs=2, accumulation=73, seed=2026112304, loss_method=PARENT_LOSS,
            selection_method=ROUTE, expected_record_exposures=175190,
            validation_and_game_steps=[0, 600, 1200, 1800, 2400], optimizer_resume=False, new_optimizer=True))
    done = dict(status='progression2400_fit_and_real_curves_complete', optimizer_steps=2400,
        actual_records_seen=175190, loss_method=PARENT_LOSS)
    evidence = dict(status='progression2400_final_evidence_audit_passed', optimizer_steps=2400,
        actual_records_seen=175190, stage3_advancement={'passed': True}, basic_retention={'passed': True},
        saved_final_reloaded_menus=23688, checked_complete_games=320,
        all376_optimizer_states_match=True, source_and_data_hashes_unchanged=True)
    execution = dict(status='actual_progression2400_full_evidence_independently_audited',
        loss_method=PARENT_LOSS, original_launch_binding_unchanged=True, actual_fit_job=123)
    return parent, done, evidence, execution, dict(parent_fit_job=123)


class RetainedYakuLaunchTests(unittest.TestCase):
    def test_synthetic_complete_contract_does_not_always_reject(self):
        require_qualified_parent(*parent_fixture())

    def test_failed_strategy_or_retention_is_rejected(self):
        for section in ('stage3_advancement', 'basic_retention'):
            values = parent_fixture(); values[2][section]['passed'] = False
            with self.subTest(section=section), self.assertRaises(ValueError): require_qualified_parent(*values)

    def test_incomplete_budget_or_reload_is_rejected(self):
        values = parent_fixture(); values[1]['optimizer_steps'] = 1800
        with self.assertRaises(ValueError): require_qualified_parent(*values)
        for field in ('saved_final_reloaded_menus', 'checked_complete_games'):
            values = parent_fixture(); values[2][field] -= 1
            with self.subTest(field=field), self.assertRaises(ValueError): require_qualified_parent(*values)

    def test_wrong_loss_or_failed_control_weights_are_rejected(self):
        values = parent_fixture(); values[3]['loss_method'] = 'old_class2'
        with self.assertRaises(ValueError): require_qualified_parent(*values)
        values = parent_fixture(); values[0]['both_failed_3A_parameter_updates_excluded'] = False
        with self.assertRaises(ValueError): require_qualified_parent(*values)

    def test_wrong_execution_or_source_evidence_is_rejected(self):
        values = parent_fixture(); values[3]['actual_fit_job'] = 999
        with self.assertRaises(ValueError): require_qualified_parent(*values)
        for field in ('all376_optimizer_states_match', 'source_and_data_hashes_unchanged'):
            values = parent_fixture(); values[2][field] = False
            with self.subTest(field=field), self.assertRaises(ValueError): require_qualified_parent(*values)

    def test_recipe_keeps_original_numerical_rules(self):
        c = json.loads(CONFIG.read_text())
        old = json.loads((ROOT/'configs/training/curriculum-yaku-fit-v2.json').read_text())
        require_same_recipe(c, old)
        for field in ('learning_rate', 'final_optimizer_steps', 'accumulation'):
            modified = copy.deepcopy(c); modified[field] *= 2
            with self.subTest(field=field), self.assertRaises(ValueError): require_same_recipe(modified, old)
        modified = copy.deepcopy(c); modified['development_targets']['min_six_source_mean_CE_improvement_from_step0'] = 0
        with self.assertRaises(ValueError): require_same_recipe(modified, old)

    def test_retained_runtime_requires_new_method_and_explicit_marker(self):
        c = dict(epochs=2, accumulation=73, seed=2026112305, loss_method=METHOD,
            selection_method=ROUTE, expected_record_exposures=182494,
            validation_and_game_steps=[0, 625, 1250, 1875, 2500], optimizer_resume=False, new_optimizer=True)
        reg = dict(status='yaku2500_registered_not_started', registered_final_step=2500,
            train_records=91247, validation_records=24516, curriculum=c, course_loss_mass=MASS,
            training_config={'loss_method': METHOD}, retained_parent_replay_objective_explicit=True)
        require_registration(reg)
        old = copy.deepcopy(reg); old['training_config']['loss_method'] = 'old_class2'
        with self.assertRaises(ValueError): require_registration(old)
        old = copy.deepcopy(reg); old['retained_parent_replay_objective_explicit'] = False
        with self.assertRaises(ValueError): require_registration(old)

    def test_conditional_configs_create_parent_and_preserve_exclusive_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'new/subdirectory/zero.json'
            write_conditional_config(path, {'contract_only': True})
            original = path.read_text()
            with self.assertRaises(FileExistsError): write_conditional_config(path, {'changed': True})
            self.assertEqual(path.read_text(), original)


if __name__ == '__main__': unittest.main()
