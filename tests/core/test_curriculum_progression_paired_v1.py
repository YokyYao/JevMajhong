import copy
import unittest

from tools.curriculum_progression_paired_v1 import (
    METHOD, ROOT, RUN, require_failed_RI_control, require_matched_recipe)


class ConditionalPairedRetryTests(unittest.TestCase):
    def fixture(self):
        done = dict(status='progression2400_fit_and_real_curves_complete', optimizer_steps=2400,
                    actual_records_seen=175190)
        evidence = dict(status='progression2400_final_evidence_audit_passed', optimizer_steps=2400,
            actual_records_seen=175190, all376_optimizer_states_match=True,
            saved_final_reloaded_menus=23688, source_and_data_hashes_unchanged=True,
            stage3_advancement=dict(passed=False),
            basic_retention=dict(passed=False, riichi_balanced_gain_by_source=dict(legacy=-.04, teacher=.01)))
        execution = dict(status='actual_progression2400_full_evidence_independently_audited',
                         original_launch_binding_unchanged=True)
        last = dict(riichi_by_source=dict(legacy=dict(tp=1, fn=2, tn=2, fp=1),
                                          teacher=dict(tp=2, fn=1, tn=2, fp=1)))
        limits = dict(min_both_RI_recalls_by_legacy_and_teacher_source=.6,
                      max_RI_balanced_accuracy_drop_from_step0=.03)
        return done, evidence, execution, last, limits

    def test_completed_failed_RI_control_retains_failure_instead_of_admitting_parent_weights(self):
        result = require_failed_RI_control(*self.fixture())
        self.assertFalse(result['original_advancement_passed'])
        self.assertTrue(result['other_failed_conditions_not_waived'])
        self.assertEqual(result['actual_RI_recalls']['legacy']['accept'], 1/3)

    def test_incomplete_or_successful_control_and_unverified_audit_are_rejected(self):
        for index, key, value in ((0, 'optimizer_steps', 1800),
                                  (1, 'all376_optimizer_states_match', False),
                                  (2, 'original_launch_binding_unchanged', False)):
            values = self.fixture()
            values[index][key] = value
            with self.assertRaises(ValueError):
                require_failed_RI_control(*values)
        values = self.fixture()
        values[1]['stage3_advancement']['passed'] = True
        with self.assertRaises(ValueError):
            require_failed_RI_control(*values)

    def test_a_different_failed_condition_cannot_trigger_RI_retry(self):
        values = self.fixture()
        values[1]['basic_retention']['riichi_balanced_gain_by_source'] = dict(legacy=0., teacher=0.)
        values[3]['riichi_by_source']['legacy'] = dict(tp=2, fn=1, tn=2, fp=1)
        with self.assertRaisesRegex(ValueError, 'No actual failed RI'):
            require_failed_RI_control(*values)

    def test_empty_or_invalid_RI_counts_are_rejected(self):
        values = self.fixture()
        values[3]['riichi_by_source']['legacy'] = dict(tp=0, fn=0, tn=2, fp=1)
        with self.assertRaises(ValueError):
            require_failed_RI_control(*values)

    def matched(self):
        control = dict(curriculum=dict(run_id='control', stage='control', out='runs/control/fit',
            plan_root='runs/control/plan', seed=2026112304),
            training_config=dict(model_id='control', out='runs/control/fit',
                trainer_config='/control/trainer.toml', trainer_config_sha256='old', parent_initialization='qualified_calls'),
            source='fixed', source_sha256='source', parent_model=dict(identity='qualified_calls'),
            course_loss_weights=dict(a=1.), course_loss_mass=dict(a=1.), expected_record_exposures=175190,
            train_records=87595, validation_records=23688, registered_final_step=2400)
        reg = copy.deepcopy(control)
        reg['curriculum'].update(run_id='jev-curriculum-progression-riichi-paired-v1', stage='stage3-riichi-paired-v1',
            out=str(RUN.relative_to(ROOT)/'stage3-fit'), plan_root=str(RUN.relative_to(ROOT)/'fit-plan'),
            paired_order_method=METHOD, retry_from_qualified_calls_not_control_final=True)
        reg['training_config'].update(model_id=reg['curriculum']['run_id'], out=reg['curriculum']['out'],
            trainer_config=str(RUN/'fit-plan/trainer.toml'), trainer_config_sha256='new')
        arguments = dict(out='/control/output', lr=3e-6, seed=2026112304, epochs=2)
        return reg, control, dict(arguments, out=str(RUN/'stage3-fit/training-run')), arguments

    def test_only_recorded_output_identity_can_change_in_matched_numerical_recipe(self):
        require_matched_recipe(*self.matched())
        values = self.matched()
        values[2]['lr'] = 6e-6
        with self.assertRaises(ValueError):
            require_matched_recipe(*values)
        values = self.matched()
        values[0]['parent_model'] = dict(identity='failed_3A_control')
        with self.assertRaises(ValueError):
            require_matched_recipe(*values)
        values = self.matched()
        values[0]['training_config']['parent_initialization'] = 'failed_3A_control'
        with self.assertRaises(ValueError):
            require_matched_recipe(*values)
        values = self.matched()
        values[0]['curriculum']['seed'] += 1
        with self.assertRaises(ValueError):
            require_matched_recipe(*values)


if __name__ == '__main__':
    unittest.main()
