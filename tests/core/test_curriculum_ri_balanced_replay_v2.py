import copy
import types
import unittest
from unittest.mock import patch
from tools import curriculum_ri_balanced_replay_v2 as retry
from tools import curriculum_ri_balanced_replay_runtime_v1 as original


def code_shape(code, undo_path=False):
    constants = []
    for item in code.co_consts:
        if isinstance(item, types.CodeType):
            constants.append(code_shape(item, undo_path))
        elif undo_path and item == 'tools/evaluate_curriculum_ri_balanced_replay_v2.py':
            constants.append('tools/evaluate_curriculum_ri_balanced_replay_v1.py')
        else:
            constants.append(item)
    return code.co_code, code.co_names, code.co_varnames, code.co_freevars, constants


class CanonicalRetryTests(unittest.TestCase):
    def fixture(self):
        old = dict(curriculum={'development_targets':{'floor':.6}, 'seed':2026112304},
            training_config={'base_revision':'fixed', 'lr':3e-6, 'parent_initialization':{'step':388}},
            source='pinned', source_sha256='abc', parent_model={'identity':'parent'}, course_loss_weights={'fixed':1})
        old_args = dict(init_from='parent388', lr=3e-6, head_lr=1.5e-5, epochs=2, batch=1, accum=73)
        c = retry.curriculum_view(old['curriculum'])
        cfg = dict(old['training_config'], model_id=c['run_id'], out=c['out'],
            trainer_config=str(retry.RUN/'fit-plan/trainer.toml'), trainer_config_sha256='sha',
            registered_final_step=2472, loss_method=original.METHOD)
        reg = dict(status='balanced_replay2472_registered_not_started', registered_final_step=2472,
            train_records=87595, validation_records=23688, physical_train_slots_per_epoch=90228,
            expected_record_exposures=180456, course_loss_mass=original.MASS, curriculum=c, training_config=cfg,
            validation_loss_method=original.VALIDATION_METHOD, standby_only_after_control_failed_strategy=True,
            **{k:old[k] for k in ('source','source_sha256','parent_model','course_loss_weights')})
        return old, old_args, reg, retry.trainer_view(old_args)

    def test_numerical_recipe_and_original_strategy_limits_cannot_change(self):
        old, old_args, reg, args = self.fixture()
        retry.check_matched(reg, old, args, old_args)
        for key, value in [('lr', 1e-5), ('init_from', 'failed2400'), ('resume', 'failed_state')]:
            with self.assertRaises(ValueError): retry.check_matched(reg, old, dict(args, **{key:value}), old_args)
        changed = copy.deepcopy(reg); changed['curriculum']['development_targets']['floor'] = .1
        with self.assertRaises(ValueError): retry.check_matched(changed, old, args, old_args)
        self.assertNotEqual(retry.RUN, original.RUN)
        self.assertIn('fit-v2', str(retry.RUN))

    def test_original_train_bytecode_only_changes_the_declared_evaluator_path(self):
        with patch.object(retry, 'read', return_value={'training_config':{}, 'curriculum':{}}):
            function = retry.train_function('not-a-real-registration')
        self.assertEqual(code_shape(function.__code__, True), code_shape(retry.original_training.train.__code__))
        self.assertIs(function.__globals__['weighted_loss'], retry.original_training.weighted_loss)
        self.assertIs(function.__globals__['random_proxy'], retry.original_training.random_proxy)
        self.assertIs(function.__globals__['require_registration'], retry.require_registration)
        self.assertIsNot(function.__globals__['EpochReplay'], retry.original_training.EpochReplay)

    def test_require_prerequisites_rejects_missing_and_changed_actual_proof(self):
        with patch.object(retry, 'verify_binding', side_effect=ValueError('changed')):
            with self.assertRaises(ValueError): retry.require_prerequisites()
        with patch.object(retry, 'verify_binding', return_value={}), patch.object(retry, 'read', return_value={
                'status':'synthetic_or_incomplete'}):
            with self.assertRaises(ValueError): retry.require_prerequisites()

    def test_queue_cannot_be_duplicated(self):
        with patch.object(retry, 'resource'), patch.object(retry, 'read'), \
                patch.object(retry, 'require_registration'), patch.object(retry, 'require_launch'), \
                patch.object(retry.Path, 'exists', return_value=True), patch.object(retry, 'submit') as submit:
            with self.assertRaises(ValueError): retry.enqueue()
            submit.assert_not_called()

    def test_wrong_actual_job_refused_before_fit(self):
        with patch.object(retry, 'resource'), patch.object(retry, 'verify_binding', return_value={'fit_job':'123'}), \
                patch.dict(retry.os.environ, {'SLURM_JOB_ID':'456'}):
            with self.assertRaises(ValueError): retry.bound_job('fit')


if __name__ == '__main__': unittest.main()
