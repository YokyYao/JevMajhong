"""Synthetic contract checks; these do not admit a real parent or student fit."""
import copy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import curriculum_paired_placement_launch_v1 as launch
from tools import curriculum_placement_paired_parent_v1 as parent
from tools import prepare_curriculum_placement_paired_launch_v1 as admission


def fixture():
    evidence = dict(status='yaku2500_final_evidence_audit_passed',
        stage3B_advancement={'passed': True}, saved_final_reloaded_menus=24516,
        source_and_data_hashes_unchanged=True, all376_optimizer_states_match=True,
        final_reserved_grid_unused=True, final_identity='model', final_policy_identity='policy')
    binding = dict(status='conditional_yaku_terminal_capture_bound_before_fit_start',
        automatic_fit=False, strategy_grade_overridden=False, fit_job='12', audit_job='13')
    proofs = {}
    for role, job in (('fit', '12'), ('audit', '13')):
        raw = (f'JobId={job} JobState=COMPLETED ExitCode=0:0 NodeList=ws-test-001 '
            'AllocTRES=cpu=4,gres/gpu=1 StartTime=2026-11-24T01:00:00 '
            f'EndTime=2026-11-24T01:10:00 Command={launch.B_ENTRY}')
        proofs[role] = dict(job_id=job, successful=True, raw=raw,
            source='actual_terminal_scontrol', exit_receipt=dict(job_id=job, phase=role, exit_code=0))
    captured = dict(status='conditional_yaku_actual_terminals_preserved_not_placement_admitted',
        both_scheduler_jobs_successful=True, strategy_grade_overridden=False,
        child_parent_sealed=False, original_final_evidence=copy.deepcopy(evidence), scheduler_proofs=proofs)
    return [dict(registered_final_step=2500),
        dict(optimizer_steps=2500, status='yaku2500_fit_and_real_curves_complete', actual_records_seen=182494),
        evidence, dict(status='actual_yaku2500_full_evidence_independently_audited',
            original_launch_binding_unchanged=True, actual_fit_job=12),
        captured, binding, launch.B/'fit-plan/registered.json']


class PairedPlacementLaunchTests(unittest.TestCase):
    def test_numerical_recipe_retains_all_original_fields(self):
        old = json.loads((launch.ROOT/'configs/training/curriculum-placement-fit-v1.json').read_text())
        new = json.loads(launch.CONFIG.read_text())
        launch.require_same_recipe(new, old)
        for key, value in (('seed', 3), ('epochs', 3), ('learning_rate', 1e-5),
                ('optimizer_resume', True), ('loss_method', 'different'), ('automatic_release', True)):
            bad = dict(new); bad[key] = value
            with self.assertRaises(ValueError): launch.require_same_recipe(bad, old)

    def test_old_or_mixed_parent_paths_are_rejected(self):
        good = dict(parent_registration='runs/curriculum-yaku-paired-v1/fit-plan/registered.json',
            parent_terminal='runs/curriculum-placement-paired-v1/parent-terminal-v1/terminal-evidence-private.json',
            out='runs/curriculum-placement-paired-v1/parent')
        parent.require_paired_request(good)
        for key in good:
            bad = dict(good); bad[key] = 'runs/old-parent/record.json'
            with self.assertRaises(ValueError): parent.require_paired_request(bad)

    def test_terminal_bridge_preserves_raw_evidence_and_original_grade(self):
        args = fixture(); before = copy.deepcopy(args)
        args[4]['scheduler_proofs']['fit']['source'] = 'original_independent_audit_fit_archive'
        result = launch.normalized_parent_terminal(*args)
        self.assertEqual(result['parent_fit_job'], 12)
        self.assertEqual(result['parent_audit_job'], 13)
        self.assertEqual(result['original_strategy_grade'], before[2]['stage3B_advancement'])
        for role in ('fit', 'audit'):
            self.assertEqual(result['scheduler_proofs'][role]['raw'], before[4]['scheduler_proofs'][role]['raw'])
            self.assertEqual(result['scheduler_proofs'][role]['receipt'], before[4]['scheduler_proofs'][role]['exit_receipt'])
        self.assertFalse(result['child_parent_sealed'])
        self.assertEqual(result['student_updates'], 0)

    def test_failed_grade_intermediate_or_incomplete_parent_cannot_advance(self):
        for index, key, value in ((0, 'registered_final_step', 2375),
                (1, 'actual_records_seen', 114), (2, 'all376_optimizer_states_match', False),
                (2, 'stage3B_advancement', {'passed': False}),
                (2, 'final_reserved_grid_unused', False), (3, 'actual_fit_job', 14),
                (4, 'strategy_grade_overridden', True), (4, 'both_scheduler_jobs_successful', False)):
            args = fixture(); args[index][key] = value
            if index == 2: args[4]['original_final_evidence'] = copy.deepcopy(args[2])
            with self.assertRaises(ValueError): launch.normalized_parent_terminal(*args)

    def test_original_final_evidence_cannot_be_replaced_after_capture(self):
        args = fixture(); args[4]['original_final_evidence']['final_identity'] = 'other'
        with self.assertRaises(ValueError): launch.normalized_parent_terminal(*args)

    def test_wrong_terminal_role_job_receipt_or_archive_source_fails(self):
        for role, key, value in (('fit', 'job_id', '999'), ('fit', 'successful', False),
                ('audit', 'source', 'original_independent_audit_fit_archive'),
                ('audit', 'exit_receipt', None), ('audit', 'exit_receipt', dict(job_id='13', phase='fit', exit_code=0))):
            args = fixture(); args[4]['scheduler_proofs'][role][key] = value
            with self.assertRaises(ValueError): launch.normalized_parent_terminal(*args)

    def test_failed_cpu_login_or_wrong_command_row_cannot_claim_success(self):
        for old, new in (('COMPLETED', 'FAILED'), ('gres/gpu=1', 'gres/gpu=0'),
                ('ws-test-001', 'login-01'), (str(launch.B_ENTRY), str(launch.ENTRY))):
            args = fixture(); p = args[4]['scheduler_proofs']['fit']; p['raw'] = p['raw'].replace(old, new)
            with self.assertRaises(ValueError): launch.normalized_parent_terminal(*args)

    def test_running_role_must_match_actual_queued_job_and_entry(self):
        q = dict(integration_job=25)
        raw = f'JobId=25 JobState=RUNNING Command={launch.ENTRY}'
        with patch.object(launch, 'read', return_value=q), patch.dict(os.environ, {'SLURM_JOB_ID': '25'}):
            with patch.object(launch, 'scheduler', return_value=SimpleNamespace(returncode=0, stdout=raw)):
                self.assertEqual(launch.require_queued_job('integration'), q)
            with patch.object(launch, 'scheduler', return_value=SimpleNamespace(returncode=0, stdout=raw.replace('RUNNING', 'COMPLETED'))):
                with self.assertRaises(ValueError): launch.require_queued_job('integration')
            with patch.dict(os.environ, {'SLURM_JOB_ID': '26'}):
                with patch.object(launch, 'scheduler', return_value=SimpleNamespace(returncode=0, stdout=raw)):
                    with self.assertRaises(ValueError): launch.require_queued_job('integration')

    def test_new_admission_keeps_full_optimizer_state_requirement(self):
        state = dict(optimizer_step=2, records_seen=114, args=dict(resume=''),
            optimizer=dict(state={i: dict(step=2) for i in range(376)}))
        admission.require_control_state(state, 2)
        for bad in (dict(state, records_seen=113), dict(state, args=dict(resume='parent'))):
            with self.assertRaises(ValueError): admission.require_control_state(bad, 2)
        state['optimizer']['state'][375]['step'] = 2502
        with self.assertRaises(ValueError): admission.require_control_state(state, 2)

    def test_failure_capture_survives_parent_nonadmission_without_child_run(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); a = root/'a'; a.mkdir(); (root/'logs').mkdir()
            q = dict(fit_job=12, audit_job=13)
            def actual(job):
                raw = (f'JobId={job} JobState=CANCELLED ExitCode=0:0 NodeList=(null) '
                    f'EndTime=2026-11-24T01:10:00 Command={launch.ENTRY}')
                return SimpleNamespace(returncode=0, stdout=raw, stderr='')
            with patch.object(launch, 'ROOT', root), patch.object(launch, 'A', a), \
                    patch.object(launch, 'RUN', root/'absent-child'), \
                    patch.object(launch, 'resource', return_value='synthetic-fixture'), \
                    patch.object(launch, 'verify_binding'), \
                    patch.object(launch, 'require_queued_job', return_value=q), \
                    patch.object(launch, 'scheduler', side_effect=actual), \
                    patch.object(launch, 'sha', return_value='synthetic-digest'):
                launch.capture()
            result = json.loads((a/'placement-terminal-capture-v1/terminal-evidence-private.json').read_text())
            self.assertFalse(result['both_scheduler_jobs_successful'])
            self.assertIsNone(result['original_final_evidence'])
            self.assertFalse(result['independent_confirmation_registered'])
            self.assertFalse((root/'absent-child').exists())
            for p in result['scheduler_proofs'].values():
                self.assertEqual(p['state'], 'CANCELLED')
                self.assertIsNone(p['exit_receipt'])
