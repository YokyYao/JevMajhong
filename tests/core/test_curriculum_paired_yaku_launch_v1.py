import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import curriculum_yaku_paired_parent_v1 as parent
from tools import curriculum_paired_yaku_launch_v1 as launch
from tools.curriculum_transfer import sha


class PairedYakuAdmissionTests(unittest.TestCase):
    def fixture(self):
        registered = dict(registered_final_step=2400,
            same_parent_data_objective_budget_only_epoch_order_changed=True)
        done = dict(status='progression2400_fit_and_real_curves_complete',
            optimizer_steps=2400, actual_records_seen=175190)
        evidence = dict(status='progression2400_final_evidence_audit_passed',
            optimizer_steps=2400, actual_records_seen=175190,
            stage3_advancement=dict(passed=True), saved_final_reloaded_menus=23688,
            all376_optimizer_states_match=True, source_and_data_hashes_unchanged=True)
        execution = dict(status='actual_progression2400_full_evidence_independently_audited',
            original_launch_binding_unchanged=True, actual_fit_job=123)
        return registered, done, evidence, execution, dict(parent_fit_job=123)

    def test_exact_final_independently_qualified_paired_parent_can_proceed(self):
        parent.require_qualified_parent(*self.fixture())

    def test_failed_strategy_is_rejected_despite_successful_execution(self):
        values = self.fixture(); values[2]['stage3_advancement']['passed'] = False
        with self.assertRaises(ValueError): parent.require_qualified_parent(*values)

    def test_partial_peak_control_or_wrong_actual_fit_is_rejected(self):
        for index, key, value in ((0, 'registered_final_step', 1800),
                (0, 'same_parent_data_objective_budget_only_epoch_order_changed', False),
                (1, 'optimizer_steps', 1800), (1, 'actual_records_seen', 175189),
                (2, 'all376_optimizer_states_match', False),
                (2, 'saved_final_reloaded_menus', 23687),
                (2, 'source_and_data_hashes_unchanged', False),
                (3, 'original_launch_binding_unchanged', False), (3, 'actual_fit_job', 999)):
            values = self.fixture(); values[index][key] = value
            with self.subTest(index=index, key=key), self.assertRaises(ValueError):
                parent.require_qualified_parent(*values)

    def test_actual_public_recipe_only_changes_explicit_parent_and_output(self):
        old = launch.read(launch.ROOT/'configs/training/curriculum-yaku-fit-v2.json')
        current = launch.read(launch.CONFIG)
        launch.require_same_recipe(current, old)
        for key, value in (('epochs', 3), ('learning_rate', 6e-6),
                ('stage_offset', 5796), ('seed', 1), ('checkpoint_every_steps', 250),
                ('final_optimizer_steps', 1250), ('expected_record_exposures', 999)):
            changed = copy.deepcopy(current); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                launch.require_same_recipe(changed, old)
        changed = copy.deepcopy(current)
        changed['development_targets']['max_RI_balanced_accuracy_drop_from_step0'] = 1.
        with self.assertRaises(ValueError): launch.require_same_recipe(changed, old)


class CapturedTerminalBridgeTests(unittest.TestCase):
    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream: json.dump(value, stream)

    def exercise(self, mutation=None):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run = root/'runs/curriculum-progression-riichi-paired-v1'
            cap = run/'terminal-capture-v2'
            source = root/'source.txt'; source.write_text('synthetic frozen capture source')
            binding_path = run/'terminal-capture-binding-v2-private.json'
            binding = dict(status='paired_retry_terminal_capture_bound_before_fit_completion',
                automatic_fit=False, strategy_grade_overridden=False, observed_step=128,
                observed_fit_scheduler='JobId=123 JobState=RUNNING', fit_job='123', audit_job='124',
                files={str(source): sha(source)})
            grade = dict(passed=True)
            result = dict(status='paired_retry_actual_terminals_preserved_not_child_admitted',
                both_scheduler_jobs_successful=True, original_strategy_grade=grade,
                strategy_grade_overridden=False, child_parent_sealed=False, student_updates=0,
                automatic_fit=False, automatic_release=False, scheduler_proofs={})
            command = root/'cluster/curriculum-progression-paired-v1.slurm'

            def raw(job, entry=command, tres='cpu=4,gres/gpu=1'):
                return (f'JobId={job} JobState=COMPLETED ExitCode=0:0 NodeList=ws-synthetic '
                    f'AllocTRES={tres} StartTime=2026-11-24T04:00:00 '
                    f'EndTime=2026-11-24T05:00:00 Command={entry}')

            for job, role in ((123, 'fit'), (124, 'audit')):
                receipt = dict(job_id=str(job), phase=role, exit_code=0)
                self.write(root/'logs'/f'curriculum-progression-paired-exit-{job}.json', receipt)
                result['scheduler_proofs'][role] = dict(job_id=str(job), exit_receipt=receipt,
                    successful=True, raw=raw(job))
            capture_receipt = dict(job_id='125', exit_code=0)
            self.write(root/'logs/curriculum-paired-terminal-v2-exit-125.json', capture_receipt)
            if mutation: mutation(binding, result)
            self.write(binding_path, binding)
            evidence_path = cap/'terminal-evidence-private.json'; self.write(evidence_path, result)
            self.write(cap/'completion-binding-private.json', dict(files={str(evidence_path): sha(evidence_path)}))
            config = dict(parent_terminal_capture_binding=str(binding_path), parent_fit_job=123,
                parent_audit_job=124, parent_terminal_capture_job=125)
            observed = SimpleNamespace(stdout=raw(125,
                root/'cluster/curriculum-paired-terminal-capture-v2.slurm'), returncode=0)
            with patch.object(parent, 'ROOT', root), patch.object(parent.subprocess, 'run', return_value=observed):
                files = {}; exits = parent.actual_terminals(config, run, dict(stage3_advancement=grade), files)
            self.assertEqual(set(exits), {'fit', 'audit', 'capture'})
            self.assertIn(str(binding_path), files)
            self.assertTrue((run/'capture-terminal-for-yaku-private.json').is_file())

    def test_original_two_phase_archives_and_actual_capture_terminal_are_bound(self):
        self.exercise()

    def test_changed_strategy_grade_or_post_completion_binding_is_rejected(self):
        mutations = [lambda b, r: b.update(observed_step=2400),
            lambda b, r: b.update(observed_fit_scheduler='JobId=123 JobState=COMPLETED'),
            lambda b, r: r.update(original_strategy_grade=dict(passed=False)),
            lambda b, r: r.update(both_scheduler_jobs_successful=False),
            lambda b, r: r['scheduler_proofs']['audit']['exit_receipt'].update(exit_code=1)]
        for mutation in mutations:
            with self.assertRaises((ValueError, RuntimeError)):
                self.exercise(mutation)


if __name__ == '__main__': unittest.main()
