from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.capture_curriculum_paired_terminal_v2 import terminal_record
from tools import capture_curriculum_paired_terminal_v1 as original
from tools import capture_curriculum_paired_terminal_v2 as corrected


class PairedTerminalCaptureTests(unittest.TestCase):
    command = Path('/synthetic/paired.slurm')

    def row(self, **changes):
        values = dict(JobId='123', JobState='COMPLETED', ExitCode='0:0',
            EndTime='2026-11-24T05:00:00', Command=str(self.command))
        values.update(changes)
        return ' '.join(f'{key}={value}' for key, value in values.items())

    def test_success_keeps_original_raw(self):
        raw = self.row()
        proof = terminal_record(raw, 123, self.command)
        self.assertEqual(proof['raw'], raw)
        self.assertTrue(proof['successful'])

    def test_failed_cancelled_and_timeout_are_preserved_without_success(self):
        for state in ('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL'):
            raw = self.row(JobState=state, ExitCode='1:0')
            proof = terminal_record(raw, 123, self.command)
            self.assertEqual(proof['state'], state)
            self.assertFalse(proof['successful'])

    def test_running_missing_wrong_job_or_wrong_entry_is_not_terminal_proof(self):
        for raw in ('', self.row(JobState='RUNNING'), self.row(JobId='124'),
                    self.row(Command='/synthetic/other.slurm'), self.row(EndTime='Unknown')):
            with self.assertRaises(ValueError):
                terminal_record(raw, 123, self.command)

    def test_completed_nonzero_exit_is_not_success(self):
        self.assertFalse(terminal_record(self.row(ExitCode='1:0'), 123, self.command)['successful'])


class ExclusiveWriteIntegrationTests(unittest.TestCase):
    def exercise(self, module):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run = root/'run'; run.mkdir(); (root/'logs').mkdir()
            binding = run/'binding.json'
            binding.write_text(json.dumps(dict(
                status='paired_retry_terminal_capture_bound_before_fit_completion',
                fit_job='123', audit_job='124', files={}, automatic_fit=False,
                strategy_grade_overridden=False)))
            for job, phase in (('123', 'fit'), ('124', 'audit')):
                (root/'logs'/f'curriculum-progression-paired-exit-{job}.json').write_text(
                    json.dumps(dict(job_id=job, phase=phase, exit_code=0)))
            (run/'final-evidence-private.json').write_text(json.dumps(
                dict(stage3_advancement=dict(passed=False))))

            def observed(job):
                raw = (f'JobId={job} JobState=COMPLETED ExitCode=0:0 NodeList=ws-synthetic '
                    f'AllocTRES=cpu=4,gres/gpu=1 StartTime=2026-11-24T04:00:00 '
                    f'EndTime=2026-11-24T05:00:00 Command={module.ENTRY}')
                return SimpleNamespace(stdout=raw, stderr='', returncode=0)

            with patch.multiple(module, ROOT=root, RUN=run, BINDING=binding), \
                    patch.object(module, 'resource', return_value='synthetic-test-allocation'), \
                    patch.object(module, 'scheduler', side_effect=observed):
                module.capture()
            result = json.loads((run/'terminal-capture-v2/terminal-evidence-private.json').read_text())
            self.assertTrue(result['both_scheduler_jobs_successful'])
            self.assertFalse(result['original_strategy_grade']['passed'])
            self.assertFalse(result['child_parent_sealed'])
            self.assertEqual(set(result['scheduler_proofs']), {'fit', 'audit'})
            self.assertEqual(result['scheduler_proofs']['audit']['job_id'], '124')
            self.assertTrue((run/'terminal-capture-v2/completion-binding-private.json').is_file())

    def test_original_two_observations_reproduce_exclusive_write_failure(self):
        # Preserve the launch-bound original source and demonstrate its actual
        # file-I/O defect without altering any real terminal or training record.
        with self.assertRaises(FileExistsError):
            self.exercise(original)

    def test_two_actual_roles_have_distinct_exclusive_files_and_grade_is_preserved(self):
        self.exercise(corrected)
