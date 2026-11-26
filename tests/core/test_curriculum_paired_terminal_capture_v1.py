from pathlib import Path
import unittest

from tools.capture_curriculum_paired_terminal_v1 import terminal_record


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
