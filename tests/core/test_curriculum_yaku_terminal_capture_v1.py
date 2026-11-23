from pathlib import Path
import unittest

from tools.capture_curriculum_yaku_terminal_v1 import accounting_proof


class TerminalAccountingTests(unittest.TestCase):
    def row(self, **changes):
        values=dict(jid='9123', state='COMPLETED', exit_code='0:0', node='ws-synthetic',
            tres='cpu=4,gres/gpu=1', start='2026-11-24T01:00:00', end='2026-11-24T02:00:00',
            command='sbatch --export=ALL,JEV_YAKU_PHASE=audit cluster/example.slurm')
        values.update(changes)
        return '|'.join(values.values())+'\n'

    def test_original_accounting_is_preserved_without_fabricating_scontrol(self):
        raw=self.row(); receipt=dict(job_id='9123',exit_code=0)
        proof=accounting_proof(raw,receipt,9123,Path('/synthetic/cluster/example.slurm'),Path('/synthetic'))
        self.assertEqual(proof['raw'],raw)
        self.assertFalse(proof['synthesized_scontrol_record'])

    def test_incomplete_cpu_wrong_command_or_failed_records_are_rejected(self):
        receipt=dict(job_id='9123',exit_code=0)
        bad=[self.row(state='RUNNING'),self.row(exit_code='1:0'),self.row(node='lo-synthetic'),
            self.row(tres='cpu=4'),self.row(end='Unknown'),self.row(command='sbatch cluster/other.slurm')]
        for raw in bad:
            with self.assertRaises(ValueError):
                accounting_proof(raw,receipt,9123,Path('/synthetic/cluster/example.slurm'),Path('/synthetic'))

    def test_duplicate_or_wrong_exact_job_records_are_rejected(self):
        for raw in (self.row()+self.row(),self.row(jid='9123.batch')):
            with self.assertRaises(ValueError):
                accounting_proof(raw,dict(job_id='9123',exit_code=0),9123,
                    Path('/synthetic/cluster/example.slurm'),Path('/synthetic'))
