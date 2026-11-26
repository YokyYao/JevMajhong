"""Safety boundaries of execution partitioning, not native-game admission."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools import curriculum_targeted_parallel_assembly_v1 as assembly
from tools import launch_curriculum_targeted_parallel_v1 as launch


class ParallelLifecycleBoundaryTests(unittest.TestCase):
    def test_partial_lane_collection_cannot_create_a_complete_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            destination=Path(directory)/'full-shard'
            # This unit fixture only exercises the boundary before raw cases
            # or native verification. It is never a source/admission receipt.
            dispatch=dict(assignments=[['a'],['b'],['c']])
            with patch.object(assembly,'FULL',destination),patch.object(assembly,'resource',return_value='unit-only'):
                with patch.object(assembly,'actual_dispatch',return_value=(dispatch,{},[])):
                    with self.assertRaises(ValueError): assembly.assemble([],[])
            self.assertFalse(destination.exists())

    def test_already_running_job_cannot_be_bound_as_pending(self):
        job='123'; entry=Path('/example/fixed-entry.slurm')
        raw=(f'JobId={job} Command={entry} JobState=RUNNING RunTime=00:00:01 '
            f'AllocTRES=cpu=4,gres/gpu=1 UserId=unit({os.getuid()})')
        observed=subprocess.CompletedProcess([],0,raw,'')
        with patch.object(launch,'scheduler',return_value=observed):
            with self.assertRaises(ValueError): launch.describe(job,'worker',entry)

    def test_every_submitted_job_is_held_until_reviewed_handoff(self):
        completed=subprocess.CompletedProcess([],0,'12345\n','')
        with patch.object(launch.subprocess,'run',return_value=completed) as run:
            self.assertEqual(launch.submit(Path('/example/entry.slurm'),['--lane',0],('afterok:1',)),'12345')
        command=run.call_args.args[0]
        self.assertIn('--hold',command)
        self.assertIn('--no-requeue',command)
        self.assertIn('--kill-on-invalid-dep=yes',command)
        self.assertIn('--dependency=afterok:1',command)


if __name__=='__main__': unittest.main()
