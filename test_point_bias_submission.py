import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.train_point_bias import advance


class PointBiasSubmissionTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.source=Path(temporary.name)/'run';(self.source/'native-development').mkdir(parents=True)
        self.selection={'checkpoint':'candidate','candidate_head_sha256':'fixed-head','config':{'out':str(self.source)+'-final'}}
        environment=patch.dict(os.environ,{'SLURM_JOB_ID':'701','SLURMD_NODENAME':'test-node'})
        environment.start();self.addCleanup(environment.stop)

    def test_dependencies_use_actual_returned_confirmation_handle(self):
        with patch('scripts.train_point_bias.audit_development',return_value={'selected':{}}), \
             patch('scripts.train_point_bias.prepare',return_value=self.selection), \
             patch('scripts.train_point_bias.subprocess.check_output',side_effect=['801\n','802\n','803\n']) as submit:
            record=advance(self.source)
        self.assertEqual(record['confirmation_job_id'],'801')
        self.assertIn('--dependency=afterok:701',submit.call_args_list[0].args[0])
        for call in submit.call_args_list[1:]:self.assertIn('--dependency=afterok:801',call.args[0])

    def test_partial_failure_preserves_handle_and_forbids_duplicate_submission(self):
        with patch('scripts.train_point_bias.audit_development',return_value={'selected':{}}), \
             patch('scripts.train_point_bias.prepare',return_value=self.selection), \
             patch('scripts.train_point_bias.subprocess.check_output',side_effect=['801\n',subprocess.CalledProcessError(1,'sbatch')]) as submit:
            with self.assertRaises(subprocess.CalledProcessError):advance(self.source)
            ledger=json.loads((self.source/'continuation-jobs.json').read_text())
            self.assertEqual(ledger['confirmation_job_id'],'801')
            with self.assertRaisesRegex(ValueError,'do not duplicate'):advance(self.source)
            self.assertEqual(submit.call_count,2)

    def test_no_positive_candidate_never_submits_a_gpu_confirmation(self):
        with patch('scripts.train_point_bias.audit_development',return_value={'selected':None}), \
             patch('scripts.train_point_bias.subprocess.check_output') as submit:
            record=advance(self.source)
        self.assertEqual(record['status'],'stop_no_positive_native_candidate')
        submit.assert_not_called()


if __name__=='__main__':unittest.main()
