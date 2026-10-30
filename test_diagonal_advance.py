import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.advance_diagonal_development import main


class DevelopmentContinuationTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.source=Path(temporary.name)/'source';(self.source/'native-development').mkdir(parents=True)
        self.selection={'checkpoint':'candidate','candidate_head_sha256':'fixed-head'}
        argv=patch('sys.argv',['advance','--run',str(self.source)]);argv.start();self.addCleanup(argv.stop)
        env=patch.dict(os.environ,{'SLURM_JOB_ID':'123','SLURMD_NODENAME':'test-node'});env.start();self.addCleanup(env.stop)

    def ledger(self):return json.loads((self.source/'continuation-jobs.json').read_text())

    def test_nonpositive_development_does_not_submit_gpu_jobs(self):
        with patch('scripts.advance_diagonal_development.audit_development',return_value={'selected':None}), \
             patch('scripts.advance_diagonal_development.submit') as submit:
            main();submit.assert_not_called()
        self.assertEqual(self.ledger()['status'],'stop_no_positive_native_candidate')
        self.assertIs(self.ledger()['goal_complete'],False)

    def test_confirmation_and_checks_use_persisted_dependencies(self):
        with patch('scripts.advance_diagonal_development.audit_development',return_value={'selected':{}}), \
             patch('scripts.advance_diagonal_development.prepare',return_value=self.selection), \
             patch('scripts.advance_diagonal_development.submit',side_effect=['11','12','13']) as submit:
            main()
        self.assertIn('--dependency=afterok:123',submit.call_args_list[0].args[0])
        for call in submit.call_args_list[1:]:self.assertIn('--dependency=afterok:11',call.args[0])
        record=self.ledger()
        self.assertEqual((record['confirmation_job_id'],record['confirmation_audit_job_id'],record['recommendation_smoke_job_id']),
                         ('11','12','13'))

    def test_partial_submission_failure_does_not_duplicate_confirmation(self):
        with patch('scripts.advance_diagonal_development.audit_development',return_value={'selected':{}}), \
             patch('scripts.advance_diagonal_development.prepare',return_value=self.selection), \
             patch('scripts.advance_diagonal_development.submit',side_effect=['11',RuntimeError('scheduler unavailable')]) as submit:
            with self.assertRaisesRegex(RuntimeError,'scheduler unavailable'):main()
            self.assertEqual(self.ledger()['confirmation_job_id'],'11')
            with self.assertRaisesRegex(ValueError,'do not duplicate'):main()
            self.assertEqual(submit.call_count,2)


if __name__=='__main__':unittest.main()
