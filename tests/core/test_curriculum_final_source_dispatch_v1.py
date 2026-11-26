import os
from types import SimpleNamespace
import unittest
from tools import launch_curriculum_final_source_proof_v1 as dispatch


class DispatchTests(unittest.TestCase):
    def test_actual_role_ids_and_no_aliases(self):
        roles = ('register','integration','prepare','fit','audit','capture')
        queue = {role+'_job':str(i+100) for i,role in enumerate(roles)}
        self.assertEqual(dispatch.actual_ids(queue)['fit'], '103')
        queue['audit_job'] = queue['fit_job']
        with self.assertRaises(ValueError): dispatch.actual_ids(queue)
        queue['audit_job'] = 'future-audit'
        with self.assertRaises(ValueError): dispatch.actual_ids(queue)

    def test_actual_held_job_requires_every_dependency(self):
        raw = (f'JobId=123 JobState=PENDING Reason=JobHeldUser UserId=student({os.getuid()}) '
            f'Command={dispatch.PROOF_ENTRY} Dependency=afterok:111(unfulfilled),afterok:112(unfulfilled)')
        result = SimpleNamespace(stdout=raw, returncode=0)
        dispatch.require_held(result,'123',dispatch.PROOF_ENTRY,[('afterok','111'),('afterok','112')])
        with self.assertRaises(ValueError):
            dispatch.require_held(result,'123',dispatch.PROOF_ENTRY,[('afterok','113')])

    def test_running_or_wrong_command_is_not_held(self):
        raw = (f'JobId=123 JobState=PENDING Reason=JobHeldUser UserId=student({os.getuid()}) '
            f'Command={dispatch.PROOF_ENTRY} Dependency=afterok:111(unfulfilled)')
        for changed in (raw.replace('JobState=PENDING','JobState=RUNNING'),
                raw.replace(str(dispatch.PROOF_ENTRY),'/tmp/unrelated.slurm')):
            with self.assertRaises(ValueError):
                dispatch.require_held(SimpleNamespace(stdout=changed,returncode=0),
                    '123',dispatch.PROOF_ENTRY,[('afterok','111')])


if __name__ == '__main__': unittest.main()
