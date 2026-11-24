import copy
import os
import unittest
from tools import capture_curriculum_targeted_shard_terminal_v3 as capture


class ShardTerminalTests(unittest.TestCase):
    def pending(self,d):
        return dict(JobId=d['job'],Command=d['command'],JobState='PENDING',RunTime='00:00:00',
            AllocTRES='(null)',StartTime='Unknown',UserId=f'fixture({os.getuid()})',
            ArrayJobId=d['array_job'] or 'unused',ArrayTaskId=str(d['shard']))

    def test_original_and_v3_descriptors_do_not_share_paths(self):
        a=capture.descriptor('original',1,322031);b=capture.descriptor('v3',1,999,1)
        self.assertNotEqual(a['folder'],b['folder']);self.assertNotEqual(a['archive'],b['archive'])
        self.assertTrue(a['folder'].endswith('shards/shard-01'))
        self.assertTrue(b['folder'].endswith('shard-01/attempt-001'))

    def test_only_actual_own_unallocated_pending_writer_can_be_bound(self):
        d=capture.descriptor('original',1,322031);v=self.pending(d);capture.require_pending(v,d)
        for key,value in [('JobState','RUNNING'),('RunTime','00:01:00'),('AllocTRES','gres/gpu=1'),
                ('StartTime','2026-11-25T00:00:00'),('UserId','other(0)'),('JobId','322030'),
                ('ArrayTaskId','0'),('ArrayJobId','111'),('Command','different')]:
            with self.assertRaises(ValueError):capture.require_pending(dict(v,**{key:value}),d)

    def test_v3_retains_its_actual_command_and_job_identity(self):
        d=capture.descriptor('v3',1,999,1);v=self.pending(d);capture.require_pending(v,d)
        with self.assertRaises(ValueError):capture.require_pending(dict(v,Command='original-entry'),d)

    def test_invalid_kind_shard_attempt_or_job_refused(self):
        for args in [('other',1,99,0),('original',4,99,0),('original',1,99,1),
                ('v3',1,99,0),('original',1,'not-a-job',0)]:
            with self.assertRaises(ValueError):capture.descriptor(*args)

    def test_failed_scheduler_state_cannot_become_success_from_zero_exit_code(self):
        d=capture.descriptor('original',1,999)
        for state in ('TIMEOUT','CANCELLED','FAILED','NODE_FAIL'):
            raw=f"JobId=999 Command={d['command']} JobState={state} ExitCode=0:0 EndTime=2026-11-25T00:00:00"
            proof=capture.terminal_record(raw,'999',capture.Path(d['command']))
            self.assertFalse(proof['successful'])

    def test_replacement_array_master_is_explicit_and_bound(self):
        d=capture.descriptor('original',1,999,array_job=888);v=self.pending(d)
        capture.require_pending(v,d)
        with self.assertRaises(ValueError):capture.require_pending(dict(v,ArrayJobId='319759'),d)
        with self.assertRaises(ValueError):capture.descriptor('v3',1,999,1,array_job=888)

    def test_actual_single_element_compressed_array_format(self):
        d=capture.descriptor('original',1,999);v=self.pending(d)
        capture.require_pending(dict(v,ArrayTaskId='1%1'),d)
        for value in ('1-2%1','1,2%1','2%1','1%1-extra',None):
            with self.assertRaises(ValueError):capture.require_pending(dict(v,ArrayTaskId=value),d)


if __name__=='__main__':unittest.main()
