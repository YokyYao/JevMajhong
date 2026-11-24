import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools import curriculum_retained_yaku_bridge_v1 as bridge
from tools.seal_curriculum_basic_repair_v2 import ParentNotReady
from tests.core.test_curriculum_yaku_parent_launch_v1 import parent_fixture


def fixture():
    p, done, evidence, execution, config = parent_fixture()
    cap = dict(status='parent_retention_actual_terminals_preserved_not_child_admitted',
        original_strategy_grade=copy.deepcopy(evidence['stage3_advancement']),
        strategy_grade_overridden=False, child_parent_sealed=False, student_updates=0,
        scheduler_proofs=dict(fit={'successful': True}, audit={'successful': True}))
    return p, done, evidence, execution, cap, config


def capture_raw(job='333', state='COMPLETED', gpu='1', command=None):
    return (f'JobId={job} JobState={state} ExitCode=0:0 NodeList=ws-test '
        f'AllocTRES=cpu=1,gres/gpu={gpu} StartTime=2026-11-24T20:00:00 EndTime=2026-11-24T20:00:10 '
        f'Command={command or bridge.CAPTURE_ENTRY}')


class RetainedYakuBridgeTests(unittest.TestCase):
    def test_complete_synthetic_parent_contract_can_pass_without_submitting(self):
        bridge.require_parent(*fixture())

    def test_failed_original_grade_or_phase_is_rejected(self):
        for field in ('stage3_advancement', 'basic_retention'):
            values = fixture(); values[2][field]['passed'] = False
            with self.subTest(field=field), self.assertRaises(ValueError): bridge.require_parent(*values)
        values = fixture(); values[4]['scheduler_proofs']['audit']['successful'] = False
        with self.assertRaises(ValueError): bridge.require_parent(*values)
        values = fixture(); values[4]['strategy_grade_overridden'] = True
        with self.assertRaises(ValueError): bridge.require_parent(*values)

    def test_only_real_complete_matching_gpu_capture_records_are_accepted(self):
        receipt = dict(job_id='333', exit_code=0)
        self.assertEqual(bridge.capture_record(capture_raw(), receipt, '333')['exit'], receipt)
        for raw in (capture_raw(job='444'), capture_raw(state='RUNNING'), capture_raw(gpu='0'),
                capture_raw(command=Path('/wrong/entry'))):
            with self.subTest(raw=raw), self.assertRaises((ValueError, ParentNotReady)):
                bridge.capture_record(raw, receipt, '333')
        with self.assertRaises(ParentNotReady): bridge.capture_record(capture_raw(), dict(receipt, exit_code=1), '333')

    def test_existing_child_queue_is_not_restarted_or_overwritten(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value); queue = root/'queue.json'; child = root/'child'
            self.assertTrue(bridge.child_submission_needed(queue, child, 'expected'))
            child.mkdir()
            with self.assertRaises(ValueError): bridge.child_submission_needed(queue, child, 'expected')
            payload = dict(status='actual_qualified_parent_yaku_jobs_bound_while_held',
                binding_sha256='expected', original_parent_grade_preserved=True)
            queue.write_text(json.dumps(payload))
            self.assertFalse(bridge.child_submission_needed(queue, child, 'expected'))
            with self.assertRaises(ValueError): bridge.child_submission_needed(queue, child, 'other')
            self.assertEqual(json.loads(queue.read_text()), payload)

    def test_actual_held_dependency_and_finite_gpu_resources_are_mandatory(self):
        raw = (f'JobState=PENDING Reason=JobHeldUser Command={bridge.ENTRY} Dependency=afterany:222(unfulfilled) '
            'TimeLimit=00:05:00 NumCPUs=1 ReqTRES=cpu=1,mem=4G,gres/gpu=1')
        bridge.require_held(raw, '222')
        for old, new in [('222', '999'), ('JobHeldUser', 'Priority'), ('00:05:00', '20:00:00'),
                         ('gres/gpu=1', 'gres/gpu=0'), ('NumCPUs=1', 'NumCPUs=4')]:
            with self.subTest(old=old), self.assertRaises(ValueError): bridge.require_held(raw.replace(old, new), '222')

    def test_new_preparation_depends_on_bridge_success_and_cancels_if_invalid(self):
        original = ['sbatch', '--hold', '--export=ALL,JEV_YAKU_PARENT_PHASE=prepare', 'entry']
        adapted = bridge.prepare_after_bridge(original, '444')
        self.assertIn('--dependency=afterok:444', adapted)
        self.assertIn('--kill-on-invalid-dep=yes', adapted)
        self.assertEqual(original, ['sbatch', '--hold', '--export=ALL,JEV_YAKU_PARENT_PHASE=prepare', 'entry'])
        for command in (['scancel', '555'], ['sbatch', '--dependency=afterok:555', '--export=ALL,JEV_YAKU_PARENT_PHASE=fit']):
            self.assertEqual(bridge.prepare_after_bridge(command, '444'), command)
        with self.assertRaises(ValueError): bridge.prepare_after_bridge(original, 'not-a-job')
        with self.assertRaises(ValueError): bridge.prepare_after_bridge(original+['--dependency=afterok:555'], '444')


if __name__ == '__main__': unittest.main()
