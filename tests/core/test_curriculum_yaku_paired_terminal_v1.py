import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import capture_curriculum_yaku_paired_terminal_v1 as capture


class YakuTerminalCaptureTests(unittest.TestCase):
    def test_bind_checks_actual_pending_chain_and_preparation_has_not_started(self):
        for defect in (None, 'wrong_dependency', 'already_prepared'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); queue_path = root/'queue.json'; binding = root/'binding.json'
                run = root/'not-yet-created'
                if defect == 'already_prepared': run.mkdir()
                queue_path.write_text(json.dumps(dict(actual_parent_terminal_capture_job='120',
                    prepare_job='122', fit_job='123', audit_job='124')))

                def observed(job):
                    previous = {'122': '120', '123': '122', '124': '123'}[job]
                    if defect == 'wrong_dependency' and job == '123': previous = '999'
                    return SimpleNamespace(returncode=0, stdout=f'JobId={job} JobState=PENDING '
                        f'Command={capture.ENTRY} Dependency=afterok:{previous}(unfulfilled)')

                with patch.multiple(capture, QUEUE=queue_path, BINDING=binding, RUN=run), \
                        patch.object(capture, 'resource', return_value='synthetic-allocation'), \
                        patch.object(capture, 'scheduler', side_effect=observed), \
                        patch.object(capture, 'fixed_paths', return_value=[]):
                    if defect:
                        with self.assertRaises(ValueError): capture.bind()
                        self.assertFalse(binding.exists())
                    else:
                        capture.bind()
                        self.assertEqual(json.loads(binding.read_text())['status'], capture.STATUS)
                        with self.assertRaises(FileExistsError): capture.bind()

    def exercise(self, *, fit_expired=False, audit_expired=False, failed=False,
                 wrong_phase=False, archive_mismatch=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); parent = root/'parent'; parent.mkdir()
            run = root/'run'; run.mkdir(); (root/'logs').mkdir()
            binding = parent/'binding.json'
            binding.write_text(json.dumps(dict(status=capture.STATUS, fit_job='123',
                audit_job='124', files={}, automatic_fit=False, strategy_grade_overridden=False)))
            evidence = dict(stage3B_advancement=dict(passed=False), sentinel='original-negative-grade')
            (run/'final-evidence-private.json').write_text(json.dumps(evidence))

            def row(job):
                return (f'JobId={job} JobState={"FAILED" if failed else "COMPLETED"} '
                    f'ExitCode={"1:0" if failed else "0:0"} NodeList=ws-synthetic '
                    'AllocTRES=cpu=4,gres/gpu=1 StartTime=2026-11-24T04:00:00 '
                    f'EndTime=2026-11-24T05:00:00 Command={capture.ENTRY}')

            for job, role in (('123', 'fit'), ('124', 'audit')):
                receipt = dict(job_id=job, phase='other' if wrong_phase else role,
                    exit_code=1 if failed else 0)
                (root/'logs'/f'curriculum-yaku-paired-exit-{job}.json').write_text(json.dumps(receipt))
                if role == 'fit':
                    archived_receipt = dict(receipt)
                    if archive_mismatch: archived_receipt['exit_code'] = 9
                    (run/'fit-terminal-scheduler-private.json').write_text(json.dumps(
                        dict(raw=row(job), exit=archived_receipt)))

            def observed(job):
                expired = (job == '123' and fit_expired) or (job == '124' and audit_expired)
                return SimpleNamespace(stdout='' if expired else row(job),
                    stderr='expired' if expired else '', returncode=1 if expired else 0)

            with patch.multiple(capture, ROOT=root, RUN=run, PARENT=parent, BINDING=binding), \
                    patch.object(capture, 'resource', return_value='synthetic-allocation'), \
                    patch.object(capture, 'scheduler', side_effect=observed):
                capture.capture()
                result = json.loads((parent/'yaku-terminal-capture-v1/terminal-evidence-private.json').read_text())
                self.assertEqual(result['original_final_evidence'], evidence)
                self.assertFalse(result['child_parent_sealed'])
                self.assertEqual(result['student_updates'], 0)
                self.assertFalse(result['strategy_grade_overridden'])
                self.assertEqual(set(result['scheduler_proofs']), {'fit', 'audit'})
                with self.assertRaises(FileExistsError): capture.capture()
            return result

    def test_two_successful_terminals_preserve_failed_grade_and_never_admit_child(self):
        result = self.exercise()
        self.assertTrue(result['both_scheduler_jobs_successful'])

    def test_failed_scheduler_results_are_evidence_not_success(self):
        self.assertFalse(self.exercise(failed=True)['both_scheduler_jobs_successful'])

    def test_only_exact_original_fit_archive_can_replace_expired_fit_observation(self):
        result = self.exercise(fit_expired=True)
        self.assertEqual(result['scheduler_proofs']['fit']['source'],
            'original_independent_audit_fit_archive')

    def test_expired_audit_is_not_synthesized(self):
        with self.assertRaises(ValueError): self.exercise(audit_expired=True)

    def test_successful_wrong_phase_receipt_is_rejected(self):
        with self.assertRaises(ValueError): self.exercise(wrong_phase=True)

    def test_mismatched_original_fit_archive_is_rejected(self):
        with self.assertRaises(ValueError): self.exercise(fit_expired=True, archive_mismatch=True)
