"""Current route cannot borrow the cancelled canonical candidate or weaken final criteria."""
from pathlib import Path
import unittest
from unittest.mock import patch

from tools import curriculum_confirmation_early_v1 as consumer
from tests.core.test_curriculum_placement_early_course_v1 import emitted_fixture


def registration():
    value=consumer.course.registration.retained_registration(emitted_fixture())
    value['placement_parent_lineage_method']=consumer.course.parent.METHOD
    return value


class EarlyConfirmationTests(unittest.TestCase):
    def test_only_the_actual_current_placement_route_can_be_consumed(self):
        value=registration();consumer.require_route(consumer.course.REGISTERED,value)
        with self.assertRaises(ValueError):consumer.require_route(consumer.previous.course.REGISTERED,value)
        value['curriculum']['stage_offset']=5968
        with self.assertRaises(ValueError):consumer.require_route(consumer.course.REGISTERED,value)

    def test_original_numeric_source_and_statistical_verifier_is_reused(self):
        self.assertEqual(consumer.verify_function().__code__,consumer.previous.verify_function().__code__)
        function=consumer.verify_function()
        self.assertIs(function.__globals__['require_final_parent'],consumer.original.require_final_parent)
        self.assertIs(function.__globals__['require_bound_files'],consumer.original.require_bound_files)

    def test_native_run_and_independent_audit_change_only_verifier(self):
        for function,original in ((consumer.run_function(),consumer.runner.run),
                (consumer.audit_function(),consumer.auditor.audit)):
            self.assertIs(function.__code__,original.__code__)
            self.assertIs(function.__globals__['verify'],consumer.verify)

    def test_original_finish_accounting_changes_only_own_auditor_and_receipt_paths(self):
        code=consumer.finish_function().__code__
        names=tuple('tools.audit_curriculum_confirmation_v1' if name=='tools.curriculum_confirmation_early_v1'
            else name for name in code.co_names)
        reverse={'curriculum-confirmation-early-run-exit-':'curriculum-confirmation-run-exit-',
            'cluster/curriculum-confirmation-early-run-v1.slurm':'cluster/curriculum-confirmation-run-v1.slurm'}
        restored=code.replace(co_names=names,co_consts=tuple(reverse.get(v,v) if isinstance(v,str) else v
            for v in code.co_consts))
        self.assertEqual(restored,consumer.finisher.finish.__code__)

    def test_missing_seal_or_source_union_refuses_before_native_execution(self):
        with patch.object(consumer,'verify_function') as actual:
            with self.assertRaises(ValueError):consumer.verify({},consumer.REGISTERED)
        actual.assert_not_called()
        reg=dict(placement_registration=str(consumer.course.REGISTERED),files={})
        with patch.object(consumer.original,'require_header'),patch.object(consumer.previous,'read',return_value=registration()), \
                patch.object(consumer,'verify_function') as actual:
            with self.assertRaises(ValueError):consumer.verify(reg,consumer.REGISTERED)
        actual.assert_not_called()

    def test_complete_bound_union_reaches_the_original_verifier_without_admission_from_fixtures(self):
        paths=(*consumer.PUBLIC,*consumer.course.PUBLIC,*consumer.course.parent.PUBLIC)
        reg=dict(placement_registration=str(consumer.course.REGISTERED),
            files={str((consumer.ROOT/name).resolve()):consumer.sha(consumer.ROOT/name) for name in paths})
        with patch.object(consumer.original,'require_header'),patch.object(consumer.previous,'read',return_value=registration()), \
                patch.object(consumer,'verify_function') as actual:
            actual.return_value.return_value='fixture-only'
            self.assertEqual(consumer.verify(reg,consumer.REGISTERED),'fixture-only')
        actual.return_value.assert_called_once_with(reg,consumer.REGISTERED)


if __name__=='__main__':unittest.main()
