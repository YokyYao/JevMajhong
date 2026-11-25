"""Final consumer compatibility cannot weaken actual admission or test scope."""
from pathlib import Path
from types import CodeType
import unittest
from unittest.mock import patch

from tools import curriculum_confirmation_canonical_v2 as consumer
from tools import curriculum_placement_runtime_v1 as legacy
from tests.core.test_curriculum_placement_canonical_course_v2 import emitted_fixture


def registration():
    value=consumer.course.registration.retained_registration(emitted_fixture())
    value['placement_parent_lineage_method']=consumer.course.parent.METHOD
    return value


class CanonicalConfirmationTests(unittest.TestCase):
    def test_actual_legacy_import_mismatch_has_a_retained_loss_registration_counterexample(self):
        value=registration();consumer.require_route(consumer.course.REGISTERED,value)
        with self.assertRaises(ValueError):legacy.require_registration(value)

    def test_verifier_changes_one_import_name_and_preserves_every_other_code_field(self):
        function=consumer.verify_function();original=consumer.original.verify
        names=tuple('tools.curriculum_placement_runtime_v1' if name=='tools.curriculum_placement_parent_runtime_v1'
            else name for name in function.__code__.co_names)
        self.assertEqual(function.__code__.replace(co_names=names),original.__code__)
        self.assertIs(function.__globals__['require_final_parent'],original.__globals__['require_final_parent'])
        self.assertIs(function.__globals__['require_bound_files'],original.__globals__['require_bound_files'])

    def test_borrowed_lineage_registration_or_changed_baseline_cannot_enter(self):
        for mutation in ('old_path','old_lineage','baseline'):
            value=registration();path=consumer.course.REGISTERED
            if mutation=='old_path':path=consumer.course.balanced.ROOT/consumer.course.balanced.IDENTITY['plan_root']/'registered.json'
            elif mutation=='old_lineage':value['placement_parent_lineage_method']=consumer.course.balanced.parent.METHOD
            else:value['curriculum']['new_policy_zero64_actual_games_required']=False
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):consumer.require_route(path,value)

    def test_native_runner_and_independent_auditor_are_original_code_objects(self):
        for function,original in ((consumer.run_function(),consumer.runner.run),
                (consumer.audit_function(),consumer.auditor.audit)):
            self.assertIs(function.__code__,original.__code__)
            self.assertIs(function.__globals__['verify'],consumer.verify)

    def test_finish_changes_only_explicit_receipt_command_and_auditor_import(self):
        code=consumer.finish_function().__code__
        names=tuple('tools.audit_curriculum_confirmation_v1' if name=='tools.curriculum_confirmation_canonical_v2'
            else name for name in code.co_names)
        reverse={'curriculum-confirmation-canonical-run-exit-':'curriculum-confirmation-run-exit-',
            'cluster/curriculum-confirmation-canonical-run-v2.slurm':'cluster/curriculum-confirmation-run-v1.slurm'}
        restored=code.replace(co_names=names,co_consts=tuple(reverse.get(value,value)
            if isinstance(value,str) else value for value in code.co_consts))
        self.assertEqual(restored,consumer.finisher.finish.__code__)

    def test_unsealed_plan_refuses_before_model_loading_or_native_games(self):
        with patch.object(consumer,'verify_function') as verify:
            with self.assertRaises(ValueError):consumer.verify({},consumer.REGISTERED)
            verify.assert_not_called()

    def test_canonical_sources_are_mandatory_before_original_verification(self):
        reg=dict(placement_registration=str(consumer.course.REGISTERED),files={})
        with patch.object(consumer.original,'require_header'), \
                patch.object(consumer,'read',return_value=registration()), \
                patch.object(consumer,'verify_function') as verify:
            with self.assertRaises(ValueError):consumer.verify(reg,consumer.REGISTERED)
            verify.assert_not_called()
            with self.assertRaises(ValueError):consumer.verify(reg,Path('/old-final-registration'))
            verify.assert_not_called()

    def test_complete_required_source_union_reaches_unchanged_original_verifier(self):
        paths=(*consumer.PUBLIC,*consumer.course.PUBLIC,*consumer.course.parent.PUBLIC)
        reg=dict(placement_registration=str(consumer.course.REGISTERED),
            files={str((consumer.ROOT/name).resolve()):consumer.sha(consumer.ROOT/name) for name in paths})
        with patch.object(consumer.original,'require_header'), \
                patch.object(consumer,'read',return_value=registration()), \
                patch.object(consumer,'verify_function') as verify:
            verify.return_value.return_value='fixture-only-original-verifier-not-executed'
            self.assertEqual(consumer.verify(reg,consumer.REGISTERED),'fixture-only-original-verifier-not-executed')
            verify.return_value.assert_called_once_with(reg,consumer.REGISTERED)


if __name__=='__main__':unittest.main()
