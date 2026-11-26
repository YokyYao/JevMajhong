"""Software connection checks do not imply two actual student updates."""
from pathlib import Path
import unittest
from unittest.mock import patch

from tools import integrate_curriculum_placement_balanced_v1 as integration


class BalancedPlacementIntegrationTests(unittest.TestCase):
    def test_complete_control_body_is_the_original_code_object(self):
        self.assertIs(integration.integration_function().__code__,integration.original.preflight.__code__)

    def test_annotation_loss_and_summarizer_match_the_production_trainer(self):
        control=integration.integration_function().__globals__
        trainer=integration.course.train_function().__globals__
        for name in ('annotate','weighted_loss','summarize'):
            self.assertIs(control[name],trainer[name])
        self.assertIs(control['require_registration'],integration.course.require_registration)
        self.assertIs(control['verify_parent'],integration.course.parent.verify_parent)

    def test_only_the_actual_route_and_isolated_output_can_be_used(self):
        integration.require_paths(integration.REGISTERED,integration.OUT)
        for registered,out in ((Path('/old-registration'),integration.OUT),
                (integration.REGISTERED,integration.ROOT/integration.course.IDENTITY['out'])):
            with self.assertRaises(ValueError):integration.require_paths(registered,out)

    def test_emitted_integration_proof_names_actual_loss_and_lineage(self):
        value=dict(status='fixture-only-not-an-actual-control',optimizer_steps=2)
        with patch.object(integration.original,'write') as write:
            integration.integration_function().__globals__['write'](Path('/fixture/integration-proof-private.json'),value)
            result=write.call_args.args[1]
        self.assertNotIn('loss_method',value)
        self.assertEqual(result['loss_method'],integration.loss.METHOD)
        self.assertEqual(result['placement_parent_lineage_method'],integration.course.parent.METHOD)
        self.assertEqual(result['global_student_stage_offset'],5968)
        self.assertEqual(result['status'],'fixture-only-not-an-actual-control')

    def test_all_software_sources_remain_bound_in_real_integration(self):
        public=integration.integration_function().__globals__['PUBLIC_FILES']
        for name in (*integration.PUBLIC,*integration.course.PUBLIC,*integration.course.parent.PUBLIC):
            self.assertIn(name,public)

    def test_missing_entry_refuses_before_loading_or_updating_a_model(self):
        with patch.object(integration,'resource',return_value='fixture-only'), \
                patch.object(integration,'read',side_effect=FileNotFoundError('entry absent')), \
                patch.object(integration,'integration_function') as function:
            with self.assertRaises(FileNotFoundError):integration.integrate()
            function.assert_not_called()


if __name__=='__main__':unittest.main()
