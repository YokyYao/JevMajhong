"""Lifecycle preparation cannot relax numeric checks or substitute real exits."""
import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import launch_curriculum_placement_balanced_v1 as lifecycle
from tests.core.test_curriculum_placement_launch_v1 import proof


class BalancedPlacementLaunchTests(unittest.TestCase):
    def test_prepare_changes_only_explicit_job_producer_loss_and_status_connections(self):
        actual=lifecycle.prepare_tree()
        reverse={lifecycle.PREFIX:'curriculum-placement-preflight-exit-',
            str(lifecycle.ENTRY.relative_to(lifecycle.ROOT)):'cluster/curriculum-placement-preflight-v1.slurm',
            'placement_retained_objective_actual_upstream_integration_and_launch_bound_v1':
                'placement_actual_upstream_integration_and_launch_bound_v1'}
        class Restore(ast.NodeTransformer):
            def visit_Name(self,node):
                if node.id=='ACTUAL_INTEGRATION_JOB':return ast.copy_location(ast.Constant(value=319427),node)
                return node
            def visit_Constant(self,node):
                if isinstance(node.value,str) and node.value in reverse:node.value=reverse[node.value]
                return node
            def visit_ImportFrom(self,node):
                if node.module=='tools.curriculum_placement_parent_metrics_v2':
                    node.module='tools.curriculum_placement_metrics_v1'
                return node
        actual=Restore().visit(actual)
        self.assertEqual(ast.dump(actual),ast.dump(ast.parse(inspect.getsource(lifecycle.original.prepare))))

    def test_actual_job_and_explicit_parent_verifier_are_injected_without_gate_changes(self):
        for job in (222,333):
            function=lifecycle.prepare_function(job)
            self.assertEqual(function.__globals__['ACTUAL_INTEGRATION_JOB'],job)
            self.assertIs(function.__globals__['verify_parent'],lifecycle.course.parent.verify_parent)
            self.assertIs(function.__globals__['require_integration'],lifecycle.require_integration)
        with self.assertRaises(ValueError):lifecycle.prepare_function('not-a-job')

    def test_real_integration_requirements_and_new_loss_lineage_are_both_mandatory(self):
        reg=dict(parent_model=dict(identity_sha256='qualified-parent'))
        value=proof();value.update(loss_method=lifecycle.integration.loss.METHOD,
            placement_parent_lineage_method=lifecycle.course.parent.METHOD,global_student_stage_offset=5968)
        lifecycle.require_integration(value,reg,319427)
        for name,bad in [('optimizer_steps',1),('actual_control_record_exposures',2),
                ('parent_tensors_unchanged',False),('node','login-01'),('loss_method','old'),
                ('placement_parent_lineage_method','original-route'),('global_student_stage_offset',5896)]:
            with self.subTest(name=name),self.assertRaises(ValueError):
                lifecycle.require_integration(dict(value,**{name:bad}),reg,319427)

    def test_missing_actual_parent_cannot_submit_any_new_job(self):
        with patch.object(lifecycle,'verify_entry',return_value={}), \
                patch.object(lifecycle.course.parent,'parent_artifacts',side_effect=FileNotFoundError('no actual parent')), \
                patch.object(lifecycle.subprocess,'run') as submit:
            with self.assertRaises(FileNotFoundError):lifecycle.submit()
            submit.assert_not_called()

    def test_failed_actual_parent_grade_cannot_submit_any_new_job(self):
        with patch.object(lifecycle,'verify_entry',return_value={}), \
                patch.object(lifecycle.course.parent,'parent_artifacts',side_effect=ValueError('actual grade failed')), \
                patch.object(lifecycle.subprocess,'run') as submit:
            with self.assertRaises(ValueError):lifecycle.submit()
            submit.assert_not_called()

    def test_exit_receipt_must_belong_to_exact_job_and_phase(self):
        with patch.object(lifecycle,'read',return_value=dict(job_id='222',phase='fit',exit_code=0)):
            self.assertEqual(lifecycle.receipt('fit','222')['exit_code'],0)
            for phase,job in [('audit','222'),('fit','333')]:
                with self.assertRaises(ValueError):lifecycle.receipt(phase,job)

    def test_parent_terminal_job_requires_actual_running_yaku_fit_before_submission(self):
        value=dict(fit_job='222',capture_job='333')
        observed=SimpleNamespace(returncode=0,stdout='JobState=PENDING Command=other')
        with patch.object(lifecycle,'verify_entry',return_value={}), \
                patch.object(lifecycle,'read',return_value=value), \
                patch.object(lifecycle,'scheduler',return_value=observed), \
                patch.object(lifecycle.subprocess,'run') as submit:
            with self.assertRaises(ValueError):lifecycle.enqueue_parent_terminal()
            submit.assert_not_called()

    def test_missing_bound_entry_refuses_without_source_or_job_mutation(self):
        with patch.object(lifecycle,'resource',return_value='fixture-only'), \
                patch.object(lifecycle,'read',side_effect=FileNotFoundError('entry absent')), \
                patch.object(lifecycle.subprocess,'run') as submit:
            with self.assertRaises(FileNotFoundError):lifecycle.submit()
            submit.assert_not_called()


if __name__=='__main__':unittest.main()
