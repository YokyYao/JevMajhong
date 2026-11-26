"""Canonical lifecycle rejects missing evidence and preserves original guards."""
import ast
import inspect
from pathlib import Path
from types import CodeType,SimpleNamespace
import unittest
from unittest.mock import patch

from tools import launch_curriculum_placement_canonical_v2 as lifecycle
from tests.core.test_curriculum_placement_launch_v1 import proof


class CanonicalPlacementLaunchTests(unittest.TestCase):
    def test_generic_lifecycle_function_code_objects_remain_unchanged(self):
        for name in ('prepare_tree','require_integration','prepare_function','verify_entry',
                'receipt','archive_success','verify_queue','enqueue_parent_terminal','submit'):
            function=getattr(lifecycle.previous,name)
            self.assertIs(lifecycle.scoped(function).__code__,function.__code__)

    def test_only_three_terminal_literal_sites_change(self):
        reverse={value:key for key,value in lifecycle.REPLACEMENTS.items()}
        def restore(code):
            return code.replace(co_consts=tuple(restore(value) if isinstance(value,CodeType)
                else reverse.get(value,value) if isinstance(value,str) else value for value in code.co_consts))
        for function in (lifecycle.previous.run,lifecycle.previous.preserve_parent_terminal):
            self.assertEqual(restore(lifecycle.scoped(function).__code__),function.__code__)

    def test_prepare_preserves_original_body_after_declared_route_substitutions(self):
        tree=lifecycle.prepare_tree()
        reverse={lifecycle.PREFIX:'curriculum-placement-preflight-exit-',
            str(lifecycle.ENTRY.relative_to(lifecycle.ROOT)):'cluster/curriculum-placement-preflight-v1.slurm',
            'placement_retained_objective_actual_upstream_integration_and_launch_bound_v1':
                'placement_actual_upstream_integration_and_launch_bound_v1'}
        class Restore(ast.NodeTransformer):
            def visit_Name(self,node):
                return ast.copy_location(ast.Constant(value=319427),node) if node.id=='ACTUAL_INTEGRATION_JOB' else node
            def visit_Constant(self,node):
                if isinstance(node.value,str) and node.value in reverse:node.value=reverse[node.value]
                return node
            def visit_ImportFrom(self,node):
                if node.module=='tools.curriculum_placement_parent_metrics_v2':node.module='tools.curriculum_placement_metrics_v1'
                return node
        self.assertEqual(ast.dump(Restore().visit(tree)),ast.dump(ast.parse(inspect.getsource(lifecycle.previous.original.prepare))))
        function=lifecycle.prepare_function(222)
        self.assertEqual(function.__globals__['ACTUAL_INTEGRATION_JOB'],222)
        self.assertIs(function.__globals__['verify_parent'],lifecycle.course.parent.verify_parent)
        self.assertIs(function.__globals__['require_registration'],lifecycle.course.require_registration)
        with self.assertRaises(ValueError):lifecycle.prepare_function('not-a-job')

    def test_real_optimizer_proof_and_canonical_lineage_are_both_required(self):
        reg=dict(parent_model=dict(identity_sha256='qualified-parent'))
        value=proof();value.update(loss_method=lifecycle.integration.loss.METHOD,
            placement_parent_lineage_method=lifecycle.course.parent.METHOD,global_student_stage_offset=5968)
        lifecycle.require_integration(value,reg,319427)
        for name,bad in [('optimizer_steps',1),('actual_control_record_exposures',2),
                ('parent_tensors_unchanged',False),('node','login-01'),('loss_method','old'),
                ('placement_parent_lineage_method',lifecycle.previous.course.parent.METHOD),
                ('global_student_stage_offset',5896)]:
            with self.subTest(name=name),self.assertRaises(ValueError):
                lifecycle.require_integration(dict(value,**{name:bad}),reg,319427)

    def test_missing_or_failed_actual_parent_submits_no_job(self):
        for error in (FileNotFoundError('no parent'),ValueError('parent grade failed')):
            with self.subTest(error=type(error).__name__), \
                    patch.object(lifecycle,'verify_entry',return_value={}), \
                    patch.object(lifecycle.course.parent,'parent_artifacts',side_effect=error), \
                    patch.object(lifecycle.subprocess,'run') as submit:
                with self.assertRaises(type(error)):lifecycle.submit()
                submit.assert_not_called()

    def test_actual_phase_receipts_cannot_be_borrowed_from_another_job(self):
        with patch.object(lifecycle,'read',return_value=dict(job_id='222',phase='fit',exit_code=0)):
            self.assertEqual(lifecycle.receipt('fit','222')['exit_code'],0)
            for phase,job in [('audit','222'),('fit','333')]:
                with self.assertRaises(ValueError):lifecycle.receipt(phase,job)

    def test_parent_terminal_enqueue_requires_the_actual_running_canonical_yaku_fit(self):
        observed=SimpleNamespace(returncode=0,stdout='JobState=PENDING Command=wrong')
        with patch.object(lifecycle,'verify_entry',return_value={}), \
                patch.object(lifecycle,'read',return_value=dict(fit_job='222',capture_job='333')), \
                patch.object(lifecycle,'scheduler',return_value=observed), \
                patch.object(lifecycle.subprocess,'run') as submit:
            with self.assertRaises(ValueError):lifecycle.enqueue_parent_terminal()
            submit.assert_not_called()

    def test_unbound_entry_cannot_mutate_sources_or_submit_jobs(self):
        with patch.object(lifecycle,'resource',return_value='fixture-only'), \
                patch.object(lifecycle,'read',side_effect=FileNotFoundError('entry absent')), \
                patch.object(lifecycle.subprocess,'run') as submit:
            with self.assertRaises(FileNotFoundError):lifecycle.submit()
            submit.assert_not_called()

    def test_finite_resources_original_data_and_numerical_modules_remain_bound(self):
        self.assertIs(lifecycle.LIMITS,lifecycle.previous.LIMITS)
        self.assertIs(lifecycle.DATA,lifecycle.previous.DATA)
        self.assertIs(lifecycle.ORIGINAL_DATA,lifecycle.previous.ORIGINAL_DATA)
        self.assertEqual(lifecycle.ENTRY.name,'curriculum-placement-canonical-course-v2.slurm')
        self.assertEqual(lifecycle.integration.OUT,lifecycle.course.INTEGRATION_OUT)
        self.assertEqual(lifecycle.integration.REGISTERED,lifecycle.course.REGISTERED)
        self.assertIs(lifecycle.previous.course,lifecycle.previous.submit.__globals__['course'])


if __name__=='__main__':unittest.main()
