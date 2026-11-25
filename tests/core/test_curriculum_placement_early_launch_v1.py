"""The finite launch points to the real active lineage and preserves all data."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

from tools import launch_curriculum_placement_early_v1 as launch


class EarlyPlacementLaunchTests(unittest.TestCase):
    def test_actual_course_paths_offsets_and_finite_roles_are_explicit(self):
        self.assertEqual(launch.course.OFFSET,5296)
        self.assertEqual(launch.ROLES,('register','integration','prepare','fit','audit'))
        self.assertEqual(launch.LIMITS,launch.previous.LIMITS)
        self.assertEqual(launch.integration.REGISTERED,launch.course.REGISTERED)
        self.assertEqual(launch.DATA,launch.previous.DATA)
        self.assertEqual(launch.ORIGINAL_DATA,launch.previous.ORIGINAL_DATA)

    def test_launch_bodies_only_change_declared_metadata_not_training(self):
        for f in (launch.previous.submit,launch.previous.verify_queue,launch.previous.prepare_function):
            self.assertIs(launch.scoped(f).__code__,f.__code__)
        with self.assertRaises(ValueError):launch.scoped(lambda:None)
        adapted=launch.scoped(launch.previous.require_integration)
        self.assertIn(5296,adapted.__code__.co_consts);self.assertNotIn(5968,adapted.__code__.co_consts)
        run=launch.scoped(launch.previous.run)
        self.assertIn('placement-early-terminal-v1',run.__code__.co_consts)

    def test_bound_preparation_keeps_real_job_and_existing_numeric_proof(self):
        tree=launch.prepare_tree();compile(tree,'fixture-only','exec')
        values=[node.value for node in ast.walk(tree) if isinstance(node,ast.Constant)]
        self.assertIn(launch.PREFIX,values)
        self.assertIn(str(launch.ENTRY.relative_to(launch.ROOT)),values)
        with self.assertRaises(ValueError):launch.prepare_function('not-a-job')

    def test_missing_parent_refuses_submission_before_any_sbatch(self):
        with patch.object(launch,'verify_entry'), \
                patch.object(launch,'QUEUE',Path('/not-an-actual-existing-launch-queue')), \
                patch.object(launch.course.parent,'parent_artifacts',side_effect=FileNotFoundError), \
                patch.object(launch.subprocess,'run') as submit:
            with self.assertRaises(FileNotFoundError):launch.submit()
        submit.assert_not_called()


if __name__=='__main__':unittest.main()
