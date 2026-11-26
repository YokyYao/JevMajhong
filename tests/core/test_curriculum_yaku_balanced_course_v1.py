import ast
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
from tools import launch_curriculum_yaku_balanced_v1 as launch
from tools.curriculum_transfer import read

ROOT=Path(__file__).resolve().parents[2]


def function_ast(path,name,adapt_parent=False):
    module=ast.parse((ROOT/'tools'/path).read_text())
    function=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name==name)
    class ParentOnly(ast.NodeTransformer):
        def visit_Constant(self,node):
            if node.value==2400:node.value=2472
            elif isinstance(node.value,str):node.value=node.value.replace('step-002400/','step-002472/')
            return node
    if adapt_parent:function=ParentOnly().visit(function)
    return ast.dump(function,include_attributes=False)


class BalancedCourseTests(unittest.TestCase):
    def test_registration_calculation_only_changes_actual_selected_parent_step(self):
        self.assertEqual(function_ast('register_curriculum_yaku_parent_v1.py','register',True),
            function_ast('register_curriculum_yaku_balanced_v1.py','register'))

    def test_training_and_warmstart_only_change_actual_parent_prediction_path(self):
        for name in ('train','write_conditional_config'):
            self.assertEqual(function_ast('train_curriculum_yaku_parent_v1.py',name,True),
                function_ast('train_curriculum_yaku_balanced_v1.py',name))

    def test_full_audit_and_conditional_grades_preserve_all_original_calculations(self):
        for name in ('audit','audit_conditional'):
            self.assertEqual(function_ast('audit_curriculum_yaku_parent_v1.py',name,True),
                function_ast('audit_curriculum_yaku_balanced_v1.py',name))

    def test_actual_config_only_changes_parent_output_and_global_offset(self):
        original=read(ROOT/'configs/training/curriculum-yaku-parent-fit-v1.json')
        candidate=read(launch.CONFIG);launch.require_same_recipe(candidate,original)
        for key,value in [('final_optimizer_steps',2472),('stage_offset',3396),
                ('parent_selection','actual_registered_final2400_only'),('learning_rate',1e-5)]:
            bad=copy.deepcopy(candidate);bad[key]=value
            with self.assertRaises(ValueError):launch.require_same_recipe(bad,original)
        bad=copy.deepcopy(candidate);bad['development_targets']['max_RI_balanced_accuracy_drop_from_step0']=.1
        with self.assertRaises(ValueError):launch.require_same_recipe(bad,original)

    def test_failed_parent_qualification_never_reaches_scheduler_submission(self):
        with patch.object(launch,'resource'),patch.object(launch,'verify_binding'), \
                patch.object(launch,'qualified_parent',side_effect=ValueError('actual parent not qualified')), \
                patch.object(launch.subprocess,'run') as submit:
            with self.assertRaisesRegex(ValueError,'actual parent not qualified'):launch.submit()
            submit.assert_not_called()

    def test_capture_archive_rejects_wrong_phase_before_any_scheduler_query(self):
        with patch.object(launch,'read',return_value={'phase':'fit'}),patch.object(launch,'scheduler') as query:
            with self.assertRaisesRegex(ValueError,'capture phase'):
                launch.archive_parent_capture({'parent_terminal_capture_job':123})
            query.assert_not_called()


if __name__=='__main__':unittest.main()
