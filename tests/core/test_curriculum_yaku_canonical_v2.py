import copy
from types import CodeType,SimpleNamespace
import unittest
from unittest.mock import patch
from tools import curriculum_yaku_canonical_v2 as course
from tests.core import test_curriculum_yaku_balanced_parent_v1 as fixtures


def shape(code,reverse=False):
    def value(v):
        if isinstance(v,CodeType):return shape(v,reverse)
        if reverse and isinstance(v,str):
            for old,new in course.STRINGS.items():v=v.replace(new,old)
        return v
    names=tuple(next((old for old,new in course.IMPORTS.items() if name==new),name)
        if reverse else name for name in code.co_names)
    return code.co_code,names,code.co_varnames,code.co_freevars,[value(v) for v in code.co_consts]


class CanonicalYakuTests(unittest.TestCase):
    def test_numeric_register_train_audit_code_objects_are_unchanged(self):
        for actual,original in ((course.register,course.old_registration.register),
            (course.train,course.old_training.train),(course.audit,course.old_auditing.audit)):
            self.assertEqual(shape(actual.__code__),shape(original.__code__))

    def test_parent_checks_change_only_declared_paths_and_identity_globals(self):
        for name in ('require_qualified_parent','actual_terminals','seal','verify_parent'):
            self.assertEqual(shape(getattr(course,name).__code__,True),
                shape(getattr(course.old_parent,name).__code__))
        self.assertEqual(course.verify_parent.__globals__['METHOD'],course.METHOD)
        self.assertEqual(course.verify_parent.__globals__['STATUS'],course.STATUS)
        self.assertIs(course.require_qualified_parent.__globals__['require_parent_registration'],
            course.canonical.require_registration)

    def test_lifecycle_redirects_only_declared_imports_and_paths(self):
        for name,function in course._lifecycle.items():
            # Several old lifecycle imports intentionally redirect to this one
            # adapter module; validate the forward map to avoid reverse ambiguity.
            original=getattr(course.old_launch,name)
            expected=course.adapt_code(original.__code__)
            self.assertEqual(shape(function.__code__),shape(expected))
            self.assertEqual(function.__code__.co_code,original.__code__.co_code)
        self.assertIn('tools.curriculum_yaku_canonical_v2',course._lifecycle['prepare'].__code__.co_names)
        self.assertIn('tools.curriculum_ri_balanced_replay_v2',course._lifecycle['bind'].__code__.co_names)

    def test_only_explicit_config_identity_and_parent_changes_allowed(self):
        c=course.read(course.CONFIG);old=course.read(course.ROOT/'configs/training/curriculum-yaku-parent-fit-v1.json')
        course.require_same_recipe(c,old)
        for key,value in (('final_optimizer_steps',100),('stage_offset',3396),('learning_rate',1e-4),
            ('parent_registration','runs/curriculum-ri-balanced-replay-fit-v1/fit-plan/registered.json')):
            bad=copy.deepcopy(c);bad[key]=value
            with self.assertRaises(ValueError):course.require_same_recipe(bad,old)

    def test_actual_binding_uses_the_canonical_recipe_checker(self):
        checker=course._lifecycle['bind'].__globals__['require_same_recipe']
        self.assertIs(checker,course.require_same_recipe)
        checker(course.read(course.CONFIG),course.read(course.ROOT/'configs/training/curriculum-yaku-parent-fit-v1.json'))
        self.assertIs(course._lifecycle['submit'].__globals__['require_qualified_parent'],course.require_qualified_parent)
        self.assertIs(course._lifecycle['prepare'].__globals__['seal'],course.seal)

    def test_failed_partial_or_unretained_parent_cannot_pass(self):
        fixture=fixtures.BalancedParentTests().fixture
        with patch.dict(course.require_qualified_parent.__globals__,{'require_parent_registration':lambda reg:None}):
            course.require_qualified_parent(*fixture())
            for target in ('stage3_advancement','basic_retention'):
                values=fixture();values[2][target]['passed']=False
                with self.assertRaises(ValueError):course.require_qualified_parent(*values)
            values=fixture();values[1]['optimizer_steps']=618
            with self.assertRaises(ValueError):course.require_qualified_parent(*values)
            values=fixture();values[2]['saved_final_reloaded_menus']=16
            with self.assertRaises(ValueError):course.require_qualified_parent(*values)

    def test_loader_restores_the_original_verifier_after_failure(self):
        previous=object();initialization=SimpleNamespace(verified_parent=previous)
        with self.assertRaises(RuntimeError):
            with course.parent_loader(initialization):
                self.assertIs(initialization.verified_parent,course.verify_parent)
                raise RuntimeError('stop')
        self.assertIs(initialization.verified_parent,previous)


if __name__=='__main__':unittest.main()
