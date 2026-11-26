import copy
import inspect
from types import CodeType
import unittest

from tools import curriculum_yaku_best3a_v1 as course


class Best3ACourseTests(unittest.TestCase):
    def values(self):
        return {618:dict(matches=16,top2_rate=.4375,net_points=-1256.25),
            1236:dict(matches=16,top2_rate=.6875,net_points=2400.),
            1854:dict(matches=16,top2_rate=.625,net_points=4000.),
            2472:dict(matches=16,top2_rate=.6875,net_points=2000.)}

    def test_choose_same_grid_top2_then_points_without_CE_or_v8_gate(self):
        values=self.values();values[618]['validation_loss']=.001
        self.assertEqual(course.choose(values),1236)
        values[2472]['net_points']=2500.;self.assertEqual(course.choose(values),2472)
        values[2472]['net_points']=2400.;self.assertEqual(course.choose(values),1236)

    def test_no_missing_different_grid_or_nonfinite_candidate(self):
        for change in ('missing','more_games','nan','step0'):
            values=self.values()
            if change=='missing':values.pop(1854)
            elif change=='more_games':values[2472]['matches']=64
            elif change=='nan':values[1854]['net_points']=float('nan')
            else:values[0]=dict(matches=16,top2_rate=1.,net_points=10000.)
            with self.subTest(change=change),self.assertRaises(ValueError):course.choose(values)

    def test_selected_step_reaches_registration_and_real_parent_prediction_path(self):
        registered=course.function(course.old_register.register,1236,
            PARENT_METHOD=course.METHOD,STATUS=course.STATUS,verify_parent=course.verify_parent)
        self.assertIn(1236,registered.__code__.co_consts);self.assertNotIn(2472,registered.__code__.co_consts)
        train=course.function(course.old_train.train,1236,verify_parent=course.verify_parent)
        def constants(code):
            return [value for value in code.co_consts if isinstance(value,str)]+[
                value for child in code.co_consts if isinstance(child,CodeType) for value in constants(child)]
        self.assertTrue(any('step-001236/development/rows.json' in value for value in constants(train.__code__)))
        self.assertFalse(any('step-002472/development/rows.json' in value for value in constants(train.__code__)))
        self.assertNotEqual(course.old_register.register.__globals__['PARENT_METHOD'],course.METHOD)

    def test_new_stage_admission_does_not_claim_old_grade_passed(self):
        source=inspect.getsource(course.select_and_seal)
        self.assertNotIn("['stage3_advancement']['passed']",source)
        self.assertNotIn("['basic_retention']['passed']",source)
        self.assertIn('stage3A_grade_blocks_3B=False',source)
        self.assertIn('raw_grade_overridden=False',source)
        self.assertIn('user_authorized_best_checkpoint_direct3B=True',source)


if __name__=='__main__':unittest.main()
