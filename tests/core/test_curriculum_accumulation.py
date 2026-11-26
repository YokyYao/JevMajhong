import math
import unittest
from tools.audit_curriculum_accumulation import suggest_accumulation


class AccumulationPlanningTests(unittest.TestCase):
    def test_count_only_sizes_avoid_tiny_tail_and_keep_all_records(self):
        for records,maximum,rare in ((9292,64,194),(20237,128,194)):
            result=suggest_accumulation(records,maximum,rare)
            size=result['accumulation'];updates=result['updates_per_epoch']
            self.assertEqual((updates-1)*size+result['final_records'],records)
            self.assertGreaterEqual(result['final_fraction'],.75)
            self.assertLessEqual(updates,rare)
            for larger in range(size+1,maximum+1):
                count=math.ceil(records/larger);tail=records-(count-1)*larger
                self.assertTrue(tail/larger<.75 or count>rare)

    def test_no_duplicate_or_drop_fallback_when_rare_count_insufficient(self):
        with self.assertRaises(ValueError):suggest_accumulation(1000,64,2)
