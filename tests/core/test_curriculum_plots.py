import unittest
from tools.plot_curriculum_fixed_grid import aligned_record

class PlotGridTests(unittest.TestCase):
    def test_final64_does_not_replace_matching16_curve_point(self):
        row=dict(matches=64,seed_groups=8,top2_rate=.75,reward=1.,local_step=1162,
            curve_grid_subset=dict(matches=16,seed_groups=2,top2_rate=.5,reward=.25))
        actual=aligned_record(row)
        self.assertEqual(actual['top2_rate'],.5);self.assertEqual(actual['matches'],16)
        self.assertEqual(row['top2_rate'],.75);self.assertEqual(row['matches'],64)
        self.assertEqual(actual['local_step'],1162)

    def test_unmatched_sample_sizes_are_rejected(self):
        with self.assertRaises(ValueError):aligned_record(dict(matches=32,seed_groups=4))
        with self.assertRaises(ValueError):aligned_record(dict(matches=64,seed_groups=8))
