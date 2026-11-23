import unittest

from tools.curriculum_progression_objective_v1 import MASS, COMPONENTS
from tools.curriculum_progression_fit_v1 import summarize


class ProgressionFitLoggingTests(unittest.TestCase):
    def test_an_update_can_contain_a_subset_while_source_counts_remain_exact(self):
        source = 'curriculum3:teacher:ordinary_completion'; weights = {s: 1. for s in MASS}; weights[source] = 2.
        rows = [dict(source=source, **{k: value if k in ('objective', 'standard_target_ce', 'ordinary_hard_ce') else 0.
            for k in COMPONENTS}) for value in (1., 3.)]
        result = summarize(rows, weights)
        self.assertEqual(result['source_counts'], {source: 2})
        self.assertEqual(result['loss'], 4.); self.assertEqual(result['ce'], 4.)
        self.assertEqual(result['extra_RI_class_loss'], 0.)

    def test_empty_nonfinite_and_unregistered_observations_cannot_make_a_curve(self):
        weights = {s: 1. for s in MASS}
        for rows in ([], [dict(source='unknown', **{k: 0. for k in COMPONENTS})],
                     [dict(source=next(iter(MASS)), **{k: float('nan') for k in COMPONENTS})]):
            with self.assertRaises(ValueError): summarize(rows, weights)
