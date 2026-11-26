from collections import Counter
import random
import unittest

from tools import curriculum_ri_coverage_replay_v2 as previous
from tools.curriculum_yaku_ri_coverage_v1 import MASS, RI, build


class YakuRICoverageTest(unittest.TestCase):
    def fixture(self):
        categories = []
        for source in MASS:
            categories += [source] * (2 if source in RI else 13)
        baseline = list(range(len(categories)))
        random.Random(97).shuffle(baseline)
        return categories, baseline

    def test_full25_and_rare_RI_replay_preserve_every_original_objective_mass(self):
        categories, baseline = self.fixture()
        result = build(categories, baseline, 83)
        order = result['indices']; counts = Counter(categories); copies = Counter(order)
        self.assertEqual(set(copies), set(range(len(categories))))
        self.assertEqual(len(order) % 73, 0)
        for index, source in enumerate(categories):
            observed = copies[index]*result['weights_by_original_index'][str(index)]/len(order)
            self.assertAlmostEqual(observed, MASS[source]/counts[source], places=13)
        for start in range(0, len(order), 73):
            self.assertTrue(set(RI) <= {categories[i] for i in order[start:start+73]})

    def test_original_nonRI_sequence_and_old_sampler_globals_stay_unchanged(self):
        categories, baseline = self.fixture()
        before = dict(previous.MASS)
        result = build(categories, baseline, 83)
        self.assertEqual(result['nonRI_original_subsequence'],
            [index for index in baseline if categories[index] not in RI])
        self.assertEqual(previous.MASS, before)
        self.assertEqual(len(before), 19)
        self.assertEqual(len(MASS), 25)

    def test_missing_route_or_broken_full_permutation_is_rejected(self):
        categories, baseline = self.fixture()
        unknown = list(categories); unknown[0] = 'unregistered-source'
        with self.assertRaises(ValueError):
            build(unknown, baseline, 83)
        with self.assertRaises(ValueError):
            build(categories, baseline[:-1], 83)

    def test_original_repeat_and_weight_limits_remain_binding(self):
        categories, baseline = self.fixture()
        with self.assertRaises(ValueError):
            build(categories, baseline, 83, max_repeat=1)
        with self.assertRaises(ValueError):
            build(categories, baseline, 83, max_weight=0.01)


if __name__ == '__main__':
    unittest.main()
