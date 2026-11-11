"""Inference timing must not affect probability parity or mask a changed action."""
import copy
import unittest
from scripts.collect_fast_win_v2 import same_distribution


class FastPreflightTest(unittest.TestCase):
    def test_timing_difference_allowed_but_probability_change_rejected(self):
        first = {'answers': {'action': {'probabilities': {'a0': .6, 'a1': .4}}}, 'latency_ms': 10.}
        second = copy.deepcopy(first); second['latency_ms'] = 20.
        self.assertTrue(same_distribution(first, second))
        second['answers']['action']['probabilities'] = {'a0': .4, 'a1': .6}
        self.assertFalse(same_distribution(first, second))
        second['answers']['action']['probabilities'] = {'a1': .4, 'a0': .6}
        self.assertFalse(same_distribution(first, second))


if __name__ == '__main__':
    unittest.main()
