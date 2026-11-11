import math
import unittest
from call_sampling_prior import CLASSES, corrected_probabilities, inverse_inclusion


def options():
    return {'c': {'type': 'chi', 'call_features': {'loses_closed_status': True}},
            'n': {'type': 'none'}}


class SamplingPriorTests(unittest.TestCase):
    def test_recovers_toy_population_prior(self):
        source = {k: 100 for k in CLASSES};source['none'] = 900
        sample = {k: 100 for k in CLASSES}
        weights = inverse_inclusion(source, sample)
        corrected = corrected_probabilities(options(), {'c': .5, 'n': .5}, weights)
        self.assertAlmostEqual(corrected['c'], .1)
        self.assertAlmostEqual(corrected['n'], .9)

    def test_recovers_conditional_probability_under_sampling_assumption(self):
        weights = {k: 1. for k in CLASSES};weights['none'] = 9.
        sampled_call = .8 / (.8 + .2 / 9)
        result = corrected_probabilities(options(), {'c': sampled_call, 'n': 1 - sampled_call}, weights)
        self.assertAlmostEqual(result['c'], .8)

    def test_preserves_keys_and_relative_combinations(self):
        criteria = options();criteria['d'] = criteria['c']
        weights = {k: 2. for k in CLASSES}
        result = corrected_probabilities(criteria, {'c': .2, 'n': .4, 'd': .4}, weights)
        self.assertEqual(list(result), ['c', 'n', 'd'])
        self.assertAlmostEqual(result['d'] / result['c'], 2.)
        self.assertAlmostEqual(math.fsum(result.values()), 1.)

    def test_rejects_unsupported_or_invalid_sampling_counts(self):
        source = {k: 100 for k in CLASSES}
        for value in (0, -1, 101, 1.5):
            sample = dict(source);sample['none'] = value
            with self.assertRaises(ValueError):inverse_inclusion(source, sample)
        with self.assertRaises(ValueError):inverse_inclusion({}, {})

    def test_rejects_invalid_probability_or_option_input(self):
        weights = {k: 1. for k in CLASSES}
        for p in ({'c': -.1, 'n': 1.1}, {'c': float('nan'), 'n': .5}, {'c': .2, 'n': .2}, {'n': .5, 'c': .5}):
            with self.assertRaises(ValueError):corrected_probabilities(options(), p, weights)
        criteria = options();criteria['c']['call_features']['loses_closed_status'] = None
        with self.assertRaises(ValueError):corrected_probabilities(criteria, {'c': .5, 'n': .5}, weights)

    def test_no_label_or_outcome_is_an_input(self):
        record = {'questions': {'action': {'criteria': options(), 'label': 'c'}}}
        weights = {k: 1. for k in CLASSES};weights['none'] = 9.
        before = corrected_probabilities(record['questions']['action']['criteria'], {'c': .5, 'n': .5}, weights)
        record['questions']['action']['label'] = 'n'
        after = corrected_probabilities(record['questions']['action']['criteria'], {'c': .5, 'n': .5}, weights)
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
