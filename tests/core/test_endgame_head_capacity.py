import copy
import json
import math
from pathlib import Path
import unittest

from tools.probe_endgame_head_capacity import (validate_config, encoded_public_state_audit,
                                              target_stats, validated_head_distribution)


class HeadCapacityProbeTests(unittest.TestCase):
    def test_probe_cannot_use_validation_or_update_a_release(self):
        config = json.loads((Path(__file__).resolve().parents[2] / 'configs/training/endgame-existing-head-capacity-v1.json').read_text())
        validate_config(config)
        for change in ({'validation_data_used': True}, {'promotion_registered': True}, {'encoder_frozen': False},
                       {'candidate_training_registered': True}, {'encoder_queries': 20}, {'max_head_updates': 256}):
            with self.assertRaises(ValueError): validate_config(dict(config, **change))

    def test_encoded_state_must_contain_public_features_and_exclude_targets(self):
        state = 'scores endgame_context gap_to_second gap_to_third'
        encoding = {'ids': [1, 2, 3], 'seg': [0, 0, 1], 'state_truncated': False}
        self.assertEqual(encoded_public_state_audit(state, state, encoding)['state_tokens'], 2)
        with self.assertRaises(ValueError): encoded_public_state_audit(state, state.replace('endgame_context', ''), encoding)
        with self.assertRaises(ValueError): encoded_public_state_audit(state, state, dict(encoding, state_truncated=True))
        with self.assertRaises(ValueError): encoded_public_state_audit(state + ' expected_terminal_utilities', state, encoding)

    def test_tied_targets_use_excess_ce_and_actual_supported_choice(self):
        target = {'a': .5, 'b': .5, 'other': 0.}
        stats = target_stats(target, {'a': .49, 'b': .49, 'other': .02}, list(target))
        self.assertTrue(stats['hit'])
        self.assertAlmostEqual(stats['target_entropy'], math.log(2))
        self.assertLess(stats['excess_ce'], .03)
        self.assertFalse(target_stats(target, {'a': .1, 'b': .1, 'other': .8}, list(target))['hit'])
        with self.assertRaises(ValueError): target_stats(target, {'a': .5, 'b': .5, 'other': float('nan')}, list(target))

    def test_cached_head_probabilities_use_the_string_key_api_contract(self):
        def validator(probabilities, keys):
            self.assertIsInstance(probabilities, dict)
            self.assertEqual(list(probabilities), keys)
            return [probabilities[k] for k in keys], None
        self.assertEqual(validated_head_distribution([.2, .8], ['a', 'b'], validator), {'a': .2, 'b': .8})
        with self.assertRaises(ValueError): validated_head_distribution([1.], ['a', 'b'], validator)


if __name__ == '__main__': unittest.main()
