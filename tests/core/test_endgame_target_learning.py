import copy
import json
from pathlib import Path
import unittest

from tools.audit_endgame_target_learning import (validate_config, original_instruction_variant,
                                                training_metrics, validation_metrics)


class TargetLearningDiagnosticTests(unittest.TestCase):
    def test_registered_diagnostic_cannot_promote_or_start_training(self):
        config = json.loads((Path(__file__).resolve().parents[2] / 'configs/training/endgame-target-learning-header-diagnostic-v1.json').read_text())
        validate_config(config)
        for change in ({'fit_registered': True}, {'promotion_registered': True}, {'max_model_queries': 305},
                       {'header_variant': 'search_prompts'}, {'control_max_abs_difference': 1e-4}):
            with self.assertRaises(ValueError): validate_config(dict(config, **change))

    def test_instruction_ablation_preserves_all_state_and_action_features(self):
        payload = {'state': {'scores': [40000, 30000, 20000, 10000], 'endgame_context': {'placement': 4}},
            'questions': {'action': {'type': 'choice', 'instructions': 'new',
                'criteria': {'a': {'type': 'dahai', 'sh': 1}, 'b': {'type': 'pon'}}}}}
        before = copy.deepcopy(payload)
        variant = original_instruction_variant(payload, {'questions': {'action': {'instructions': 'original'}}})
        self.assertEqual(payload, before)
        variant['questions']['action']['instructions'] = 'new'
        self.assertEqual(variant, before)
        with self.assertRaises(ValueError): original_instruction_variant(payload, {'questions': {'action': {}}})

    def test_target_learning_counts_actual_safe_choices_not_probability_drift(self):
        record = {'state': {'seat': 3}, '_meta': {'id': 'r', 'expected_terminal_utilities': {'old': -.05, 'better': 0.},
            'terminal_escape4_probabilities': {'old': 0., 'better': 1.}},
            'questions': {'action': {'src': 'endgame_terminal_fourth', 'label': 'better',
                'target': {'old': 0., 'better': 1.}, 'criteria': {'old': {'type': 'dahai'}, 'better': {'type': 'dahai'}}}}}
        ref = {'r': {'probabilities': {'old': .99, 'better': .01}}}
        weak = training_metrics([record], {'r': {'old': .6, 'better': .4}}, ref)
        strong = training_metrics([record], {'r': {'old': .4, 'better': .6}}, ref)
        self.assertEqual(weak['counts']['strict_target_hits'], 0)
        self.assertEqual(strong['counts']['strict_target_hits'], 1)
        self.assertLess(strong['mean_target_ce'], weak['mean_target_ce'])

    def test_validation_rejects_collapsing_or_omitting_escape_cohorts(self):
        with self.assertRaises(ValueError): validation_metrics([], {}, {})


if __name__ == '__main__': unittest.main()
