import copy
import unittest
from tools.audits.evaluation_recipes import check_recipes


def recipes():
    common = dict(discard_checkpoint='models/discard', call_checkpoint='models/call',
                  candidate_run='runs/candidate', updated_scopes=['ordinary'],
                  candidate_activation='kan_windows', train_scopes=['ordinary'],
                  decision_filter='kan_windows', confirmation_objective='hand_win',
                  fast_gamma=.98, max_dealin_rate_increase=.01,
                  required_fit_audit_config='runs/configs/fit-audit.json')
    development = dict(common, stage='development', out='runs/development', seed_start=100, seeds=8)
    preflight = dict(development, out='runs/preflight', preflight_only=True)
    confirmation = dict(common, stage='independent_confirmation', out='runs/confirmation',
                        development_out='runs/development', seed_start=200, seeds=32)
    return preflight, development, confirmation


class EvaluationRecipesTests(unittest.TestCase):
    def test_valid_links_are_read_only_and_do_not_claim_model_verification(self):
        configs = recipes(); original = copy.deepcopy(configs)
        result = check_recipes(*configs)
        self.assertEqual(configs, original)
        self.assertEqual(result['development_wall_range'], (100, 108))
        self.assertFalse(result['actual_weights_or_fit_eligibility_verified'])
        self.assertFalse(result['independent_strength_proven'])

    def test_copied_confirmation_cannot_reference_the_previous_candidate_development(self):
        p, d, c = recipes(); c['development_out'] = 'runs/previous-candidate-development'
        with self.assertRaisesRegex(ValueError, 'different development'):
            check_recipes(p, d, c)

    def test_candidate_scope_reference_and_risk_must_be_preserved(self):
        for field, value in [('candidate_run', 'runs/previous-candidate'),
                             ('updated_scopes', ['calling']), ('max_dealin_rate_increase', .02),
                             ('call_context_baseline', {'checkpoint': 'other', 'sha256': 'changed'}),
                             ('required_fit_audit_config', 'other-audit')]:
            with self.subTest(field=field):
                p, d, c = recipes(); c[field] = value
                with self.assertRaisesRegex(ValueError, 'identity differs'):
                    check_recipes(p, d, c)

    def test_overlapping_walls_preflight_only_matches_and_cached_references_are_rejected(self):
        for changes in ({'seed_start': 107}, {'out': 'runs/development'},
                        {'preflight_only': True}, {'reference_cache_out': 'runs/old-games'}):
            with self.subTest(changes=changes):
                p, d, c = recipes(); c.update(changes)
                with self.assertRaises(ValueError):check_recipes(p, d, c)
        p, d, c = recipes(); p['seeds'] = True
        with self.assertRaisesRegex(ValueError, 'wall counts'):check_recipes(p, d, c)


if __name__ == '__main__':
    unittest.main()
