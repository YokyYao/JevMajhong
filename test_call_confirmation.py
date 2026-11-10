import copy
import unittest
from scripts.qualify_call_confirmation import qualify
from scripts.watch_call_repair import observation_complete


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.audit = {'status': 'passed_complete', 'complete': True, 'suites': {
            s: {'comparisons': {'candidate_minus_reference': {'gain': 100, 'ci95': [1, 199]}}}
            for s in ('closed', 'mixed')}}
        self.summaries = {s: {'is_final_confirmation': True, 'policies': {'candidate': {'mean_net_points': 20}}}
                          for s in ('closed', 'mixed')}

    def test_both_suites_must_confirm_not_just_positive_means(self):
        self.assertTrue(qualify(self.audit, self.summaries)['qualified'])
        self.audit['suites']['mixed']['comparisons']['candidate_minus_reference']['ci95'][0] = -1
        self.assertFalse(qualify(self.audit, self.summaries)['qualified'])

    def test_negative_absolute_points_do_not_qualify(self):
        self.summaries['mixed']['policies']['candidate']['mean_net_points'] = -1
        self.assertFalse(qualify(self.audit, self.summaries)['qualified'])

    def test_partial_or_development_results_cannot_qualify(self):
        bad = copy.deepcopy(self.audit); bad['complete'] = False
        with self.assertRaises(ValueError): qualify(bad, self.summaries)
        self.summaries['closed']['is_final_confirmation'] = False
        with self.assertRaises(ValueError): qualify(self.audit, self.summaries)

    def test_observer_waits_for_audit_and_qualification_even_when_games_finish(self):
        state = {'strength': {'status': 'development_complete'}}
        cfg = {'evaluation_stage': 'independent_confirmation'}
        self.assertFalse(observation_complete(cfg, state))
        state['audit'] = {'status': 'passed_complete', 'complete': True}
        self.assertFalse(observation_complete(cfg, state))
        state['qualification'] = {'qualified': False}
        self.assertTrue(observation_complete(cfg, state))
        self.assertTrue(observation_complete({'evaluation_stage': 'development'}, state))
        dev = {'evaluation_stage': 'development', 'require_complete_audit': True}
        self.assertFalse(observation_complete(dev, {'strength': {'status': 'development_complete'}}))
        self.assertTrue(observation_complete(dev, state))


if __name__ == '__main__': unittest.main()
