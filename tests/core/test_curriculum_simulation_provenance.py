import copy
import unittest

import numpy as np

from tools.audit_curriculum_mc_worlds import derive, outcome
from tools.trace_curriculum_mc_provenance import content_identity


def hand(winner=None, target=None):
    events = [dict(type='start_kyoku'), dict(type='tsumo'), dict(type='tsumo')]
    if winner is not None:
        events.append(dict(type='hora', actor=winner, target=target))
    return events+[dict(type='end_kyoku')]


class SimulationProvenanceTests(unittest.TestCase):
    def test_anonymized_ids_preserve_content_but_target_changes_do_not(self):
        a = dict(state={'seat': 0}, questions={'action': {'target': {'a0': 1.}}}, _meta={'id': 'private'})
        b = copy.deepcopy(a)
        b['_meta'] = {'id': 'anonymous', 'group_id': 'hash'}
        self.assertEqual(content_identity(a), content_identity(b))
        b['questions']['action']['target']['a0'] = .9
        self.assertNotEqual(content_identity(a), content_identity(b))

    def test_outcomes_count_all_draws_and_separate_other_win_from_dealin(self):
        self.assertEqual(outcome(hand(0, 0), 0, 0, 1), (1., .98, 0.))
        self.assertEqual(outcome(hand(1, 2), 0, 0, 1), (0., 0., 0.))
        self.assertEqual(outcome(hand(1, 0), 0, 0, 1), (0., 0., 1.))
        with self.assertRaises(AssertionError):
            outcome(hand(0, 0)[:-1], 0, 0, 1)

    def test_unobserved_action_is_not_negative_or_evidence_of_improvement(self):
        # Both labelled choices have the same outcomes. A legal, unobserved
        # third action stays masked, and ties give half the conditional mass.
        criteria = {'a0': {'type': 'none'}, 'a1': {'type': 'pon'}, 'a2': {'type': 'daiminkan'}}
        request = {'state': {'seat': 0}, 'questions': {'action': {'criteria': criteria}}}
        record = copy.deepcopy(request)
        record['questions']['action']['target'] = {'a0': .5, 'a1': .5, 'a2': 0.}
        row = dict(eligible=[0, 1], teacher_action=0, action=1, public_request=request,
            source_split='train', hand_index=0, episode_key=[11, 0], draws_before=0,
            counterfactual_worlds=[dict(future_seed=i, control={'mjai_log': hand()}, branch={'mjai_log': hand()}) for i in (7, 8)])
        result = derive([row], record, 42, 2, np)
        self.assertEqual(result['target_argmax_conditional_best_mass'], .5)
        self.assertTrue(result['target_argmax_is_control'])
        self.assertFalse(result['noncontrol_positive_lower_bound'])
        self.assertEqual(result['target_reproduction_max_error'], 0.)
        changed = copy.deepcopy(record)
        changed['questions']['action']['target'] = {'a0': .9, 'a1': .1}
        with self.assertRaises(AssertionError):
            derive([row], changed, 42, 2, np)
        bad = copy.deepcopy(row)
        bad['source_split'] = 'held'
        with self.assertRaises(AssertionError):
            derive([bad], record, 42, 2, np)


if __name__ == '__main__':
    unittest.main()
