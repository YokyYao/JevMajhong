import copy
import sys
import types
import unittest
from unittest.mock import Mock, patch

from tools import curriculum_riichi_risk_scope_policy_v1 as policy
from tools.curriculum_riichi_class_policy_v1 import select


class RiichiRiskScopeTests(unittest.TestCase):
    def state(self, riichi): return dict(seat=0, riichi=riichi, melds=[[], [], [], []])

    def test_only_threatened_ordinary_discards_release_the_guard_in_open_and_closed_hands(self):
        for opened in (False, True):
            state = self.state([1]); state['melds'][0] = ['pon'] if opened else []
            saved = copy.deepcopy(state)
            self.assertEqual(policy.eligible(['dahai']*3, state, [1]), ([0, 1, 2], True))
            self.assertEqual(state, saved)
        for types, state in ((['dahai']*3, self.state([])), (['dahai']*3, self.state([0, 1])),
            (['dahai', 'ankan'], self.state([1])),
            (['none', 'chi'], self.state([1])), (['none', 'daiminkan'], self.state([1]))):
            self.assertEqual(policy.eligible(types, state, [0]), ([0], False))

    def test_legal_fold_can_be_selected_without_changing_RI_or_unthreatened_decisions(self):
        types = ['dahai']*3; p = [.7, .2, .1]
        allowed, scope = policy.eligible(types, self.state([1]), [1, 2])
        self.assertTrue(scope); self.assertEqual(select(types, p, allowed)[0], 0)
        allowed, scope = policy.eligible(types, self.state([]), [1, 2])
        self.assertFalse(scope); self.assertEqual(select(types, p, allowed)[0], 1)
        types = ['reach', 'dahai', 'dahai']; p = [.4, .35, .25]
        allowed, scope = policy.eligible(types, self.state([1]), [0, 1, 2])
        self.assertTrue(scope); self.assertEqual(select(types, p, allowed), (1, True))
        # Release a previously blocked discard, keeping the reach/decline
        # class fixed at the exact same0.5 operating point.
        allowed, scope = policy.eligible(types, self.state([1]), [0, 2])
        self.assertEqual(select(types, p, [0, 2])[0], 2)
        self.assertEqual(select(types, p, allowed)[0], 1)
        self.assertEqual(select(types, [.6, .3, .1], allowed)[0], 0)

    def test_public_RI_is_a_list_of_seat_indices_not_boolean_flags(self):
        for riichi in ([False, True, False, False], [1, 1], [4], None):
            with self.assertRaises(ValueError): policy.in_scope(['dahai'], self.state(riichi))
        with self.assertRaises(ValueError): policy.eligible(['dahai'], self.state([1]), [])

    def test_forced_wins_and_single_actions_skip_model_and_validator(self):
        state = types.ModuleType('jevmahjong.core.state')
        state.build_request = Mock(side_effect=AssertionError('No request needed'))
        agents = types.ModuleType('jevmahjong.evaluation.agents'); agents.guarded_indices = Mock()
        native = types.ModuleType('riichienv'); native.ActionType = types.SimpleNamespace(RON=1, TSUMO=2)
        client = Mock(); client.system_one.side_effect = AssertionError('No forward needed')
        win = types.SimpleNamespace(action_type=1); discard = types.SimpleNamespace(action_type=3)
        with patch.dict(sys.modules, {'riichienv': native, 'jevmahjong.core.state': state,
                                     'jevmahjong.evaluation.agents': agents}):
            state.legal_candidates = lambda _: [discard, win]
            self.assertIs(policy.RiichiRiskScopePolicyAgent(client).act(None), win)
            state.legal_candidates = lambda _: [discard]
            self.assertIs(policy.RiichiRiskScopePolicyAgent(client).act(None), discard)
        client.system_one.assert_not_called(); state.build_request.assert_not_called()


if __name__ == '__main__': unittest.main()
