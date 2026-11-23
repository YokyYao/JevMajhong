import sys
import types
import unittest
from unittest.mock import Mock, patch

from tools import curriculum_complete_legal_class_policy_v1 as policy


class CompleteLegalClassTests(unittest.TestCase):
    def test_ordinary_and_mixed_kan_menus_keep_all_legal_discards(self):
        self.assertEqual(policy.select(['dahai']*3, [.7, .2, .1]), (0, False))
        for kind in ('ankan', 'kakan', 'ryukyoku'):
            self.assertEqual(policy.select(['dahai', 'dahai', kind], [.7, .2, .1]), (0, False))
        for kind in ('chi', 'pon', 'daiminkan'):
            self.assertEqual(policy.select(['none', kind], [.1, .9]), (1, False))
            self.assertEqual(policy.select(['none', kind], [.9, .1]), (0, False))

    def test_frozen_RI_full_class_threshold_is_preserved(self):
        kinds = ['reach', 'dahai', 'dahai']
        self.assertEqual(policy.select(kinds, [.4, .35, .25]), (1, True))
        self.assertEqual(policy.select(kinds, [.5, .3, .2]), (0, True))
        self.assertEqual(policy.select(kinds, [.6, .3, .1]), (0, True))
        with self.assertRaises(ValueError): policy.select(kinds, [.4, .4, .4])

    def test_single_actions_and_actual_win_types_skip_the_model(self):
        state = types.ModuleType('jevmahjong.core.state')
        state.build_request = Mock(side_effect=AssertionError('No request needed'))
        native = types.ModuleType('riichienv'); native.ActionType = types.SimpleNamespace(RON=1, TSUMO=2)
        agents = types.ModuleType('jevmahjong.evaluation.agents'); agents.guarded_indices = Mock()
        client = Mock(); client.system_one.side_effect = AssertionError('No model needed')
        discard = types.SimpleNamespace(action_type=3)
        with patch.dict(sys.modules, {'riichienv': native, 'jevmahjong.core.state': state,
                                     'jevmahjong.evaluation.agents': agents}):
            for kind in (1, 2):
                win = types.SimpleNamespace(action_type=kind)
                state.legal_candidates = lambda _: [discard, win]
                self.assertIs(policy.CompleteLegalClassPolicyAgent(client).act(None), win)
            state.legal_candidates = lambda _: [discard]
            self.assertIs(policy.CompleteLegalClassPolicyAgent(client).act(None), discard)
        client.system_one.assert_not_called(); state.build_request.assert_not_called()


if __name__ == '__main__': unittest.main()
