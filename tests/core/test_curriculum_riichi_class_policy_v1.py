import sys
import types
import unittest
from unittest.mock import Mock,patch
from tools.curriculum_riichi_class_policy_v1 import select,RiichiClassPolicyAgent


class RiichiClassSelection(unittest.TestCase):
    def test_fixed_class_changes_only_reach_discard_scope(self):
        self.assertEqual(select(['dahai','dahai','reach'],[.35,.25,.4],[0,1,2]),(0,True))
        self.assertEqual(select(['dahai','reach'],[.5,.5],[0,1]),(1,True))
        self.assertEqual(select(['dahai','dahai'],[.5,.5],[0,1]),(0,False))
        self.assertEqual(select(['dahai','reach','ankan'],[.2,.4,.4],[0,1,2]),(1,False))
        self.assertEqual(select(['chi','none'],[.7,.3],[1]),(1,False))

    def test_guard_is_retained_even_if_excluded_discard_has_max_score(self):
        self.assertEqual(select(['dahai','dahai','reach'],[.4,.2,.4],[1,2]),(1,True))
        self.assertEqual(select(['dahai','dahai','reach'],[.1,.3,.6],[0,1,2]),(2,True))

    def test_reject_invalid_distributions_and_missing_guarded_actions(self):
        for p,allowed in [([.4,.4],[0,1]),([float('nan'),.5],[0,1]),([-.1,1.1],[0,1]),([.5,.5],[]),([.5,.5],[1]),([.5,.5],[0]),([.5,.5],[2])]:
            with self.subTest(p=p,allowed=allowed),self.assertRaises(ValueError):select(['dahai','reach'],p,allowed)

    def test_forced_win_and_single_action_need_no_model_or_validator(self):
        state=types.ModuleType('jevmahjong.core.state');state.build_request=Mock(side_effect=AssertionError('No request needed'))
        env=types.ModuleType('riichienv');env.ActionType=types.SimpleNamespace(RON=1,TSUMO=2)
        client=Mock();client.system_one.side_effect=AssertionError('No forward needed')
        win=types.SimpleNamespace(action_type=1);discard=types.SimpleNamespace(action_type=3)
        with patch.dict(sys.modules,{'riichienv':env,'jevmahjong.core.state':state}):
            state.legal_candidates=lambda _: [discard,win]
            self.assertIs(RiichiClassPolicyAgent(client).act(None),win)
            state.legal_candidates=lambda _: [discard]
            self.assertIs(RiichiClassPolicyAgent(client).act(None),discard)
        client.system_one.assert_not_called();state.build_request.assert_not_called()
