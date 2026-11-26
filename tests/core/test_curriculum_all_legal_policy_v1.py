import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from riichienv import ActionType
from tools.curriculum_all_legal_policy_v1 import AllLegalPolicyAgent


class AllLegalPolicyTests(unittest.TestCase):
    def action(self, kind, name):
        return SimpleNamespace(action_type=kind, to_mjai=lambda:json.dumps({'type':name}))

    def test_raw_legal_choice_survives_shadow_guard_exclusion_with_one_forward(self):
        actions=[self.action(ActionType.DISCARD,'dahai') for _ in range(2)]
        request=dict(state={},questions={'action':{'criteria':{'a':{},'b':{}}}})
        calls=[]
        def answer(**value):
            calls.append(value)
            return {'answers':{'action':{'probabilities':{'a':0.1,'b':0.9}}}}
        validator=Mock(return_value=([0.1,0.9],None))
        agent=AllLegalPolicyAgent(SimpleNamespace(system_one=answer),validator=validator)
        with patch('jevmahjong.core.state.legal_candidates',return_value=actions), \
             patch('jevmahjong.core.state.build_request',return_value=(request,actions,[{'sh':0},{'sh':1}])), \
             patch('jevmahjong.evaluation.agents.guarded_indices',return_value=[0]):
            self.assertIs(agent.act(SimpleNamespace()),actions[1])
        self.assertEqual(calls,[request])
        validator.assert_called_once_with({'a':0.1,'b':0.9},['a','b'])
        self.assertEqual(agent.shadow['raw_choice_excluded_by_original_guard'],1)

    def test_native_win_and_single_action_do_not_call_the_model(self):
        def forbidden(**value):raise AssertionError('No forward pass is needed')
        agent=AllLegalPolicyAgent(SimpleNamespace(system_one=forbidden))
        discard=self.action(ActionType.DISCARD,'dahai')
        for kind in (ActionType.RON,ActionType.TSUMO):
            win=self.action(kind,'hora')
            with patch('jevmahjong.core.state.legal_candidates',return_value=[discard,win]):
                self.assertIs(agent.act(SimpleNamespace()),win)
        with patch('jevmahjong.core.state.legal_candidates',return_value=[discard]):
            self.assertIs(agent.act(SimpleNamespace()),discard)

    def test_invalid_distribution_cannot_select_an_out_of_menu_action(self):
        actions=[self.action(ActionType.DISCARD,'dahai') for _ in range(2)]
        request=dict(state={},questions={'action':{'criteria':{'a':{},'b':{}}}})
        client=SimpleNamespace(system_one=lambda **kw:{'answers':{'action':{'probabilities':{'unknown':1.0}}}})
        validator=Mock(side_effect=ValueError('Rejected distribution'))
        with patch('jevmahjong.core.state.legal_candidates',return_value=actions), \
             patch('jevmahjong.core.state.build_request',return_value=(request,actions,[{},{}])):
            with self.assertRaises((ValueError,KeyError)):
                AllLegalPolicyAgent(client,validator=validator).act(SimpleNamespace())
        validator.assert_called_once_with({'unknown':1.0},['a','b'])


if __name__=='__main__':unittest.main()
