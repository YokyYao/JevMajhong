"""Genuine CPU simulator branches with a rule fixture, not expert training data."""
import unittest
import torch
from arena import EfficiencyAgent
from call_counterfactual import CallInterventionPolicy, replay_call_prefix
from riichienv import ActionType
from call_policy import is_call_request
from semantic import build_request, legal_candidates
from single_decision_rl import action_key
from scripts.train_single_decision import intervention_advantages


class FixtureHead(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, decide, options): return options[:, 0] + self.anchor * 0


class RuleCallFixture(CallInterventionPolicy):
    def __init__(self, take_calls=False):
        self.config = {'seed': 2026111006, 'game_mode': '4p-red-east', 'seed_scheme': 'spaced-v1',
                       'reward_scale': 1000., 'exploration_uniform': .9, 'training_replicas': 32}
        self.reference = FixtureHead(); self.controls = {}; self.baseline = EfficiencyAgent()
        self.take_calls = take_calls

    def act(self, obs, frozen=False, stochastic=False, trace=None):
        if not frozen or stochastic: raise AssertionError('Continuation must be fixed')
        action = self.baseline.act(obs)
        actions = legal_candidates(obs)
        calls = [a for a in actions if a.action_type in (ActionType.CHI, ActionType.PON)]
        if self.take_calls and calls and action.action_type not in (ActionType.RON, ActionType.TSUMO): action = calls[0]
        if trace is not None:
            actions = legal_candidates(obs)
            if len(actions) > 1:
                request, actions, _ = build_request(obs)
                if is_call_request(request):
                    index = next(i for i, a in enumerate(actions) if action_key(a) == action_key(action))
                    options = torch.zeros(len(actions), 1); options[index] = .5
                    p = options[:, 0].softmax(0)
                    trace.append({'decide': torch.zeros(1), 'options': options, 'eligible': list(range(len(actions))),
                        'action': index, 'old_logp': float(p[index].log()), 'reference': p, 'old_value': 0.,
                        'action_keys': [action_key(a) for a in actions], 'had_open_meld': any(obs.to_dict()['melds']),
                        'types': [a['type'] for a in request['questions']['action']['criteria'].values()]})
        return action


class CallCounterfactualTests(unittest.TestCase):
    def test_only_one_legal_call_decision_receives_actual_terminal_score_credit(self):
        policy = RuleCallFixture(); seed, seat = 9717000, 0
        control, trace = policy.game(seed, seat, frozen=True, replica=-1)
        self.assertFalse(trace)
        self.assertGreater(control['available_intervention_decisions'], 0)
        for decision in policy.controls[seed, seat]['decisions']:
            kinds = set(decision['row']['types'])
            self.assertTrue(kinds <= {'chi', 'pon', 'none'})
            self.assertIn('none', kinds)
            self.assertTrue(kinds & {'chi', 'pon'})
        games, rows = [], []
        for replica in range(32):
            game, row = policy.game(seed, seat, training=True, replica=replica)
            self.assertEqual(len(row), 1); rows.extend(row); games.append(game)
            self.assertEqual(game['trainable_decisions'], 1)
            self.assertEqual(game['terminal_point_delta'], game['score'] - control['score'])
            if game['reused_control_outcome']:
                self.assertEqual(game['terminal_point_delta'], 0)
                self.assertEqual(row[0]['action'], row[0]['teacher_action'])
            else: self.assertNotEqual(row[0]['action'], row[0]['teacher_action'])
        self.assertTrue(any(g['reused_control_outcome'] for g in games))
        self.assertTrue(any(not g['reused_control_outcome'] for g in games))
        self.assertTrue(any(g['terminal_point_delta'] for g in games))
        diagnostic = intervention_advantages([control], games, rows, 32, 10.)
        self.assertEqual(diagnostic['states'], 32)
        for game, row in zip(games, rows):
            self.assertAlmostEqual(row['advantage'], game['terminal_point_delta'] / 10000.)
        self.assertFalse(policy.controls)

    def test_collector_refuses_strength_evaluation_and_other_opponent_protocol(self):
        policy = RuleCallFixture()
        with self.assertRaises(ValueError): policy.game(9717000, 0)
        with self.assertRaises(ValueError): policy.game(9717000, 0, frozen=True, replica=-1, old_opponent=True)

    def test_native_serialization_replays_observation_with_open_melds(self):
        policy = RuleCallFixture(take_calls=True); seed, seat = 9717000, 0
        control, _ = policy.game(seed, seat, frozen=True, replica=-1)
        saved = policy.controls[seed, seat]
        opened = next(d for d in saved['decisions'] if d['row']['had_open_meld'])
        observations = replay_call_prefix(policy._env(seed), saved['prefix'][:opened['step']], opened['digest'])
        self.assertTrue(any(observations[seat].to_dict()['melds']))
        with self.assertRaisesRegex(ValueError, 'state differs'):
            replay_call_prefix(policy._env(seed), saved['prefix'][:opened['step']], 'wrong')


if __name__ == '__main__': unittest.main()
