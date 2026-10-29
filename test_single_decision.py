import copy
import unittest

import torch
from riichienv import GameRule, RiichiEnv

from arena import EfficiencyAgent
from head_rl import mixed_probabilities
from semantic import legal_candidates
from single_decision_rl import SingleDecisionPolicy, action_key, derived_seed, observation_digest, replay_prefix
from scripts.train_single_decision import intervention_advantages


class FakeHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, decide, options):
        return options[:, 0] + self.anchor * 0


class RuleTeacher(SingleDecisionPolicy):
    """CPU integration fixture: deterministic teacher and genuine simulator branches."""
    def __init__(self):
        self.config = dict(seed=17, game_mode='4p-red-east', seed_scheme='spaced-v1', reward_scale=1000.,
                           exploration_uniform=.15, training_replicas=32)
        self.reference = FakeHead()
        self.controls = {}
        self.baseline = EfficiencyAgent()

    def act(self, obs, frozen=False, stochastic=False, trace=None):
        if stochastic or not frozen:
            raise AssertionError('Prefix and continuation must use a frozen greedy teacher')
        actions = legal_candidates(obs)
        action = self.baseline.act(obs)
        index = next(i for i, a in enumerate(actions) if action_key(a) == action_key(action))
        if trace is not None and len(actions) > 1:
            options = torch.zeros(len(actions), 1)
            options[index] = .5
            p = torch.softmax(options[:, 0], 0)
            trace.append(dict(decide=torch.zeros(1), options=options, eligible=list(range(len(actions))),
                              action=index, old_logp=float(p[index].log()), reference=p, old_value=0.))
        return actions[index]


class SingleDecisionTests(unittest.TestCase):
    def test_replay_complete_prefix_and_state_mismatch(self):
        config = dict(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=3400000 << 32)
        env, teacher = RiichiEnv(**config), EfficiencyAgent()
        observations, prefix = env.reset(), []
        digests = []
        while not env.done():
            digests.append(observation_digest(observations))
            selected = {p: teacher.act(o) for p, o in observations.items()}
            prefix.append(selected)
            observations = env.step(selected)
        for step in [0, len(prefix)//2, len(prefix)-1]:
            other = RiichiEnv(**config)
            replayed = replay_prefix(other, prefix[:step], digests[step])
            self.assertEqual(observation_digest(replayed), digests[step])
            for batch in prefix[step:]:
                replayed = other.step(batch)
            self.assertTrue(other.done())
            self.assertEqual(other.scores(), env.scores())
            self.assertEqual(other.ranks(), env.ranks())
        with self.assertRaisesRegex(ValueError, 'state differs'):
            replay_prefix(RiichiEnv(**config), prefix[:2], 'invalid')
        with self.assertRaisesRegex(ValueError, 'player coverage'):
            replay_prefix(RiichiEnv(**config), [{}], digests[0])

    def test_one_decision_credit_and_exact_control_reuse(self):
        policy, seed, seat = RuleTeacher(), 3400001, 0
        control, trace = policy.game(seed, seat, frozen=True, replica=-1)
        self.assertFalse(trace)
        games, rows = [], []
        reused = branched = 0
        for replica in range(policy.config['training_replicas']):
            game, trace = policy.game(seed, seat, training=True, replica=replica)
            games.append(game); rows.extend(trace)
            self.assertEqual(len(trace), 1)
            self.assertEqual(game['trainable_decisions'], 1)
            row = trace[0]
            self.assertEqual(row['episode_key'], (seed, seat, replica))
            expected = mixed_probabilities(policy.reference(row['decide'], row['options']), row['eligible'], .15)
            self.assertTrue(torch.equal(expected, row['reference']))
            self.assertAlmostEqual(float(expected[row['action']].log()), row['old_logp'])
            if game['reused_control_outcome']:
                reused += 1
                self.assertEqual(game['terminal_point_delta'], 0)
                self.assertEqual(game['score'], control['score'])
                self.assertEqual(row['action'], row['teacher_action'])
            else:
                branched += 1
                self.assertNotEqual(row['action'], row['teacher_action'])
                self.assertGreater(game['simulated_continuation_steps'], 0)
        self.assertGreater(reused, 0)
        self.assertGreater(branched, 0)
        diagnostic = intervention_advantages([control], games, rows, 32, 10.)
        self.assertEqual(diagnostic['states'], 32)
        self.assertEqual(diagnostic['actual_branch_simulations'], branched)
        for game, row in zip(games, rows):
            self.assertAlmostEqual(row['advantage'], (game['score']-control['score'])/10000.)
        broken = copy.deepcopy(rows)
        broken[0]['intervention_state_digest'] = 'different'
        with self.assertRaisesRegex(ValueError, 'verified intervention'):
            intervention_advantages([control], games, broken, 32, 10.)
        self.assertFalse(policy.controls)

    def test_sampling_domains_and_replicas_are_separate(self):
        values = {derived_seed(17, s, seat, replica, domain) for s in range(2)
                  for seat in range(4) for replica in range(4) for domain in ['position', 'action']}
        self.assertEqual(len(values), 64)
        self.assertEqual(derived_seed(17, 1, 2, 3, 'position'), derived_seed(17, 1, 2, 3, 'position'))


if __name__ == '__main__':
    unittest.main()
