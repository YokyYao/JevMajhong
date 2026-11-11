"""Mechanism tests for one-action score credit and deterministic collection."""
import copy
from types import SimpleNamespace
import unittest

import torch
from call_policy import request_hash
from feature_point_collector import alternate_action, suite_for_seed
from guide_call_features import VERSION, NAMES
from scripts.collect_feature_points import validate_wall
from scripts.fit_feature_points import paired_features
from diagonal_value import pointer_products, wall_folds


def example():
    cfg = {'training_seed_start': 100, 'training_seeds_per_stage': 3,
        'training_replicas': 1, 'reward_scale': 1000., 'intervention_sampling': 'uniform_non_teacher'}
    controls, games, rows = [], [], []
    for seat in range(4):
        base = dict(seed=100, seat=seat, replica=-1, score=25000, reward=0.,
            point_components={'win': 0}, settlement_ledger_passed=True,
            suite='closed', old_opponent=False, game_mode='4p-red-east', seed_scheme='spaced-v1')
        controls.append(base)
        game = dict(base, replica=0, score=26000, reward=1., point_components={'win': 1000},
            terminal_point_delta=1000, reused_control_outcome=False, trainable_decisions=1,
            teacher_action=0, sampled_action=1, intervention_state_digest='digest')
        games.append(game)
        request = {'state': {}, 'questions': {'action': {'criteria': {'a0': {'type': 'none'}, 'a1': {'type': 'pon'}}}}}
        rows.append(dict(episode_key=(100, seat, 0), action=1, teacher_action=0,
            options=torch.zeros(2, 3), strategy_features=torch.zeros(2, len(NAMES)),
            strategy_feature_version=VERSION, strategy_feature_names=list(NAMES),
            behavior_probabilities=torch.tensor([0., 1.]), reference=torch.tensor([.7, .3]),
            intervention_sampling='uniform_non_teacher', intervention_state_digest='digest',
            public_request=request, request_sha256=request_hash(request), **{'return': 1.}))
    return {'seed': 100, 'controls': controls, 'games': games, 'rows': rows}, cfg


class FeaturePointsTest(unittest.TestCase):
    def test_alternatives_cover_every_non_teacher_and_are_reproducible(self):
        for n in range(2, 8):
            for teacher in range(n):
                g = torch.Generator().manual_seed(77)
                draws = [alternate_action(n, teacher, g) for _ in range(150)]
                self.assertEqual(set(draws), set(range(n)) - {teacher})
                h = torch.Generator().manual_seed(77)
                self.assertEqual(draws, [alternate_action(n, teacher, h) for _ in draws])
        with self.assertRaises(ValueError): alternate_action(1, 0, torch.Generator())

    def test_suites_balanced_and_range_checked(self):
        cfg = {'training_seed_start': 100, 'training_seeds_per_stage': 96}
        suites = [suite_for_seed(n, cfg) for n in range(100, 196)]
        self.assertEqual({s: suites.count(s) for s in set(suites)}, {'closed': 32, 'mixed': 32, 'strong': 32})
        with self.assertRaises(ValueError): suite_for_seed(196, cfg)

    def test_ledger_and_single_action_credit_validated(self):
        payload, cfg = example(); validate_wall(payload, 100, cfg)
        bad = copy.deepcopy(payload); bad['games'][0]['terminal_point_delta'] = 900
        with self.assertRaises(ValueError): validate_wall(bad, 100, cfg)
        bad = copy.deepcopy(payload); bad['games'][0]['point_components']['win'] = 900
        with self.assertRaises(ValueError): validate_wall(bad, 100, cfg)
        bad = copy.deepcopy(payload); bad['rows'][0]['teacher_action'] = 1
        with self.assertRaises(ValueError): validate_wall(bad, 100, cfg)
        bad = copy.deepcopy(payload); bad['games'][0]['suite'] = 'strong'
        with self.assertRaises(ValueError): validate_wall(bad, 100, cfg)

    def test_paired_projection_uses_supplied_teacher_and_is_antisymmetric(self):
        torch.manual_seed(8)
        head = SimpleNamespace(q=torch.nn.Linear(3, 2), k=torch.nn.Linear(3, 2),
            residual=None, temperature=1., scale=.7)
        row = dict(decide=torch.randn(3), options=torch.randn(2, 3),
            strategy_features=torch.randn(2, len(NAMES)), action=1, teacher_action=0)
        actual = paired_features(head, row)
        features = torch.cat((pointer_products(head, row['decide'], row['options']), row['strategy_features']), 1)
        self.assertTrue(torch.allclose(actual, features[1] - features[0]))
        reversed_row = dict(row, action=0, teacher_action=1)
        self.assertTrue(torch.equal(actual, -paired_features(head, reversed_row)))

    def test_cv_keeps_all_seats_and_replicas_of_wall_together(self):
        seeds = [wall for wall in range(16) for _ in range(16)]
        indices = wall_folds(seeds, folds=4)
        assignment = {}
        for fold, held in enumerate(indices):
            for i in held:
                self.assertEqual(assignment.setdefault(seeds[i], fold), fold)
        self.assertEqual(len(assignment), 16)


if __name__ == '__main__': unittest.main()
