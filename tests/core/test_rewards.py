"""Check actual reward credit, all kan routing, and the direction of head updates."""
import copy
from types import SimpleNamespace
import unittest

import torch
from jevmahjong.training.rewards import alternate_indices, credit_pair, fast_game_stats, fast_reward, hand_outcomes, pair_loss, pairwise_preference_loss, scope_for, HeadPolicyClient
from jevmahjong.training.utility import hand_utility


def log(winner, draws, second_winner=None):
    result = [{'type': 'start_kyoku'}] + [{'type': 'tsumo'} for _ in range(draws)]
    result += [{'type': 'ryukyoku'}] if winner is None else [{'type': 'hora', 'actor': winner}]
    result += [{'type': 'end_kyoku'}]
    if second_winner is not None:
        result += log(second_winner, 2)
    return result


def request(types):
    return {'state': {}, 'questions': {'action': {'criteria': {f'a{i}': {'type': k} for i, k in enumerate(types)}}}}


class FastWinTest(unittest.TestCase):
    def test_point_primary_utility_preserves_payment_severity_without_saturation(self):
        config = {'mode': 'net_points_with_fast_support_v1', 'net_weight': 1., 'fast_weight': .1,
            'dealin_penalty': .1, 'point_scale': 8000.}
        def settled(amount):
            return [{'type': 'start_kyoku'}, {'type': 'hora', 'actor': 1, 'target': 0,
                'deltas': [-amount, amount, 0, 0]}, {'type': 'end_kyoku'}]
        small = hand_utility(settled(8000), 0, 0, 0, .995, config)
        large = hand_utility(settled(16000), 0, 0, 0, .995, config)
        self.assertAlmostEqual(small['utility'], -1.1)
        self.assertAlmostEqual(large['utility'], -2.1)
        self.assertAlmostEqual(small['utility'] - large['utility'], 1.)
        config['fast_weight'] = -1
        with self.assertRaises(ValueError): hand_utility(settled(8000), 0, 0, 0, .995, config)

    def test_risk_utility_penalizes_dealin_and_keeps_own_win_primary(self):
        config = {'mode': 'risk_adjusted_fast_win_v1', 'dealin_penalty': .3, 'net_weight': .1, 'point_scale': 8000.}
        def settled(actor, target, delta, draws=10):
            return [{'type': 'start_kyoku'}] + [{'type': 'tsumo'}] * draws + [
                {'type': 'hora', 'actor': actor, 'target': target, 'deltas': delta}, {'type': 'end_kyoku'}]
        ron = hand_utility(settled(1, 0, [-8000, 8000, 0, 0]), 0, 0, 0, .995, config)
        tsumo = hand_utility(settled(1, 1, [-8000, 12000, -2000, -2000]), 0, 0, 0, .995, config)
        self.assertAlmostEqual(tsumo['utility'] - ron['utility'], .3)
        cheap = hand_utility(settled(0, 1, [1000, -1000, 0, 0]), 0, 0, 0, .995, config)
        valuable = hand_utility(settled(0, 1, [12000, -12000, 0, 0]), 0, 0, 0, .995, config)
        self.assertGreater(cheap['utility'], 0)
        self.assertGreater(valuable['utility'], cheap['utility'])
        self.assertEqual(valuable['fast_reward'], cheap['fast_reward'])

    def test_risk_utility_counts_current_deposit_but_never_future_hand_points(self):
        config = {'mode': 'risk_adjusted_fast_win_v1', 'dealin_penalty': .3, 'net_weight': .1, 'point_scale': 8000.}
        events = [{'type': 'start_kyoku'}, {'type': 'reach_accepted', 'actor': 0},
            {'type': 'ryukyoku', 'deltas': [0] * 4}, {'type': 'end_kyoku'}]
        before = hand_utility(events, 0, 0, 0, .995, config)
        events += [{'type': 'start_kyoku'}, {'type': 'tsumo'},
            {'type': 'hora', 'actor': 0, 'target': 1, 'deltas': [48000, -48000, 0, 0]}, {'type': 'end_kyoku'}]
        self.assertEqual(hand_utility(events, 0, 0, 0, .995, config), before)
        self.assertEqual(before['net_points'], -1000)
        self.assertEqual(before['fast_reward'], 0.)

    def test_direct_ranking_prioritizes_wrong_order_over_already_correct_order(self):
        logits = torch.tensor([0., -4.], requires_grad=True)
        reference = logits.detach().clone()
        relative = pairwise_preference_loss(logits, reference, 1, 0, -1.)
        correct = pairwise_preference_loss(logits, reference, 1, 0, -1., reference_margin_weight=0.)
        wrong = pairwise_preference_loss(logits, reference, 1, 0, 1., reference_margin_weight=0.)
        self.assertLess(float(correct.detach()), float(relative.detach()) / 10)
        self.assertGreater(float(wrong.detach()), float(relative.detach()) * 5)
        wrong.backward()
        self.assertLess(float(logits.grad[1]), float(logits.grad[0]))
        with self.assertRaises(ValueError):
            pairwise_preference_loss(logits, reference, 1, 0, 1., reference_margin_weight=-1.)

    def test_opponent_wins_and_draws_zero_faster_own_win_better(self):
        self.assertEqual(fast_reward(hand_outcomes(log(1, 4))[0], 0, 0, .995), 0.)
        self.assertEqual(fast_reward(hand_outcomes(log(None, 60))[0], 0, 0, .995), 0.)
        self.assertGreater(fast_reward(hand_outcomes(log(0, 4))[0], 0, 0, .995),
            fast_reward(hand_outcomes(log(0, 40))[0], 0, 0, .995))

    def test_future_hand_wins_and_points_never_reward_current_hand(self):
        a, b = log(1, 10, 0), log(1, 10, 2)
        a[1]['score'] = 48000
        self.assertEqual(credit_pair(a, b, 0, 0, 5, .995)['fast_advantage'], 0.)
        self.assertGreater(fast_game_stats(a, 0, .995)['fast_reward_sum'], 0.)

    def test_same_hand_win_and_speed_labels_have_correct_sign(self):
        self.assertGreater(credit_pair(log(1, 10), log(0, 20), 0, 0, 2, .995)['fast_advantage'], 0)
        self.assertGreater(credit_pair(log(0, 30), log(0, 10), 0, 0, 2, .995)['fast_advantage'], 0)
        with self.assertRaises(ValueError):
            fast_reward(hand_outcomes(log(0, 10))[0], 0, 12, .995)

    def test_all_three_kans_retain_discard_encoder_and_are_explicit_contrasts(self):
        self.assertEqual(scope_for(request(['chi', 'pon', 'none'])), 'calling')
        for kind in ('ankan', 'daiminkan', 'kakan'):
            req = request(['none', kind, 'pon'])
            self.assertEqual(scope_for(req), 'ordinary')
            row = {'public_request': req, 'eligible': [0, 1, 2], 'teacher_action': 0}
            self.assertEqual(alternate_indices(row, kind), [1])
            row['teacher_action'] = 1
            self.assertEqual(alternate_indices(row, kind), [0, 2])
        self.assertIsNone(scope_for(request(['hora', 'none'])))

    def test_reference_relative_gradient_increases_rewarded_alternative(self):
        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.logits = torch.nn.Parameter(torch.zeros(2))
            def forward(self, *args):
                return self.logits
        head = Head(); row = {'decide': torch.zeros(1), 'options': torch.zeros(2, 1),
            'reference': torch.tensor([.5, .5]), 'eligible': [0, 1], 'teacher_action': 0,
            'action': 1, 'fast_advantage': 1.}
        loss, _, _ = pair_loss(head, row, .05, 1.); loss.backward()
        self.assertLess(head.logits.grad[1], head.logits.grad[0])
        head.zero_grad(); row['fast_advantage'] = -1.
        pair_loss(head, row, .05, 1.)[0].backward()
        self.assertGreater(head.logits.grad[1], head.logits.grad[0])
        head.zero_grad(); row['fast_advantage'] = 1.; row['training_advantage'] = -1.
        pair_loss(head, row, .05, 1.)[0].backward()
        self.assertGreater(head.logits.grad[1], head.logits.grad[0])

    def test_temporary_ordinary_head_restored_for_frozen_strong_opponent(self):
        original, updated = object(), object()
        model = SimpleNamespace(head=original)
        class Client:
            runtime = SimpleNamespace(model=model)
            def system_one(self, **payload):
                self.seen = self.runtime.model.head
                raise RuntimeError('test inference failure')
        old = Client(); router = HeadPolicyClient(old, None, {'ordinary': updated})
        with self.assertRaises(RuntimeError):
            router.system_one(**request(['dahai', 'ankan']))
        self.assertIs(old.seen, updated); self.assertIs(model.head, original)


if __name__ == '__main__':
    unittest.main()
