"""Check actual reward credit, all kan routing, and the direction of head updates."""
import copy
from types import SimpleNamespace
import unittest

import torch
from jevmahjong.training.rewards import alternate_indices, credit_pair, fast_game_stats, fast_reward, hand_outcomes, pair_loss, pairwise_preference_loss, scope_for, HeadPolicyClient, opportunity
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
    def test_win_first_reward_uses_actual_winner_and_speed_not_hand_value(self):
        cfg=dict(mode='own_win_first_v1',win_weight=1.,fast_weight=.25,dealin_penalty=1.,
            net_weight=0.,point_scale=8000.,speed_gamma=.98)
        def settled(actor,target,points,draws):
            delta=[0]*4;delta[actor]=points;delta[target]-=points
            return ([{'type':'start_kyoku'}]+[{'type':'tsumo'}]*draws+
                [{'type':'hora','actor':actor,'target':target,'deltas':delta},{'type':'end_kyoku'}])
        cheap=hand_utility(settled(0,1,1000,8),0,0,0,.995,cfg)
        from jevmahjong.evaluation.call_head_audit import native_win_credit,check_close
        check_close(cheap,native_win_credit(settled(0,1,1000,8),0,0,0,cfg))
        expensive=hand_utility(settled(0,1,8000,8),0,0,0,.995,cfg)
        self.assertEqual(cheap['utility'],expensive['utility'])
        self.assertNotEqual(cheap['net_points'],expensive['net_points'])
        slow=hand_utility(settled(0,1,8000,24),0,0,0,.995,cfg)
        self.assertGreater(cheap['utility'],slow['utility'])
        other=hand_utility(settled(1,2,8000,8),0,0,0,.995,cfg)
        self.assertEqual(other['own_win'],0);self.assertEqual(other['utility'],0)
        hit=hand_utility(settled(1,0,8000,8),0,0,0,.995,cfg)
        self.assertEqual(hit['own_win'],0);self.assertEqual(hit['utility'],-1.)

    def test_call_in_kan_sampling_requires_mixed_menu_and_keeps_requested_alternative(self):
        row={'scope':'ordinary','eligible':[0,1,2],'teacher_action':0,
            'public_request':request(['none','chi','daiminkan'])}
        row['public_request']['state'].update(seat=0,riichi=[1],melds=[[],[],[],[]])
        for category in ('chi_in_kan','chi_in_kan_threat','chi_threat_in_kan'):
            self.assertTrue(opportunity(row,category))
            self.assertEqual(alternate_indices(row,category),[1])
        row['public_request']['questions']['action']['criteria']['a2']['type']='pon'
        self.assertFalse(opportunity(row,'chi_in_kan'))
        row['public_request']['questions']['action']['criteria']['a2']['type']='daiminkan'
        row['public_request']['state']['riichi']=[]
        self.assertFalse(opportunity(row,'chi_in_kan_threat'))

    def test_future_sampling_preserves_current_state_and_changes_only_draw_order(self):
        from collections import Counter
        from riichienv import RiichiEnv
        from jevmahjong.training.futures import sample_future
        from jevmahjong.core.state import build_request
        env=RiichiEnv(seed=7);env.reset();wall=list(env.wall);hands=env.hands
        future=sample_future(env,123)
        self.assertEqual(env.wall,wall);self.assertEqual(future.hands,hands)
        self.assertEqual(Counter(future.wall),Counter(wall));self.assertEqual(future.wall[:14],wall[:14])
        self.assertNotEqual(future.wall[14:],wall[14:])
        a=env.clone().get_observations();b=future.clone().get_observations()
        for seat in a:self.assertEqual(build_request(a[seat])[0],build_request(b[seat])[0])
        unseen=sample_future(env,123,'unseen')
        self.assertEqual(unseen.hands,hands);self.assertEqual(Counter(unseen.wall),Counter(wall))
        for tile in env.dora_indicators:self.assertEqual(unseen.wall[wall.index(tile)],tile)
        self.assertNotEqual(unseen.wall[:14],wall[:14])
        b=unseen.clone().get_observations()
        for seat in a:self.assertEqual(build_request(a[seat])[0],build_request(b[seat])[0])

    def test_exact_prefix_replay_and_hand_only_rollout_keep_native_settlements(self):
        from riichienv import RiichiEnv,ActionType
        from jevmahjong.training.futures import replay_prefix,finish_intervention_hand
        from jevmahjong.training.collector import call_observation_digest
        def choose(obs):
            legal=obs.legal_actions()
            return next((a for a in legal if a.action_type==ActionType.PASS),
                next((a for a in legal if a.action_type==ActionType.DISCARD),legal[0]))
        env=RiichiEnv(seed=17);observations=env.reset();expected=None
        for step in range(18):
            if step==12:expected=call_observation_digest(observations)
            observations=env.step({p:choose(o) for p,o in observations.items()})
        restored=RiichiEnv(seed=17);actual=replay_prefix(restored,env.mjai_log,12)
        self.assertEqual(call_observation_digest(actual),expected)
        class Policy:
            def act(self,obs,frozen=False):return choose(obs)
            def opponent(self,player,obs,seat,suite):return choose(obs)
        result=finish_intervention_hand(restored,{p:choose(o) for p,o in actual.items()},Policy(),0,'closed',0)
        hands=hand_outcomes(result['mjai_log'])
        self.assertTrue(hands[0]['complete']);self.assertEqual(sum(h['complete'] for h in hands),1)
        self.assertEqual(sum(result['scores_at_stop'])+1000*result['point_ledger']['unclaimed_riichi_sticks'],100000)

    def test_targeted_collection_uses_public_threat_and_open_status_without_forcing_calls(self):
        from riichienv import MeldType
        req = request(['chi', 'pon', 'none'])
        req['state'] = {'seat': 0, 'riichi': [0], 'melds': [[], [], [], []]}
        row = {'public_request': req, 'scope': 'calling', 'eligible': [0, 1, 2], 'teacher_action': 2}
        self.assertFalse(opportunity(row, 'chi_threat'))
        req['state']['riichi'] = [1]
        self.assertTrue(opportunity(row, 'chi_threat'))
        self.assertEqual(alternate_indices(row, 'chi_threat'), [0])
        # If the behavior calls, the counterfactual must include declining it.
        row['teacher_action'] = 0
        self.assertIn(2, alternate_indices(row, 'chi_threat'))
        row['scope'] = 'ordinary'; req['questions']['action']['criteria'] = {'a0': {'type': 'dahai'}, 'a1': {'type': 'dahai'}}
        req['state']['melds'][0] = [{'type': int(MeldType.Ankan)}]
        self.assertFalse(opportunity(row, 'ordinary_open_threat'))
        req['state']['melds'][0] = [{'type': int(MeldType.Pon)}]
        self.assertTrue(opportunity(row, 'ordinary_open_threat'))
        req['state']['riichi'] = []
        self.assertFalse(opportunity(row, 'ordinary_open_threat'))

    def test_candidate_collector_hooks_updated_heads_and_keeps_opponent_heads_frozen(self):
        from unittest.mock import patch
        from jevmahjong.training.collector import NativeCollector
        old = {scope: torch.nn.Linear(2, 2) for scope in ('ordinary', 'calling')}
        new = {scope: torch.nn.Linear(2, 2) for scope in old}
        clients = [SimpleNamespace(runtime=SimpleNamespace(model=torch.nn.Module())) for _ in range(2)]
        for client, head in zip(clients, old.values()): client.runtime.model.head = head
        cfg = {'discard_checkpoint': 'discard', 'call_checkpoint': 'call', 'behavior_heads': {'checkpoint': 'candidate'}}
        with patch('jevmahjong.training.collector.load_model', side_effect=clients), \
             patch('jevmahjong.training.collector.snapshot', return_value={'model_files': {}, 'behavior_heads': {'checkpoint': 'candidate'}}), \
             patch('jevmahjong.training.collector.load_updated_heads', return_value=new):
            collector = NativeCollector(cfg, 'cpu')
        try:
            self.assertIs(collector.policy_heads, new)
            for scope, client in zip(old, clients):
                self.assertIs(client.runtime.model.head, old[scope])
                self.assertFalse(old[scope]._forward_pre_hooks)
                self.assertEqual(len(new[scope]._forward_pre_hooks), 1)
        finally: collector.close()
        self.assertTrue(all(not h._forward_pre_hooks for h in new.values()))

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
