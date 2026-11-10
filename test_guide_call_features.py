import copy
import unittest
from guide_call_features import NAMES, call_discards, strategy_features, tile_ids


class GuideCallFeatureTests(unittest.TestCase):
    def test_native_mjai_actions_and_public_state_use_different_notations(self):
        self.assertEqual(tile_ids('5mr 5pr 5sr E S W N P F C'), tile_ids('0m 0p 0s 1z 2z 3z 4z 5z 6z 7z'))

    def test_open_yakuhai_and_red_value_are_covariates_not_extra_rewards(self):
        state = {'seat': 1, 'dealer': 0, 'round_wind': 0, 'honba': 0, 'sticks': 0,
            'scores': [25000] * 4, 'hand': '5z 5z 2m 3m 4m 4p 0p 6p 6s 7s 8s 2p 2p',
            'rivers': ['', '', '', ''], 'riichi': [], 'melds': [[], [], [], []], 'dora_indicators': '9m'}
        request = {'state': state, 'questions': {'action': {'criteria': {
            'a0': {'type': 'pon', 'pai': '5z', 'consumed': ['5z', '5z']}, 'a1': {'type': 'none'}}}}}
        values = strategy_features(request); f = dict(zip(NAMES, values[0]))
        self.assertEqual(f['secured_yakuhai_han'], .25)
        self.assertEqual(f['retained_red'], 1 / 3)
        self.assertEqual(f['first_open'], 1)
        self.assertEqual(values[1], [0.] * len(NAMES))
        changed = copy.deepcopy(request)
        changed['state']['future_wall'] = 'not an input'
        changed['state']['opponent_concealed_hands'] = 'not an input'
        self.assertEqual(strategy_features(changed), values)

    def test_kuikae_excludes_called_and_alternative_end_tiles(self):
        c = {'type': 'chi', 'pai': '1m', 'consumed': ['2m', '3m']}
        discards = {t // 4 for t, rest in call_discards(tile_ids('1m 2m 3m 4m 5m'), c)}
        self.assertEqual(discards, {4})

    def test_forced_discard_candidates_match_actual_native_environment(self):
        from test_call_counterfactual import RuleCallFixture
        from call_counterfactual import replay_call_prefix
        from semantic import build_request, legal_candidates, tile_name
        from riichienv import ActionType
        policy = RuleCallFixture(take_calls=True); seed, seat = 9717000, 0
        policy.game(seed, seat, frozen=True, replica=-1)
        saved = policy.controls[seed, seat]; tested = 0
        for decision in saved['decisions'][:5]:
            for i, kind in enumerate(decision['row']['types']):
                if kind == 'none': continue
                env = policy._env(seed)
                obs = replay_call_prefix(env, saved['prefix'][:decision['step']], decision['digest'])
                request, actions, _ = build_request(obs[seat]); features = strategy_features(request)
                criterion = list(request['questions']['action']['criteria'].values())[i]
                expected = {tile_name(t) for t, rest in call_discards(tile_ids(request['state']['hand']), criterion)}
                selected = dict(saved['prefix'][decision['step']]); selected[seat] = actions[i]
                after = env.step(selected)
                self.assertIn(seat, after)
                actual = {tile_name(a.tile) for a in legal_candidates(after[seat]) if a.action_type == ActionType.DISCARD}
                self.assertEqual(expected, actual)
                self.assertEqual(len(features), len(actions)); tested += 1
        self.assertGreater(tested, 0)

    def test_counterfactual_branch_keeps_its_own_audited_point_breakdown(self):
        import torch
        from test_call_counterfactual import RuleCallFixture
        from guide_call_collector import GuideCallInterventionPolicy
        from guide_call_features import VERSION
        from scripts.collect_guide_call_points import validate_wall
        from semantic import build_request

        class Fixture(GuideCallInterventionPolicy, RuleCallFixture):
            def act(self, obs, frozen=False, stochastic=False, trace=None):
                local = [] if trace is not None else None
                action = RuleCallFixture.act(self, obs, frozen, stochastic, local)
                if trace is not None:
                    for row in local:
                        request = build_request(obs)[0]
                        row.update(public_request=request, strategy_features=torch.tensor(strategy_features(request)),
                            strategy_feature_names=list(NAMES), strategy_feature_version=VERSION)
                        trace.append(row)
                return action

        policy = Fixture(take_calls=True); policy.config['training_replicas'] = 4
        controls, games, rows = [], [], []; seed = 9717000
        for seat in range(4):
            control, _ = policy.game(seed, seat, frozen=True, replica=-1); controls.append(control)
            for replica in range(4):
                game, trace = policy.game(seed, seat, training=True, replica=replica)
                games.append(game); rows.extend(trace)
        payload = {'seed':seed,'controls':controls,'games':games,'rows':rows}
        validate_wall(payload, seed, policy.config)
        self.assertTrue(any(g['terminal_point_delta'] != 0 for g in games))
        corrupted = copy.deepcopy(payload)
        branch = next(g for g in corrupted['games'] if g['terminal_point_delta'] != 0)
        branch['point_components']['invented_call_bonus'] = 1000
        with self.assertRaisesRegex(ValueError, 'Ledger/terminal'):
            validate_wall(corrupted, seed, policy.config)


if __name__ == '__main__': unittest.main()
