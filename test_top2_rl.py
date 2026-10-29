import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from arena import EfficiencyAgent
from head_rl import HeadPolicy, development_gate, paired_primary_comparison, plateau_checks, summarize, terminal_reward
from riichienv import RiichiEnv


class Top2EastTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path('head-rl-top2-east.json').read_text())

    def test_rewards_distinguish_top2_from_first_without_score_threshold(self):
        self.assertEqual([terminal_reward(i, 'top2') for i in range(1, 5)], [1, 1, 0, 0])
        self.assertEqual([terminal_reward(i) for i in range(1, 5)], [1, 0, 0, 0])
        with self.assertRaises(ValueError):
            terminal_reward(1, 'unknown')

    def test_gate_uses_top2_and_secondary_risk_metrics(self):
        base = {'top2_rate': .5, 'first_rate': .25, 'mean_net_points': 0, 'fourth_rate': .25}
        self.assertEqual(development_gate(1, base, base, self.config)[0], 'continue')
        self.assertEqual(development_gate(2, base, base, self.config)[0], 'stop_no_early_progress')
        higher_first_only = dict(base, first_rate=.375)
        self.assertEqual(development_gate(2, higher_first_only, base, self.config)[0], 'stop_no_early_progress')
        intermediate = dict(base, top2_rate=.53125)
        self.assertEqual(development_gate(2, intermediate, base, self.config)[0], 'continue')
        good = dict(base, top2_rate=39/64)
        self.assertEqual(development_gate(1, good, base, self.config)[0], 'target_reached')
        for bad in [dict(good, mean_net_points=-2000), dict(good, fourth_rate=.3125), dict(good, top2_rate=.4375)]:
            self.assertEqual(development_gate(1, bad, base, self.config)[0], 'stop_regression')
        self.assertEqual(development_gate(4, intermediate, base, self.config)[0], 'stop_target_not_reached')
        strong_base = dict(base, top2_rate=.75)
        self.assertEqual(development_gate(1, dict(base, top2_rate=.8), strong_base, self.config)[0], 'continue')

    def test_plateau_means_no_new_best_across_two_checks(self):
        self.assertEqual(plateau_checks(.5, [.5], .5), 2)
        self.assertEqual(plateau_checks(.5, [.55], .6), 0)
        self.assertEqual(plateau_checks(.5, [.6, .55], .55), 2)

    def test_paired_summary_counts_second_place_as_success(self):
        old = [{'seed': s, 'seat': seat, 'rank': 3, 'score': 24000} for s in [1, 2] for seat in range(4)]
        new = [dict(r, rank=2) for r in old]
        result = paired_primary_comparison(new, old, self.config)
        self.assertEqual(result['first_delta'], 0)
        self.assertEqual(result['primary_delta'], 1)
        self.assertEqual(summarize(new)['top2_rate'], 1)
        with self.assertRaises(ValueError):
            paired_primary_comparison(new[:-1], old, self.config)

    def test_real_environment_uses_east_and_final_rank_reward(self):
        self.assertEqual(RiichiEnv(game_mode='4p-red-east').game_mode, 1)
        self.assertEqual(RiichiEnv(game_mode='4p-red-half').game_mode, 2)
        policy = object.__new__(HeadPolicy)
        policy.config = self.config
        agent = EfficiencyAgent()
        policy.act = lambda obs, **kwargs: agent.act(obs)
        with patch('head_rl.RiichiEnv', wraps=RiichiEnv) as constructor:
            result, trace = policy.game(1599000, 0)
        self.assertEqual(constructor.call_args.kwargs['game_mode'], '4p-red-east')
        self.assertEqual(result['reward'], float(result['rank'] <= 2))
        self.assertEqual(result['engine_seed'], 1599000 << 32)
        self.assertEqual(trace, [])

    def test_seed_sets_and_training_budget(self):
        c = self.config
        train = set(range(c['training_seed_start'], c['training_seed_start']+c['max_stages']*c['training_seeds_per_stage']))
        dev = set(range(c['development_seed_start'], c['development_seed_start']+c['development_seeds']))
        confirm = set(range(c['confirmation_seed_start'], c['confirmation_seed_start']+c['confirmation_seeds']))
        self.assertFalse(train & dev or train & confirm or dev & confirm)
        self.assertEqual(c['max_stages']*c['updates_per_stage'], 32)
        self.assertEqual(len(train)*4, 128)

    def test_confirmation_qualifies_second_place_gain_without_first_place_gain(self):
        from scripts import confirm_head_rl
        config = dict(self.config, confirmation_seeds=2)
        def arena(policy, directory, c, deadline, frozen):
            rows = [{'seed': s, 'seat': seat, 'rank': 3 if frozen else 2,
                     'score': 24000 if frozen else 26000}
                    for s in range(c['development_seed_start'], c['development_seed_start']+c['development_seeds'])
                    for seat in range(4)]
            return summarize(rows), rows
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            (out/'confirmation-selection.json').write_text(json.dumps({'config': config, 'checkpoint': 'frozen-candidate'}))
            with patch('sys.argv', ['confirm_head_rl.py', '--run', str(out)]), \
                 patch.object(confirm_head_rl, 'HeadPolicy') as policy, \
                 patch.object(confirm_head_rl, 'evaluate', side_effect=arena) as evaluate:
                confirm_head_rl.main()
            result = json.loads((out/'confirmation-result.json').read_text())
            self.assertTrue(result['qualified'])
            self.assertEqual(result['primary_metric'], 'top2_rate')
            self.assertEqual(result['primary_delta_ci95'], [1, 1])
            self.assertEqual(result['paired']['first_delta'], 0)
            policy.assert_called_once_with(config, 'frozen-candidate')
            self.assertEqual([c.kwargs['frozen'] for c in evaluate.call_args_list], [True, False])


if __name__ == '__main__':
    unittest.main()
