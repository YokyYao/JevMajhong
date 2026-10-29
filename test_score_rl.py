import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from arena import EfficiencyAgent
from head_rl import HeadPolicy, development_gate, fingerprint, format_metric, paired_primary_comparison, state_value, summarize, terminal_reward


class ScoreRewardTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path('head-rl-score-east.json').read_text())

    def test_terminal_reward_is_signed_net_points_without_rank_bonus_or_clipping(self):
        self.assertEqual(terminal_reward(1, 'net_points', score=35000), 10)
        self.assertEqual(terminal_reward(3, 'net_points', score=15000), -10)
        self.assertEqual(terminal_reward(2, 'net_points', score=25000), 0)
        self.assertEqual(terminal_reward(4, 'net_points', score=-1000), -26)
        self.assertEqual(terminal_reward(1, 'net_points', score=100000), 75)
        self.assertEqual(terminal_reward(3, 'net_points', score=35000), 10)
        for kw in [{'score': None}, {'score': math.nan}, {'score': 35000, 'scale': 0}]:
            with self.assertRaises(ValueError):
                terminal_reward(1, 'net_points', **kw)

    def test_critic_can_predict_negative_and_large_values(self):
        critic = torch.nn.Linear(1, 1)
        with torch.no_grad():
            critic.weight.fill_(1); critic.bias.zero_()
        feature = torch.tensor([-10.])
        value = state_value(critic, feature, self.config)
        self.assertEqual(float(value.detach()), -10)
        (.5*(value-15).square()).backward()
        self.assertTrue(torch.isfinite(critic.weight.grad).all())
        self.assertEqual(float(state_value(critic, torch.tensor([15.]), self.config).detach()), 15)
        binary = float(state_value(critic, feature, {'reward_mode': 'top2'}).detach())
        self.assertGreater(binary, 0); self.assertLess(binary, 1)

    def test_net_point_gate_ignores_rank_rate_gains(self):
        base = {'mean_net_points': 2000, 'top2_rate': .5, 'first_rate': .25, 'fourth_rate': .25}
        self.assertEqual(development_gate(1, base, base, self.config)[0], 'continue')
        self.assertEqual(development_gate(2, base, base, self.config)[0], 'stop_no_early_progress')
        high_rank = dict(base, top2_rate=.9, first_rate=.6)
        self.assertEqual(development_gate(2, high_rank, base, self.config)[0], 'stop_no_early_progress')
        self.assertEqual(development_gate(1, dict(base, mean_net_points=1000), base, self.config)[0], 'stop_regression')
        self.assertEqual(development_gate(2, dict(base, mean_net_points=2250), base, self.config)[0], 'stop_target_not_reached')
        high_score = dict(base, mean_net_points=3000, first_rate=.1, top2_rate=.4, fourth_rate=.4)
        self.assertEqual(development_gate(1, high_score, base, self.config)[0], 'target_reached')
        negative = dict(base, mean_net_points=-3000)
        self.assertEqual(development_gate(1, dict(base, mean_net_points=-1500), negative, self.config)[0], 'continue')

    def test_real_east_terminal_score_is_the_recorded_reward(self):
        policy = object.__new__(HeadPolicy)
        policy.config = self.config
        agent = EfficiencyAgent()
        policy.act = lambda obs, **kwargs: agent.act(obs)
        result, trace = policy.game(1899000, 0)
        self.assertEqual(result['game_mode'], '4p-red-east')
        self.assertEqual(result['reward'], (result['score']-25000)/1000)
        self.assertEqual(result['reward_mode'], 'net_points')
        self.assertEqual(trace, [])

    def test_optional_development_precision_gate_cannot_select_noisy_gain(self):
        config = dict(self.config, development_min_ci95_lower=0, max_validation_checks=4)
        base = dict(mean_net_points=1000, first_rate=.25, top2_rate=.5, fourth_rate=.25)
        improved = dict(base, mean_net_points=2000)
        noisy = {'primary_delta_ci95': [-100, 2100]}
        positive = {'primary_delta_ci95': [10, 1990]}
        self.assertEqual(development_gate(1, improved, base, config, paired=noisy)[0], 'continue')
        self.assertEqual(development_gate(1, improved, base, config, paired=positive)[0], 'target_reached')
        self.assertEqual(development_gate(4, improved, base, config, paired=noisy)[0], 'stop_target_not_reached')
        with self.assertRaisesRegex(ValueError, 'finite paired development interval'):
            development_gate(1, improved, base, config)

    def test_paired_point_units_and_confirmation_use_net_points(self):
        from scripts import confirm_head_rl
        config = dict(self.config, confirmation_seeds=2)
        def arena(policy, directory, c, deadline, frozen):
            rows = [{'seed': s, 'seat': seat, 'rank': 2, 'score': 26000 if frozen else 28000}
                    for s in range(c['development_seed_start'], c['development_seed_start']+c['development_seeds'])
                    for seat in range(4)]
            return summarize(rows), rows
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            (out/'confirmation-selection.json').write_text(json.dumps({'config': config, 'checkpoint': 'frozen-score-candidate'}))
            with patch('sys.argv', ['confirm_head_rl.py', '--run', str(out)]), \
                 patch.object(confirm_head_rl, 'HeadPolicy'), patch.object(confirm_head_rl, 'evaluate', side_effect=arena):
                confirm_head_rl.main()
            result = json.loads((out/'confirmation-result.json').read_text())
            self.assertTrue(result['qualified'])
            self.assertEqual(result['primary_metric'], 'mean_net_points')
            self.assertEqual(result['primary_delta_ci95'], [2000, 2000])
            self.assertEqual(result['paired']['first_delta'], 0)
            self.assertIn('+3000.0点', (out/'analysis.md').read_text())
            self.assertEqual(format_metric(3000, 'mean_net_points'), '+3000.0点')

    def test_fresh_streams_and_short_budget(self):
        c = self.config
        a = set(range(c['training_seed_start'], c['training_seed_start']+c['max_stages']*c['training_seeds_per_stage']))
        b = set(range(c['development_seed_start'], c['development_seed_start']+c['development_seeds']))
        d = set(range(c['confirmation_seed_start'], c['confirmation_seed_start']+c['confirmation_seeds']))
        self.assertFalse(a & b or a & d or b & d)
        self.assertEqual(c['max_stages']*c['updates_per_stage'], 16)
        self.assertEqual(len(a)*4, 64)

    def test_confirmation_rejects_changed_preselected_head_before_loading(self):
        from scripts import confirm_head_rl
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            checkpoint = out / 'checkpoint'
            checkpoint.mkdir()
            head = checkpoint / 'head.pt'
            head.write_bytes(b'frozen candidate')
            selection = {'config': self.config, 'checkpoint': str(checkpoint),
                         'candidate_head_sha256': fingerprint(head)}
            (out / 'confirmation-selection.json').write_text(json.dumps(selection))
            head.write_bytes(b'changed candidate')
            with patch('sys.argv', ['confirm_head_rl.py', '--run', str(out)]), \
                 patch.object(confirm_head_rl, 'HeadPolicy') as loader:
                with self.assertRaisesRegex(ValueError, 'Preselected confirmation head changed'):
                    confirm_head_rl.main()
                loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
