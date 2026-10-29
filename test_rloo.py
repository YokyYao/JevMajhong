import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from head_rl import development_gate, rloo_advantages, summarize


class RLOOTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path('head-rl-score-rloo.json').read_text())

    def episode(self, replica, reward, seed=1):
        return {'seed': seed, 'seat': 0, 'replica': replica, 'reward': reward,
                'engine_seed': seed << 32, 'game_mode': '4p-red-east',
                'old_opponent': False, 'trainable_decisions': 1}

    def test_leave_one_out_uses_only_other_episode_return(self):
        games = [self.episode(0, 10), self.episode(1, -5)]
        rows = [{'episode_key': (1, 0, i), 'return': g['reward']} for i, g in enumerate(games)]
        report = rloo_advantages(games, rows, scale=10)
        self.assertEqual([r['baseline_return'] for r in rows], [-5, 10])
        self.assertEqual([r['advantage'] for r in rows], [1.5, -1.5])
        self.assertEqual(report['informative_groups'], 1)

    def test_equal_return_groups_remain_in_batch_with_zero_advantage(self):
        games = [self.episode(0, 3), self.episode(1, 3)]
        rows = [{'episode_key': (1, 0, i), 'return': 3} for i in range(2)]
        report = rloo_advantages(games, rows)
        self.assertEqual(len(rows), 2)
        self.assertEqual(report['informative_groups'], 0)
        self.assertEqual([r['advantage'] for r in rows], [0, 0])

    def test_group_and_trace_mismatches_are_rejected(self):
        a, b = self.episode(0, 1), self.episode(1, 2)
        with self.assertRaises(ValueError):
            rloo_advantages([a], [])
        with self.assertRaises(ValueError):
            rloo_advantages([a, dict(b, engine_seed=9)], [])
        with self.assertRaises(ValueError):
            rloo_advantages([a, b], [{'episode_key': (1, 0, 0), 'return': 999}])
        with self.assertRaises(ValueError):
            rloo_advantages([a, b], [])

    def test_net_point_gate_counts_validation_checks_separately_from_data_phases(self):
        base = {'mean_net_points': 2000, 'top2_rate': .5, 'first_rate': .25, 'fourth_rate': .25}
        candidate = dict(base, mean_net_points=2300)
        self.assertEqual(development_gate(2, candidate, base, self.config)[0], 'continue')
        self.assertEqual(development_gate(4, candidate, base, self.config)[0], 'stop_target_not_reached')

    def test_runner_validates_at8_then_stops_at16_without_collecting_next_phase(self):
        from scripts import train_head_rl as runner
        config = dict(self.config)
        checkpoints, calls = [], []
        class FakePolicy:
            def __init__(self, config, resume):
                self.head = torch.nn.Linear(1, 1)
                self.critic = torch.nn.Linear(1, 1).requires_grad_(False)
                self.model = SimpleNamespace(lm=torch.nn.Linear(1, 1).requires_grad_(False))
                self.checks = 32
            def game(self, seed, seat, training, old_opponent, replica):
                calls.append((seed, seat, replica))
                g = {'seed':seed, 'seat':seat, 'replica':replica, 'engine_seed':seed<<32,
                     'game_mode':'4p-red-east','old_opponent':old_opponent,
                     'reward':float(replica), 'score':25000+replica*1000, 'rank':2,
                     'trainable_decisions':1}
                return g, [{'episode_key':(seed,seat,replica),'return':float(replica)}]
        class FakeTrainer:
            def __init__(self, policy):
                self.optimizer = torch.optim.AdamW(policy.head.parameters())
            def update(self, rows): return {'loss':0.0}
            def reference_kl(self, rows): return 0.0
        def save(policy, trainer, out, update, *rest):
            checkpoints.append(update)
            p = out/'checkpoints'/f'update-{update:06d}'; p.mkdir(parents=True,exist_ok=True); return p
        def evaluate(policy, directory, c, deadline, frozen=False):
            rows = [{'seed':s,'seat':seat,'rank':2,'score':26000}
                    for s in range(c['development_seed_start'],c['development_seed_start']+c['development_seeds'])
                    for seat in range(4)]
            return summarize(rows), rows
        diagnostics = {'sampled_states':32,'greedy_flips_vs_reference':0,'greedy_flip_fraction':0,'mean_entropy':1}
        with tempfile.TemporaryDirectory(dir=runner.ROOT/'runs', prefix='rloo-test-') as temporary:
            parent = Path(temporary); reference = parent/'reference'; reference.mkdir()
            for name in ['head.pt','adapter_model.safetensors']: (reference/name).write_bytes(b'frozen-reference')
            config.update(out=str(parent/'trial'),reference=str(reference))
            path = parent/'config.json'; path.write_text(json.dumps(config))
            with patch('sys.argv',['train_head_rl.py','--config',str(path)]), \
                 patch.object(runner,'HeadPolicy',FakePolicy), patch.object(runner,'PPO',FakeTrainer), \
                 patch.object(runner,'save_checkpoint',side_effect=save), patch.object(runner,'evaluate',side_effect=evaluate) as evaluate_mock, \
                 patch.object(runner,'rollout_diagnostics',return_value=diagnostics), patch('builtins.print'):
                runner.main()
            out = Path(config['out'])
            result = json.loads((out/'pilot-analysis.json').read_text())
            self.assertEqual(result['updates'],16)
            self.assertEqual(result['stop_reason'],'stop_no_early_progress')
            self.assertEqual([r['update'] for r in result['development']],[8,16])
            self.assertEqual(checkpoints,[0,4,8,12,16])
            self.assertEqual(len(calls),32)
            self.assertEqual(evaluate_mock.call_count,3)
            self.assertEqual(result['critic_trainable_parameters'],0)
            self.assertFalse((out/'training-games-stage-02.json').exists())
            self.assertIsNone(result['selected_for_confirmation'])


if __name__ == '__main__':
    unittest.main()
