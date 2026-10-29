"""Hardware smoke check: native feature replay, finite PPO gradients, frozen backbone."""
import copy
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from riichienv import RiichiEnv, GameRule
from arena import EfficiencyAgent
from head_rl import HeadPolicy, PPO, fingerprint, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT/'head-rl-pilot.json')
    parser.add_argument('--out', type=Path, default=ROOT/'runs/head-rl-smoke.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source = ROOT/config['reference']
    original_hash = fingerprint(source/'head.pt')
    policy = HeadPolicy(config)
    initial_head = copy.deepcopy(policy.head.state_dict())
    env = RiichiEnv(game_mode=config.get('game_mode', '4p-red-half'), rule=GameRule.default_mjsoul(), seed=1199000)
    observations, baseline, trace = env.reset(), EfficiencyAgent(), []
    for _ in range(3000):
        selected = {p: policy.act(o, stochastic=True, trace=trace) if p == 0 else baseline.act(o)
                    for p, o in observations.items()}
        observations = env.step(selected)
        if len(trace) >= 16:
            break
        if env.done():
            raise ValueError('Smoke game ended before collecting enough decisions')
    if len(trace) < 16 or policy.checks < 16:
        raise ValueError('Insufficient native probability and feature-alignment checks')
    # Synthetic returns below are only a gradient unit test, never training rollouts or reported win rates.
    for i, row in enumerate(trace):
        row['return'] = (-10.0 if i % 2 == 0 else 15.0) if config.get('reward_mode') == 'net_points' else float(i % 2)
        row['advantage'] = 1. if i % 2 else -1.
    optimizer = PPO(policy)
    losses = optimizer.update(trace)
    changed = any(not torch.equal(initial_head[k], policy.head.state_dict()[k]) for k in initial_head)
    if not changed or any(p.grad is not None for p in policy.model.lm.parameters()):
        raise ValueError('Head-only gradient isolation failed')
    policy.head.load_state_dict(initial_head)
    if fingerprint(source/'head.pt') != original_hash:
        raise ValueError('Reference checkpoint was modified')
    result = {'native_feature_checks': policy.checks, 'synthetic_gradient_test': True,
              'head_parameters_changed_then_restored': changed,
              'backbone_received_gradients': False, 'reference_file_unchanged': True, 'losses': losses}
    result['game_mode'] = config.get('game_mode', '4p-red-half')
    result['reward_mode'] = config.get('reward_mode', 'first')
    result['synthetic_return_range'] = [min(r['return'] for r in trace), max(r['return'] for r in trace)]
    write(args.out, result)
    print('head_rl_smoke '+json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
