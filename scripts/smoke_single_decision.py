"""Hardware verification of actual native control/intervention alignment, no training."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from head_rl import fingerprint, write
from single_decision_rl import SingleDecisionPolicy
from scripts.train_single_decision import intervention_advantages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config['training_replicas'] = 8
    seed, seat = 3401000, 0  # Smoke-only stream; disjoint from training/development/final.
    for start, length in [('training_seed_start', 'training_seeds_per_stage'),
                          ('development_seed_start', 'development_seeds'),
                          ('confirmation_seed_start', 'confirmation_seeds')]:
        if seed in range(config[start], config[start]+config[length]):
            raise ValueError('Smoke stream overlaps the experiment')
    source = ROOT/config['reference']/'head.pt'
    original_sha = fingerprint(source)
    policy = SingleDecisionPolicy(config)
    initial = copy.deepcopy(policy.head.state_dict())
    control, _ = policy.game(seed, seat, frozen=True, replica=-1)
    games, rows = [], []
    for replica in range(config['training_replicas']):
        game, trace = policy.game(seed, seat, training=True, replica=replica)
        games.append(game); rows.extend(trace)
    diagnostic = intervention_advantages([control], games, rows, config['training_replicas'], config['advantage_scale'])
    if len(rows) != config['training_replicas'] or diagnostic['actual_branch_simulations'] < 1:
        raise ValueError('Smoke did not exercise a verified changed-action branch')
    if policy.checks < 1 or policy.controls:
        raise ValueError('Native feature check or control cache cleanup failed')
    if any(not torch.equal(v, policy.head.state_dict()[k]) for k, v in initial.items()):
        raise ValueError('Counterfactual collection modified the behavior head')
    if fingerprint(source) != original_sha or any(p.grad is not None for p in policy.model.lm.parameters()):
        raise ValueError('Counterfactual smoke changed source weights or backbone gradients')
    result = dict(status='passed', smoke_seed=seed, native_feature_checks=policy.checks,
                  behavior_head_unchanged=True, actual_policy_updates=0, **diagnostic)
    write(args.out, result)
    print('single_decision_smoke '+json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
