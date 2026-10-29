"""Measure head-only training score gradients without changing policy weights."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from jevany.model import PointerHead
from head_rl import PPO
from gradient_baseline import score_weighted_baselines, trajectory_score_norm
from reference_calibration import calibrated_baselines


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Never overwrite a gradient diagnostic')
    protocol = json.loads((args.source / 'protocol.json').read_text())
    config = protocol['config']
    reference = ROOT / config['reference']
    if fingerprint(reference / 'head.pt') != protocol['reference_head_sha256']:
        raise ValueError('Behavior head changed')
    meta = torch.load(reference / 'head.pt', map_location='cpu', weights_only=False)
    if meta['head_residual_dim'] != 0:
        raise ValueError('This diagnostic expects the saved plain pointer head')
    dp, d = meta['head']['q.weight'].shape
    head = PointerHead(d, dp).to('cuda').eval()
    head.load_state_dict(meta['head'])
    initial = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    games = json.loads((args.source / 'training-games-stage-01.json').read_text())
    controls = json.loads((args.source / 'reference-controls-stage-01.json').read_text())
    rows = torch.load(args.source / 'rollout-stage-01.pt', map_location='cpu', weights_only=False)
    # Check whether the recorded first shuffle can reproduce the original update4.
    start = torch.load(args.source / 'checkpoints/update-000000/rl_state.pt', map_location='cpu', weights_only=False)
    fourth = torch.load(args.source / 'checkpoints/update-000004/rl_state.pt', map_location='cpu', weights_only=False)
    rng = random.Random()
    rng.setstate(start['python_rng'])
    order = list(range(len(rows)))
    rng.shuffle(order)
    replay = {'shuffle_rng_matches': rng.getstate() == fourth['python_rng'], 'optimizer_steps_replayed': 0}
    if replay['shuffle_rng_matches']:
        critic = torch.nn.Linear(d, 1).to('cuda').requires_grad_(False)
        torch.nn.init.zeros_(critic.weight)
        torch.nn.init.zeros_(critic.bias)
        policy = SimpleNamespace(head=head, critic=critic, config=config,
                                 model=SimpleNamespace(lm=torch.nn.Module()))
        optimizer = PPO(policy)
        for i in range(4):
            indices = order[i*config['minibatch']:(i+1)*config['minibatch']]
            optimizer.update([rows[j] for j in indices])
        expected = torch.load(args.source / 'checkpoints/update-000004/head.pt', map_location='cpu', weights_only=False)['head']
        difference = max(float((v.detach().cpu()-expected[k]).abs().max()) for k, v in head.state_dict().items())
        replay.update(optimizer_steps_replayed=4, max_abs_parameter_difference=difference,
                      reproduces_update4=difference <= 1e-6)
        head.load_state_dict(initial)
        head.zero_grad(set_to_none=True)
        print('optimizer_replay '+json.dumps(replay), flush=True)
    grouped = {(g['seed'], g['seat'], g['replica']): [] for g in games}
    for row in rows:
        grouped[tuple(row['episode_key'])].append(row)
    weights = {}
    for i, (key, trace) in enumerate(grouped.items(), 1):
        weights[key] = trajectory_score_norm(head, trace, config['exploration_uniform'], verify_cache=True)
        if i % 32 == 0 or i == len(grouped):
            print(json.dumps({'phase': 'gradient_diagnostic', 'episodes': i, 'planned': len(grouped)}), flush=True)
    if any(not torch.equal(initial[k], v.detach().cpu()) for k, v in head.state_dict().items()):
        raise ValueError('Diagnostic changed model weights')
    plain, _ = calibrated_baselines(games, controls)
    weighted, fits = score_weighted_baselines(games, controls, weights)
    control = {(g['seed'], g['seat']): g['reward'] for g in controls}
    keys = [(g['seed'], g['seat'], g['replica']) for g in games]
    candidates = {'raw_reference': {k: control[k[:2]] for k in keys},
                  'unweighted_crossfit': plain, 'score_weighted_crossfit': weighted}
    stats = {}
    for name, baseline in candidates.items():
        residual = [g['reward'] - baseline[k] for g, k in zip(games, keys)]
        stats[name] = {'return_rms': math.sqrt(sum(x*x for x in residual) / len(residual)),
                       'trajectory_score_weighted_rms': math.sqrt(sum(weights[k]*x*x for k, x in zip(keys, residual)) / sum(weights.values()))}
    record = {'training_only': True, 'new_policy_updates': 0, 'weights_restored_unchanged': True,
              'original_optimizer_replay': replay,
              'source_run': str(args.source), 'source_head_sha256': protocol['reference_head_sha256'],
              'source_rollout_sha256': fingerprint(args.source / 'rollout-stage-01.pt'),
              'source_games_sha256': fingerprint(args.source / 'training-games-stage-01.json'),
              'verified_cached_states': len(rows), 'independent_streams': len({g['seed'] for g in games}),
              'metric': 'Unclipped trajectory log-probability score gradient norm; not PPO performance',
              'residual_statistics': stats, 'fits': fits,
              'trajectory_weights': [{'episode_key': list(k), 'score_norm_squared': weights[k]} for k in keys],
              'code_sha256': {p: fingerprint(ROOT / p) for p in ['gradient_baseline.py', 'scripts/diagnose_gradient_baseline.py']}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + '\n')
    print('gradient_baseline_result ' + json.dumps(stats), flush=True)


if __name__ == '__main__':
    main()
