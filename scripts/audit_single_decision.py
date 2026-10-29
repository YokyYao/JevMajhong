"""Recompute sampling and one-decision credit from private training records.

This checks data/likelihood/initialization consistency. It does not certify policy
strength or re-simulate the recorded branches; native smoke and evaluation do that.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from head_rl import deployment_greedy_index, mixed_probabilities
from single_decision_rl import derived_seed


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_records(games, controls, rows, config, head):
    start = config['training_seed_start']
    seeds = set(range(start, start + config['training_seeds_per_stage']))
    expected_controls = {(s, seat) for s in seeds for seat in range(4)}
    expected_games = {(s, seat, r) for s, seat in expected_controls for r in range(config['training_replicas'])}
    refs = {(g['seed'], g['seat']): g for g in controls}
    episodes = {(g['seed'], g['seat'], g['replica']): g for g in games}
    require(len(refs) == len(controls) and refs.keys() == expected_controls, 'Incomplete or duplicate control coverage')
    require(len(episodes) == len(games) and episodes.keys() == expected_games, 'Incomplete or duplicate proposal coverage')
    for game in controls + games:
        require(game['engine_seed'] == game['seed'] << 32 and game['seed_scheme'] == 'spaced-v1', 'Wrong wall stream')
        require(game['game_mode'] == '4p-red-east' and game['reward_mode'] == 'net_points', 'Wrong match/reward')
        require(game['reward_scale'] == config['reward_scale'], 'Wrong reward scale')
        require(game['old_opponent'] == bool(game['seed'] % 2), 'Wrong opponent composition')
        require(isinstance(game['score'], int) and game['rank'] in (1, 2, 3, 4), 'Invalid terminal outcome')
        require(math.isclose(game['reward'], (game['score'] - 25000) / config['reward_scale'], abs_tol=1e-10),
                'Incorrect terminal reward')
        require(game['collection_method'] == 'one_sampled_decision_frozen_continuation', 'Wrong collection method')
    for control in controls:
        require(control['frozen_control'] and control['trainable_decisions'] == 0
                and control['replica'] == -1 and control['sampling_seed'] is None, 'Invalid frozen control')
    traces = {}
    for row in rows:
        key = tuple(row['episode_key'])
        require(key in episodes and key not in traces, 'Each proposal may have only one training decision')
        traces[key] = row
    sampling_seeds, position_seeds, positive, negative, zero = set(), set(), 0, 0, 0
    branches, reused, signal_groups = 0, 0, set()
    with torch.no_grad():
        for key, game in episodes.items():
            control = refs[key[:2]]
            require(not game['frozen_control'], 'Proposal incorrectly marked as control')
            require(game['available_intervention_decisions'] == control['available_intervention_decisions'],
                    'Intervention decision coverage differs from its control')
            require(game['control_reward'] == control['reward'], 'Wrong paired control return')
            delta = game['score'] - control['score']
            require(game['terminal_point_delta'] == delta, 'Wrong paired point delta')
            sampling = derived_seed(config['seed'], *key, 'action')
            position = derived_seed(config['seed'], *key, 'position')
            require(game['sampling_seed'] == sampling and game['position_seed'] == position, 'Wrong independent sampling seeds')
            sampling_seeds.add(sampling); position_seeds.add(position)
            count = game['available_intervention_decisions']
            if count == 0:
                require(key not in traces and game['trainable_decisions'] == 0 and delta == 0
                        and game['reused_control_outcome'], 'A forced-action control cannot train an intervention')
                continue
            require(key in traces and game['trainable_decisions'] == 1, 'Missing single-decision trace')
            row = traces[key]
            selected_position = int(torch.randint(count, (1,), generator=torch.Generator().manual_seed(position)))
            require(selected_position == game['intervention_decision'] == row['intervention_decision'], 'Sampled position does not replay')
            require(0 <= game['intervention_step'] < control['environment_steps']
                    and game['intervention_step'] == row['intervention_step'], 'Invalid intervention step')
            require(len(game['intervention_state_digest']) == 64
                    and game['intervention_state_digest'] == row['intervention_state_digest'], 'Intervention state metadata differs')
            require(len(row['eligible']) > 1 and len(set(row['eligible'])) == len(row['eligible']), 'Invalid eligible action set')
            require(all(isinstance(i, int) and 0 <= i < len(row['options']) for i in row['eligible']), 'Invalid cached action indices')
            logits = head(row['decide'], row['options'])
            p = mixed_probabilities(logits, row['eligible'], config['exploration_uniform'])
            require(torch.allclose(p, row['reference'], atol=2e-5, rtol=2e-4), 'Cached behavior probabilities differ')
            sampled = int(torch.multinomial(p, 1, generator=torch.Generator().manual_seed(sampling)))
            require(sampled == row['action'] == game['sampled_action'], 'Sampled action does not replay')
            teacher = deployment_greedy_index(logits, row['eligible'])
            require(teacher == row['teacher_action'] == game['teacher_action'], 'Frozen teacher action differs')
            require(abs(float(p[sampled].log()) - row['old_logp']) < 2e-4, 'Wrong sampled-action log likelihood')
            require(row['return'] == game['reward'] and row['baseline_return'] == control['reward'], 'Wrong one-step return/baseline')
            require(abs(row['advantage'] - delta/config['reward_scale']/config['advantage_scale']) < 1e-10,
                    'One-step advantage differs from paired points')
            require(row['old_value'] == 0., 'Unexpected critic estimate')
            if game['reused_control_outcome']:
                reused += 1
                require(sampled == teacher and delta == 0 and game['rank'] == control['rank']
                        and game['simulated_prefix_steps'] == game['simulated_continuation_steps'] == 0,
                        'Invalid deterministic control reuse')
            else:
                branches += 1
                require(sampled != teacher and game['simulated_prefix_steps'] == game['intervention_step']
                        and game['simulated_continuation_steps'] == game['environment_steps'] - game['intervention_step']
                        and game['simulated_continuation_steps'] > 0, 'Invalid intervention branch step accounting')
            positive += delta > 0; negative += delta < 0; zero += delta == 0
            if delta:
                signal_groups.add(key[:2])
    require(len(sampling_seeds) == len(games) and len(position_seeds) == len(games)
            and not sampling_seeds & position_seeds, 'Sampling seeds collide across domains/replicas')
    return {'audit_status': 'passed', 'controls': len(controls), 'logical_proposals': len(games),
            'single_decision_states': len(rows), 'independent_training_streams': len(seeds),
            'actual_branch_simulations': branches, 'unchanged_action_reuses': reused,
            'positive_point_deltas': positive, 'negative_point_deltas': negative, 'zero_point_deltas': zero,
            'informative_groups': len(signal_groups), 'verified_sampling_replays': len(rows),
            'training_records_only': True, 'performance_claim': False}


def audit(run):
    from jevany.model import PointerHead
    protocol = json.loads((run / 'protocol.json').read_text())
    config = protocol['config']
    require(config['collection_method'] == 'single-decision-frozen-teacher-v1', 'Wrong pilot collection protocol')
    require(hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() == protocol['sha256'], 'Protocol changed')
    for name, expected in protocol['code_sha256'].items():
        require(sha256(ROOT / name) == expected, f'Numerical code changed: {name}')
    reference = ROOT / config['reference']
    require(sha256(reference / 'head.pt') == protocol['reference_head_sha256'], 'Source teacher weights changed')
    require(sha256(reference / 'adapter_model.safetensors') == protocol['reference_adapter_sha256'], 'Source LoRA changed')
    meta = torch.load(reference / 'head.pt', map_location='cpu', weights_only=False)
    require(meta['head_residual_dim'] == 0 and meta['temperature'] == 1., 'Unsupported pointer-head format')
    dp, d = meta['head']['q.weight'].shape
    head = PointerHead(d, dp).cpu().eval()
    head.load_state_dict(meta['head'])
    initial = torch.load(run / 'checkpoints/update-000000/head.pt', map_location='cpu', weights_only=False)
    require(initial['head'].keys() == meta['head'].keys()
            and all(torch.equal(v, initial['head'][k]) for k, v in meta['head'].items()), 'Initialization differs from frozen teacher')
    files = ['training-games-stage-01.json', 'reference-controls-stage-01.json', 'rollout-stage-01.pt']
    games, controls = [json.loads((run / name).read_text()) for name in files[:2]]
    rows = torch.load(run / files[2], map_location='cpu', weights_only=False)
    result = audit_records(games, controls, rows, config, head)
    for prefix, count in [('development', config['development_seeds']), ('confirmation', config['confirmation_seeds'])]:
        start = config[prefix + '_seed_start']
        require(not {g['seed'] for g in games} & set(range(start, start+count)), 'Training and evaluation streams overlap')
    result.update(protocol_sha256=protocol['sha256'], source_head_sha256=protocol['reference_head_sha256'],
                  initialization_tensor_identical=True, raw_sha256={n: sha256(run/n) for n in files},
                  scope='Training-data consistency; native replay smoke and independent complete-match evaluation required separately')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Preserve the existing audit; use a new output path')
    report = audit(args.run)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
