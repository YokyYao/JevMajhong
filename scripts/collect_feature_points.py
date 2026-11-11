"""Register and collect checksummed stage2-native counterfactual whole walls."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from call_policy import digest, request_hash
from checkpoint_identity import completed_checkpoint_files
from feature_point_collector import FeaturePointPolicy, suite_for_seed
from guide_call_features import VERSION, NAMES
from head_rl import write
from scripts.collect_guide_call_points import validate_wall as validate_original_wall

CODE = ['feature_point_collector.py', 'scripts/collect_feature_points.py',
    'scripts/fit_feature_points.py', 'call_counterfactual.py', 'guide_call_collector.py',
    'guide_call_features.py', 'scoring_audit.py', 'single_decision_rl.py', 'head_rl.py',
    'arena.py', 'semantic.py', 'seed_control.py', 'call_policy.py', 'checkpoint_identity.py',
    'clean_policy_runtime.py', 'clean_policy_data.py', 'encoder_call_policy.py',
    'feature_policy_runtime.py', 'call_option_features.py', 'diagonal_value.py',
    'scripts/evaluate_call_repair.py', 'scripts/train_call_repair.py',
    'scripts/collect_guide_call_points.py', 'vendor/JevAny/jevany/model.py',
    'vendor/JevAny/jevany/inference.py', 'vendor/JevAny/jevany/data.py',
    'vendor/JevAny/jevany/client.py']


def validate_wall(payload, seed, cfg):
    validate_original_wall(payload, seed, cfg)
    suite = suite_for_seed(seed, cfg)
    if payload['seed'] != seed: raise ValueError('Wrong shard identity')
    games = {(g['seed'], g['seat'], g['replica']): g for g in payload['games']}
    for g in payload['controls'] + payload['games']:
        if g['suite'] != suite or g['old_opponent'] != (suite == 'strong'):
            raise ValueError('Opponent metadata differs from actual registered suite')
        if g['game_mode'] != '4p-red-east' or g['seed_scheme'] != 'spaced-v1':
            raise ValueError('Match rules changed')
    for row in payload['rows']:
        g = games[tuple(row['episode_key'])]
        teacher, action = row['teacher_action'], row['action']
        n = len(row['options'])
        if not (0 <= teacher < n and 0 <= action < n) or teacher == action:
            raise ValueError('Intervention must change one legal action')
        p = row['behavior_probabilities']
        expected = torch.full((n,), 1. / (n - 1)); expected[teacher] = 0.
        if not torch.equal(p, expected): raise ValueError('Wrong alternative sampling probabilities')
        if (g['sampled_action'] != action or g['teacher_action'] != teacher
                or g['reused_control_outcome'] or row['intervention_state_digest'] != g['intervention_state_digest']):
            raise ValueError('Branch and feature credit do not match')
        if row['intervention_sampling'] != cfg['intervention_sampling']:
            raise ValueError('Sampling rule changed')
        if row['request_sha256'] != request_hash(row['public_request']):
            raise ValueError('Public request changed')
        if list(row['public_request']['questions']['action']['criteria']) != [f'a{i}' for i in range(n)]:
            raise ValueError('Native action order changed')
        teacher_p = row['reference']
        if (len(teacher_p) != n or not torch.isfinite(teacher_p).all()
                or int(teacher_p.argmax()) != teacher or not torch.isclose(teacher_p.sum(), torch.tensor(1.))):
            raise ValueError('Frozen teacher distribution differs')


def assert_frozen(protocol, cfg):
    for name, h in protocol['code_sha256'].items():
        if digest(ROOT / name) != h: raise ValueError('Frozen collection source changed: ' + name)
    for name, h in protocol['reference_files'].items():
        if digest(ROOT / cfg['reference'] / name) != h: raise ValueError('Ordinary reference changed')
    files = completed_checkpoint_files(ROOT / cfg['teacher_checkpoint'], cfg['teacher_step'])
    if files != protocol['teacher_files']: raise ValueError('Stage2 teacher checkpoint changed')


def check_disjoint(cfg):
    ranges = [('training_seed_start', 'training_seeds_per_stage'),
              ('development_seed_start', 'development_seeds'),
              ('confirmation_seed_start', 'confirmation_seeds')]
    proposed = []
    for first, count in ranges:
        values = set(range(cfg[first], cfg[first] + cfg[count]))
        if any(values & old for old in proposed): raise ValueError('New split ranges overlap')
        proposed.append(values)
    for path in (ROOT / 'runs').glob('*/protocol.json'):
        old = json.loads(path.read_text()).get('config', {})
        for first, count in ranges:
            if first in old and count in old:
                stages = old.get('max_stages', 1) if first == 'training_seed_start' else 1
                values = set(range(old[first], old[first] + old[count] * stages))
                if any(values & new for new in proposed):
                    raise ValueError('Previously used/reserved wall overlap: ' + str(path))


def native_teacher_parity(policy, cfg):
    """Check existing validation predictions and exact ordinary routing, no test."""
    from jevany.data import api_request
    source_cfg = json.loads((ROOT / cfg['teacher_source_config']).read_text())
    run = ROOT / source_cfg['out']; data = ROOT / source_cfg['data']
    preparation = json.loads((run / 'preparation.json').read_text())
    for split in ('train', 'validation'):
        if digest(data / f'{split}.jsonl') != preparation['files'][split]:
            raise ValueError('Teacher parity data changed')
    saved = json.loads((run / 'training-run' / 'training_eval' /
        f"step-{cfg['teacher_step']:06d}" / 'development' / 'rows.json').read_text())
    index = {r['id']: r for r in saved}
    counts = {'reaction': 0, 'ordinary': 0}; maximum = 0.
    for line in (data / 'validation.jsonl').open():
        record = json.loads(line); meta = record['_meta']
        if meta['phase'] != 'reaction' or meta['view'] != 'native': continue
        expected = index[meta['id']]; request = api_request(record)
        keys = list(request['questions']['action']['criteria'])
        p = policy.router.system_one(**request)['answers']['action']['probabilities']
        if keys != expected['keys'] or list(p) != keys: raise ValueError('Teacher native order changed')
        actual = torch.tensor([p[k] for k in keys], dtype=torch.float64)
        target = torch.tensor(expected['p'], dtype=torch.float64)
        error = float((actual - target).abs().max()); maximum = max(maximum, error)
        if error > 2e-5 or actual.argmax() != target.argmax():
            raise ValueError('Stage2 native teacher differs from saved validation')
        counts['reaction'] += 1
        if counts['reaction'] == 6: break
    for line in (data / 'train.jsonl').open():
        record = json.loads(line)
        if record['_meta']['phase'] == 'reaction': continue
        request = api_request(record)
        before = policy.ordinary.system_one(**request)['answers']['action']['probabilities']
        after = policy.router.system_one(**request)['answers']['action']['probabilities']
        if before != after: raise ValueError('Counterfactual teacher changed ordinary decisions')
        counts['ordinary'] += 1
        if counts['ordinary'] == 6: break
    if counts != {'reaction': 6, 'ordinary': 6}: raise ValueError('Teacher parity coverage incomplete')
    return {'checked': counts, 'maximum_probability_error': maximum,
        'native_decisions_equal': True, 'ordinary_probabilities_unchanged': True,
        'source': 'already opened teacher TRAIN and validation only; no human test'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(); cfg = json.loads(args.config.read_text()); out = ROOT / cfg['out']
    # Resolve local paths for model loading; cfg/protocol stay portable and unchanged.
    out.mkdir(parents=True, exist_ok=args.resume)
    lock = (out / 'collector.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = json.loads((ROOT / cfg['reference_training_protocol']).read_text())
    teacher_files = completed_checkpoint_files(ROOT / cfg['teacher_checkpoint'], cfg['teacher_step'])
    if teacher_files['head.pt'] != cfg['teacher_head_sha256']: raise ValueError('Teacher head identity mismatch')
    protocol = {'config': cfg, 'feature_version': VERSION, 'feature_names': list(NAMES),
        'code_sha256': {n: digest(ROOT / n) for n in CODE},
        'reference_files': source['reference_files'], 'teacher_files': teacher_files,
        'human_test_opened': False, 'actor_updates': 0,
        'opponents': 'cyclic closed/mixed/strong by whole wall; same suite for all seats and replicas',
        'credit': 'one uniformly selected observed call position, one uniformly selected non-teacher action; full-East terminal delta',
        'policy': 'exact frozen stage2 feature chi/pon/pass and original oldB ordinary/strong opponent'}
    path = out / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol: raise ValueError('Resume protocol/source changed')
    else:
        check_disjoint(cfg); write(path, protocol)
    assert_frozen(protocol, cfg)
    shards = out / 'walls-private'; shards.mkdir(exist_ok=True)
    policy = None; total_rows = total_branches = 0; started = time.monotonic()
    try:
        for completed, seed in enumerate(range(cfg['training_seed_start'],
                cfg['training_seed_start'] + cfg['training_seeds_per_stage']), 1):
            shard = shards / f'{seed}.pt'; audit_path = shard.with_suffix('.audit.json')
            if shard.exists() or audit_path.exists():
                if not (args.resume and shard.exists() and audit_path.exists()):
                    raise ValueError('Incomplete existing shard; inspect before resuming')
                if digest(shard) != json.loads(audit_path.read_text())['sha256']:
                    raise ValueError('Completed shard changed')
                payload = torch.load(shard, map_location='cpu', weights_only=False)
            else:
                if policy is None:
                    write(out / 'status.json', {'status': 'loading_frozen_stage2_teacher', 'walls_complete': completed - 1})
                    policy = FeaturePointPolicy(cfg)
                    write(out / 'teacher-native-parity.json', native_teacher_parity(policy, cfg))
                controls, games, rows = [], [], []
                for seat in range(4):
                    control, _ = policy.game(seed, seat, frozen=True, replica=-1); controls.append(control)
                    for replica in range(cfg['training_replicas']):
                        game, trace = policy.game(seed, seat, training=True, replica=replica)
                        games.append(game); rows.extend(trace)
                        write(out / 'status.json', {'status': 'collecting_stage2_point_pairs',
                            'walls_complete': completed - 1, 'walls_total': cfg['training_seeds_per_stage'],
                            'suite': suite_for_seed(seed, cfg), 'seed': seed, 'seat': seat, 'replica': replica,
                            'rows_current_wall': len(rows), 'native_probability_checks': policy.checks,
                            'seconds': time.monotonic() - started, 'actor_updates': 0})
                payload = {'seed': seed, 'controls': controls, 'games': games, 'rows': rows}
                validate_wall(payload, seed, cfg)
                tmp = shard.with_suffix('.tmp'); torch.save(payload, tmp); os.replace(tmp, shard)
                write(audit_path, {'seed': seed, 'sha256': digest(shard), 'rows': len(rows),
                    'suite': suite_for_seed(seed, cfg), 'ledger_passed': True,
                    'native_probability_checks_cumulative': policy.checks})
            validate_wall(payload, seed, cfg)
            total_rows += len(payload['rows'])
            total_branches += sum(not g['reused_control_outcome'] for g in payload['games'])
            status = {'status': 'collecting_stage2_point_pairs', 'walls_complete': completed,
                'walls_total': cfg['training_seeds_per_stage'], 'rows': total_rows,
                'actual_branches': total_branches, 'actor_updates': 0, 'seconds': time.monotonic() - started}
            write(out / 'status.json', status); print(json.dumps(status), flush=True)
        assert_frozen(protocol, cfg)
        result = {'status': 'collection_complete', 'walls': cfg['training_seeds_per_stage'],
            'rows': total_rows, 'actual_branches': total_branches, 'actor_updates': 0,
            'human_test_opened': False, 'strength_improvement_proven': False}
        write(out / 'collection-result.json', result); write(out / 'status.json', result)
    finally:
        if policy is not None: policy.close()


if __name__ == '__main__': main()
