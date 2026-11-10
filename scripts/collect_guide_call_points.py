"""Resumable whole-wall point pairs; guide features never alter the reward."""
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
from call_policy import digest
from guide_call_collector import GuideCallInterventionPolicy
from guide_call_features import VERSION, NAMES
from head_rl import write

CODE = ['call_counterfactual.py', 'guide_call_collector.py', 'guide_call_features.py', 'scoring_audit.py',
        'single_decision_rl.py', 'head_rl.py', 'arena.py', 'semantic.py', 'seed_control.py', 'call_policy.py',
        'scripts/evaluate_call_repair.py', 'scripts/train_call_repair.py', 'scripts/collect_guide_call_points.py',
        'scripts/fit_guide_call_points.py', 'diagonal_value.py']


def validate_wall(payload, seed, cfg):
    controls, games, rows = payload['controls'], payload['games'], payload['rows']
    if len(controls) != 4 or {(g['seed'], g['seat']) for g in controls} != {(seed, p) for p in range(4)}:
        raise ValueError('Incomplete whole-wall controls')
    expected = {(seed, p, r) for p in range(4) for r in range(cfg['training_replicas'])}
    if len(games) != len(expected) or {(g['seed'], g['seat'], g['replica']) for g in games} != expected:
        raise ValueError('Incomplete or duplicate branches')
    mapping = {(g['seed'], g['seat'], g['replica']): g for g in games}
    seen = set()
    for g in controls + games:
        if not g['settlement_ledger_passed'] or sum(g['point_components'].values()) != g['score'] - 25000:
            raise ValueError('Ledger/terminal reward mismatch')
        if abs(g['reward'] - (g['score'] - 25000) / cfg['reward_scale']) > 1e-9:
            raise ValueError('Reward was reshaped')
    control_map = {g['seat']: g for g in controls}
    for g in games:
        if g['terminal_point_delta'] != g['score'] - control_map[g['seat']]['score']:
            raise ValueError('Wrong paired credit')
        if g['reused_control_outcome'] and g['terminal_point_delta'] != 0: raise ValueError('Same-action nonzero credit')
    for row in rows:
        key = tuple(row['episode_key']); g = mapping[key]
        if key in seen or g['trainable_decisions'] != 1: raise ValueError('Multiple call credits per branch')
        seen.add(key)
        if row['strategy_feature_version'] != VERSION or row['strategy_feature_names'] != list(NAMES):
            raise ValueError('Strategy schema mismatch')
        if row['strategy_features'].shape != (len(row['options']), len(NAMES)):
            raise ValueError('Strategy/action alignment mismatch')
        if not torch.isfinite(row['strategy_features']).all(): raise ValueError('Nonfinite strategy features')
        if abs(row['return'] - g['reward']) > 1e-9: raise ValueError('Trace return misaligned')
    if seen != {k for k, g in mapping.items() if g['trainable_decisions'] == 1}:
        raise ValueError('Missing call trace')


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    cfg = json.loads(args.config.read_text()); out = ROOT / cfg['out']
    out.mkdir(parents=True, exist_ok=args.resume)
    lock = (out / 'collector.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = json.loads((ROOT / cfg['reference_training_protocol']).read_text())
    protocol = {'config': cfg, 'feature_version': VERSION, 'feature_names': list(NAMES),
                'code_sha256': {n: digest(ROOT / n) for n in CODE}, 'reference_files': source['reference_files'],
                'call_head_sha256': cfg['call_head_sha256'], 'human_test_opened': False,
                'opponents': 'three frozen EfficiencyAgent; mixed/stronger-opponent development still required'}
    path = out / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol: raise ValueError('Resume protocol/source changed')
    else:
        proposed = set(range(cfg['training_seed_start'], cfg['training_seed_start'] + cfg['training_seeds_per_stage']))
        for old_path in (ROOT / 'runs').glob('*/protocol.json'):
            old = json.loads(old_path.read_text()).get('config', {})
            for first, count in [('training_seed_start','training_seeds_per_stage'),
                                 ('development_seed_start','development_seeds'), ('confirmation_seed_start','confirmation_seeds')]:
                if first in old and count in old:
                    stages = old.get('max_stages', 1) if first == 'training_seed_start' else 1
                    if proposed & set(range(old[first], old[first] + old[count] * stages)):
                        raise ValueError(f'Wall range overlaps {old_path}')
        write(path, protocol)
    for name, h in source['reference_files'].items():
        if digest(ROOT / cfg['reference'] / name) != h: raise ValueError('Frozen source changed')
    shards = out / 'walls-private'; shards.mkdir(exist_ok=True)
    policy = None; total_rows = total_branches = 0; started = time.monotonic()
    try:
        for completed, seed in enumerate(range(cfg['training_seed_start'], cfg['training_seed_start'] + cfg['training_seeds_per_stage']), 1):
            shard = shards / f'{seed}.pt'; audit_path = shard.with_suffix('.audit.json')
            if shard.exists() or audit_path.exists():
                if not (args.resume and shard.exists() and audit_path.exists()): raise ValueError('Incomplete shard; inspect before retry')
                saved = json.loads(audit_path.read_text())
                if digest(shard) != saved['sha256']: raise ValueError('Corrupt completed shard')
                payload = torch.load(shard, map_location='cpu', weights_only=False)
            else:
                if policy is None:
                    write(out / 'status.json', {'status':'loading_frozen_teacher','walls_complete':completed-1})
                    policy = GuideCallInterventionPolicy(cfg)
                controls, games, rows = [], [], []
                for seat in range(4):
                    control, _ = policy.game(seed, seat, frozen=True, replica=-1); controls.append(control)
                    for replica in range(cfg['training_replicas']):
                        game, trace = policy.game(seed, seat, training=True, replica=replica)
                        games.append(game); rows.extend(trace)
                        write(out / 'status.json', {'status':'collecting_point_pairs','walls_complete':completed-1,
                            'walls_total':cfg['training_seeds_per_stage'],'seed':seed,'seat':seat,'replica':replica,
                            'rows_current_wall':len(rows),'native_probability_checks':policy.checks,
                            'seconds':time.monotonic()-started,'actor_updates':0})
                payload = {'seed':seed,'controls':controls,'games':games,'rows':rows}
                validate_wall(payload, seed, cfg)
                tmp = shard.with_suffix('.tmp'); torch.save(payload, tmp); os.replace(tmp, shard)
                write(audit_path, {'seed':seed,'sha256':digest(shard),'rows':len(rows),
                    'native_probability_checks_cumulative':policy.checks,'ledger_passed':True})
            validate_wall(payload, seed, cfg)
            total_rows += len(payload['rows'])
            total_branches += sum(not g['reused_control_outcome'] for g in payload['games'])
            status = {'status':'collecting_point_pairs','walls_complete':completed,'walls_total':cfg['training_seeds_per_stage'],
                'rows':total_rows,'actual_branches':total_branches,'actor_updates':0,'seconds':time.monotonic()-started}
            write(out / 'status.json', status); print(json.dumps(status), flush=True)
        for name, h in protocol['code_sha256'].items():
            if digest(ROOT / name) != h: raise ValueError('Collection code changed during run')
        for name, h in source['reference_files'].items():
            if digest(ROOT / cfg['reference'] / name) != h: raise ValueError('Frozen source changed')
        if digest(ROOT / cfg['call_checkpoint']) != cfg['call_head_sha256']: raise ValueError('Call head changed')
        result = {'status':'collection_complete','walls':cfg['training_seeds_per_stage'],'rows':total_rows,
                  'actual_branches':total_branches,'actor_updates':0,'human_test_opened':False,'strength_improvement_proven':False}
        write(out / 'collection-result.json', result); write(out / 'status.json', result)
    finally:
        if policy is not None: policy.close()


if __name__ == '__main__': main()
