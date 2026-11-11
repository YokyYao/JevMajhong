"""Register and collect immutable native same-hand fast-win contrasts."""
import argparse
from collections import Counter
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
from fast_win import alternate_indices, opportunity, validate_credit
from fast_win_collector import FastWinCollector
from head_rl import write
from scoring_audit import settlement_ledger
from scripts.collect_feature_points import CODE as PARENT_CODE, check_disjoint, native_teacher_parity

CODE = sorted(set(PARENT_CODE + ['fast_win.py', 'fast_win_collector.py',
    'scripts/collect_fast_win.py', 'scripts/fit_fast_win.py', 'point_pairwise.py',
    'stage2_ablation.py', 'test_fast_win.py', 'scripts/collect_fast_win_v2.py',
    'test_fast_win_preflight.py']))


def same_distribution(first, second):
    a = first['answers']['action']['probabilities']
    b = second['answers']['action']['probabilities']
    return list(a) == list(b) and a == b


def verify_source(protocol, cfg):
    if protocol['config'] != cfg:
        raise ValueError('Registered fast-win config changed')
    for name, h in protocol['code_sha256'].items():
        if digest(ROOT / name) != h:
            raise ValueError('Frozen fast-win source changed: ' + name)
    for name, h in protocol['reference_files'].items():
        if digest(ROOT / cfg['reference'] / name) != h:
            raise ValueError('Ordinary source changed')
    if completed_checkpoint_files(ROOT / cfg['teacher_checkpoint'], cfg['teacher_step']) != protocol['teacher_files']:
        raise ValueError('Stage2 source changed')


def validate(payload, seed, cfg):
    controls = {g['seat']: g for g in payload['controls']}
    if len(controls) != 4 or set(controls) != set(range(4)):
        raise ValueError('Incomplete controls')
    games = {(g['seed'], g['seat'], g['replica']): g for g in payload['games']}
    expected = {(seed, p, r) for p in range(4) for r in range(cfg['training_replicas'])}
    if len(games) != len(payload['games']) or set(games) != expected:
        raise ValueError('Incomplete or duplicate branch grid')
    for g in payload['controls'] + payload['games']:
        ledger = settlement_ledger(g['mjai_log'], [25000 + sum(c.values()) for c in g['all_point_components']])
        if (g['seed'] != seed or ledger['final_scores'][g['seat']] != g['score']
                or ledger['components'][g['seat']] != g['point_components'] or not g['settlement_ledger_passed']):
            raise ValueError('Actual native log and point ledger differ')
    seen = set()
    for row in payload['rows']:
        key = tuple(row['episode_key'])
        if key in seen or key not in games:
            raise ValueError('Duplicate or foreign trace')
        seen.add(key); g = games[key]; category = cfg['intervention_categories'][g['replica']]
        if g['reused_control_outcome'] or g['trainable_decisions'] != 1:
            raise ValueError('Action trace must correspond to one actual branch')
        if row['category'] != category or not opportunity(row, category):
            raise ValueError('Wrong registered opportunity stratum')
        alternatives = alternate_indices(row, category)
        if row['action'] not in alternatives or abs(row['alternative_probability'] - 1 / len(alternatives)) > 1e-12:
            raise ValueError('Wrong stratified action sampling')
        if row['request_sha256'] != request_hash(row['public_request']):
            raise ValueError('Public input changed')
        validate_credit(row, g, controls[g['seat']]['mjai_log'], cfg['fast_gamma'])
    if seen != {k for k, g in games.items() if g['trainable_decisions'] == 1}:
        raise ValueError('Branch trace coverage differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); cfg = json.loads(args.config.read_text()); out = ROOT / cfg['out']
    if cfg['training_replicas'] != len(cfg['intervention_categories']):
        raise ValueError('One registered branch per category required')
    if cfg.get('smoke_config'):
        smoke = json.loads((ROOT / json.loads((ROOT / cfg['smoke_config']).read_text())['out'] / 'collection-result.json').read_text())
        if smoke['status'] != 'collection_complete' or not smoke['mechanism_checks_passed']:
            raise ValueError('Completed mechanism smoke required')
    out.mkdir(parents=True, exist_ok=args.resume)
    lock = (out / 'collector.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = json.loads((ROOT / cfg['reference_training_protocol']).read_text())
    protocol = {'config': cfg, 'code_sha256': {n: digest(ROOT / n) for n in CODE},
        'reference_files': source['reference_files'],
        'teacher_files': completed_checkpoint_files(ROOT / cfg['teacher_checkpoint'], cfg['teacher_step']),
        'training_reward': 'self wins the intervention hand:gamma**remaining_table_draws; opponent win/draw:0',
        'net_points_are_diagnostic_only': True, 'actor_updates_during_collection': 0,
        'human_test_opened': False}
    path = out / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError('Resume source or protocol changed')
    else:
        check_disjoint(cfg); write(path, protocol)
    verify_source(protocol, cfg)
    shards = out / 'walls-private'; shards.mkdir(exist_ok=True)
    policy = None; counts = Counter(); contrasts = Counter(); total = 0; started = time.monotonic()
    try:
        for number, seed in enumerate(range(cfg['training_seed_start'], cfg['training_seed_start'] + cfg['training_seeds_per_stage']), 1):
            shard = shards / f'{seed}.pt'; audit_path = shard.with_suffix('.audit.json')
            if shard.exists() or audit_path.exists():
                if not args.resume or not shard.exists() or not audit_path.exists():
                    raise ValueError('Inspect incomplete existing shard before resume')
                if digest(shard) != json.loads(audit_path.read_text())['sha256']:
                    raise ValueError('Completed shard changed')
                payload = torch.load(shard, map_location='cpu', weights_only=False)
            else:
                if policy is None:
                    write(out / 'status.json', {'status': 'loading_frozen_hybrid', 'walls_complete': number - 1})
                    policy = FastWinCollector(cfg)
                    write(out / 'teacher-native-parity.json', native_teacher_parity(policy, cfg))
                    if cfg.get('kan_probe_cases'):
                        import copy
                        from fast_win import FastHeadClient
                        probes = ROOT / cfg['kan_probe_cases']
                        if digest(probes) != cfg['kan_probe_sha256']:
                            raise ValueError('Native kan probes changed')
                        heads = {'ordinary': copy.deepcopy(policy.ordinary.runtime.model.head),
                            'calling': copy.deepcopy(policy.model.head)}
                        router = FastHeadClient(policy.ordinary, policy.client, heads)
                        checked = Counter()
                        for line in probes.open():
                            probe = json.loads(line); request = probe['request']
                            before = policy.router.system_one(**request)
                            after = router.system_one(**request)
                            if not same_distribution(before, after):
                                raise ValueError('Zero-update kan head replacement changes native probabilities')
                            checked[probe['kan_type']] += 1
                        if any(checked[k] < 1 for k in ('ankan', 'daiminkan', 'kakan')):
                            raise ValueError('Three kan probe coverage missing')
                        write(out / 'native-kan-head-parity.json', {'checked': dict(checked),
                            'zero_update_probabilities_identical': True, 'diagnostic_probes_not_training_labels': True})
                        del router, heads
                controls, games, rows = [], [], []
                for seat in range(4):
                    control, _ = policy.game(seed, seat, frozen=True, replica=-1); controls.append(control)
                    for replica in range(cfg['training_replicas']):
                        game, trace = policy.game(seed, seat, training=True, replica=replica)
                        games.append(game); rows.extend(trace)
                        write(out / 'status.json', {'status': 'collecting_fast_win_pairs',
                            'walls_complete': number - 1, 'walls_total': cfg['training_seeds_per_stage'],
                            'seed': seed, 'seat': seat, 'replica': replica,
                            'actual_rows_current_wall': len(rows), 'seconds': time.monotonic() - started})
                payload = {'seed': seed, 'controls': controls, 'games': games, 'rows': rows}
                validate(payload, seed, cfg)
                tmp = shard.with_suffix('.tmp'); torch.save(payload, tmp); os.replace(tmp, shard)
                write(audit_path, {'sha256': digest(shard), 'rows': len(rows), 'ledger_and_reward_checked': True})
            validate(payload, seed, cfg)
            for row in payload['rows']:
                counts[row['category']] += 1
                if abs(row['fast_advantage']) > 1e-12:
                    contrasts[row['category']] += 1
            total += len(payload['rows'])
            write(out / 'status.json', {'status': 'collecting_fast_win_pairs', 'walls_complete': number,
                'walls_total': cfg['training_seeds_per_stage'], 'rows': total,
                'rows_by_category': dict(counts), 'nonzero_rewards_by_category': dict(contrasts),
                'seconds': time.monotonic() - started})
        verify_source(protocol, cfg)
        result = {'status': 'collection_complete', 'walls': cfg['training_seeds_per_stage'],
            'rows': total, 'rows_by_category': dict(counts), 'nonzero_rewards_by_category': dict(contrasts),
            'mechanism_checks_passed': True, 'actor_updates': 0, 'human_test_opened': False,
            'strength_improvement_proven': False}
        if not counts['calling'] or not counts['ordinary']:
            raise ValueError('Missing ordinary or call training scope')
        write(out / 'collection-result.json', result); write(out / 'status.json', result)
        print(json.dumps(result), flush=True)
    finally:
        if policy is not None:
            policy.close()


if __name__ == '__main__':
    main()
