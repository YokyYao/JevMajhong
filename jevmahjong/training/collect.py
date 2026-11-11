"""Collect immutable whole-wall native data; no parameter updates here."""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import os
import torch
from jevmahjong.core.io import digest, write, request_hash
from jevmahjong.core.scoring import settlement_ledger
from .rewards import alternate_indices, opportunity, validate_credit
from .collector import NativeCollector
from .provenance import snapshot, verify


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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(); cfg=json.loads(args.config.read_text()); out=Path(cfg['out'])
    if cfg['training_replicas'] != len(cfg['intervention_categories']):
        raise ValueError('Branch count must match action categories')
    out.mkdir(parents=True,exist_ok=args.resume)
    lock=(out/'collector.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=snapshot(cfg)
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text()) != protocol:
            raise ValueError('Resume must preserve the original code, models and configuration')
    else: write(out/'protocol.json',protocol)
    shards=out/'walls-private'; shards.mkdir(exist_ok=True)
    policy=NativeCollector(cfg); totals=Counter(); nonzero=Counter(); count=0
    try:
        for completed,seed in enumerate(range(cfg['training_seed_start'],cfg['training_seed_start']+cfg['training_seeds_per_stage']),1):
            saved=shards/f'{seed}.pt'; audit=saved.with_suffix('.audit.json')
            if saved.exists() or audit.exists():
                if not args.resume or not saved.exists() or not audit.exists():
                    raise ValueError('Inspect incomplete existing shard before resuming')
                if digest(saved) != json.loads(audit.read_text())['sha256']:
                    raise ValueError('Completed shard changed')
                payload=torch.load(saved,map_location='cpu',weights_only=False); validate(payload,seed,cfg)
                for row in payload['rows']:
                    totals[row['category']]+=1
                    if abs(row['fast_advantage'])>1e-12:nonzero[row['category']]+=1
                count+=len(payload['rows']); continue
            controls=[]; games=[]; rows=[]
            for seat in range(4):
                control,_=policy.game(seed,seat,frozen=True,replica=-1); controls.append(control)
                for replica in range(cfg['training_replicas']):
                    game,trace=policy.game(seed,seat,training=True,replica=replica)
                    games.append(game); rows.extend(trace)
                    write(out/'status.json',{'status':'collecting','walls_complete':completed-1,
                        'walls_total':cfg['training_seeds_per_stage'],'seat':seat,'replica':replica})
            payload={'seed':seed,'controls':controls,'games':games,'rows':rows}
            validate(payload,seed,cfg)
            path=shards/f'{seed}.pt'; tmp=path.with_suffix('.tmp'); torch.save(payload,tmp);os.replace(tmp,path)
            write(path.with_suffix('.audit.json'),{'sha256':digest(path),'rows':len(rows)})
            for row in rows:
                totals[row['category']]+=1
                if abs(row['fast_advantage'])>1e-12:nonzero[row['category']]+=1
            count+=len(rows)
            write(out/'status.json',{'status':'collecting','walls_complete':completed,
                'walls_total':cfg['training_seeds_per_stage'],'rows':count,
                'rows_by_category':dict(totals),'nonzero_rewards_by_category':dict(nonzero)})
        verify(protocol)
        result={'status':'collection_complete','walls':cfg['training_seeds_per_stage'],
            'rows':count,'rows_by_category':dict(totals),'nonzero_rewards_by_category':dict(nonzero),
            'optimizer_updates':0,'human_test_opened':False}
        write(out/'collection-result.json',result);write(out/'status.json',result)
    finally:
        policy.close()

if __name__=='__main__': main()
