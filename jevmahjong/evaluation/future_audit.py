"""Reread complete paired-future data and replay its registered native inputs.

This reuses the project's rule engine and sampler; it is a data consistency
audit, not an independent engine or policy-strength evaluation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import torch
from riichienv import GameRule, RiichiEnv
from jevmahjong.core.io import digest, request_hash, write
from jevmahjong.core.seeds import engine_seed
from jevmahjong.core.state import build_request
from jevmahjong.data.native_pairs import load_sources
from jevmahjong.data.native_states import decision_records
from jevmahjong.policy.routing import make_head
from jevmahjong.training.collector import action_key, call_observation_digest
from jevmahjong.training.futures import replay_prefix, sample_future
from jevmahjong.training.provenance import snapshot, verify
from jevmahjong.training.sampling import state_key


def audit(cfg, resume=False):
    torch.set_num_threads(4)
    out = Path(cfg['out']); out.mkdir(parents=True, exist_ok=resume)
    root = Path(cfg['collection']).resolve()
    marker = json.loads((root/'collection-result.json').read_text())
    collection_protocol = json.loads((root/'protocol.json').read_text())
    if marker['status'] != 'collection_complete': raise ValueError('Complete collection required')
    verify(collection_protocol)
    protocol = snapshot(cfg)
    if protocol['model_files'] != collection_protocol['model_files']:
        raise ValueError('Audit and collection baseline models differ')
    from jevmahjong.policy.call_baseline import identity
    if identity(cfg)!=identity(collection_protocol['config']):raise ValueError('Audit calling baseline differs')
    parts, hashes, sources = load_sources(marker['source_outputs'], protocol['model_files'],
        expected_call_baseline=cfg.get('call_context_baseline'))
    for path in (root/'protocol.json', root/'collection-result.json'):
        hashes[str(path)] = digest(path)
    for source in sources:
        directory = Path(source['path'])
        for name in ('protocol.json','collection-result.json'):
            hashes[str(directory/name)] = digest(directory/name)
        for path in (directory/'walls-private').glob('*.audit.json'):
            hashes[str(path)] = digest(path)
    protocol['source_files'] = hashes
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text()) != protocol:
        raise ValueError('Audit resume inputs or source changed')
    write(out/'protocol.json', protocol)
    heads = {scope:make_head(torch.load(Path(cfg[role+'_checkpoint'])/'head.pt',
        map_location='cpu',weights_only=True),'cpu').eval().requires_grad_(False)
        for scope,role in [('ordinary','discard'),('calling','call')]}
    max_error = 0.
    with torch.no_grad():
        for rows in parts.values():
            for row in rows:
                p = heads[row['scope']](row['decide'],row['options']).softmax(-1)
                max_error = max(max_error,float((p-row['reference']).abs().max()))
                if not torch.allclose(p,row['reference'],atol=2e-5,rtol=2e-4):
                    raise ValueError('Cached reference differs from original head')
    split_pairs = {split:len(rows) for split,rows in parts.items()}
    split_states = {split:{state_key(r) for r in rows} for split,rows in parts.items()}
    if split_states['train'] & split_states['held']: raise ValueError('State crosses data split')
    if {k[0] for k in split_states['train']} & {k[0] for k in split_states['held']}:
        raise ValueError('Whole wall crosses data split')
    pair_count = sum(split_pairs.values()); replayed = walls = arm_records = 0
    seen_origins = set(); world_hashes = {}
    for source in sources:
        directory = Path(source['path'])
        subprotocol = json.loads((directory/'protocol.json').read_text()); subcfg = subprotocol['config']
        if subcfg['worlds'] != collection_protocol['config']['worlds']:
            raise ValueError('World counts differ within a collection')
        for path in sorted((directory/'walls-private').glob('*.pt')):
            payload = torch.load(path,map_location='cpu',weights_only=False)
            receipt = json.loads(path.with_suffix('.audit.json').read_text())
            if receipt['rows'] != len(payload['rows']): raise ValueError('Shard receipt count differs')
            original_path = Path(payload['source_path']).resolve()
            original = torch.load(original_path,map_location='cpu',weights_only=False)
            for row in payload['rows']:
                origin = tuple(row.get('origin_episode_key',row['episode_key']))
                identity = (source['path'],origin)
                arm_records += len(row['counterfactual_worlds'])*2
                if identity in seen_origins:
                    for world in row['counterfactual_worlds']:
                        if world_hashes[identity,world['future_seed']] != world['future_wall_sha256']:
                            raise ValueError('Shared origin changed its future')
                    continue
                seen_origins.add(identity)
                seed,seat,replica = origin
                game = next(g for g in decision_records(original) if (g['seed'],g['seat'],g['replica'])==origin)
                control = next(g for g in original['controls'] if g['seat']==seat)
                env = RiichiEnv(game_mode=subcfg['game_mode'],rule=GameRule.default_mjsoul(),
                    seed=engine_seed(seed,subcfg['seed_scheme']))
                observations = replay_prefix(env,control['mjai_log'],game['intervention_step'])
                observed_digest = call_observation_digest(observations)
                request,actions,_ = build_request(observations[seat])
                if (observed_digest != row['source_state_digest']
                        or request_hash(request) != row['request_sha256']
                        or [action_key(a) for a in actions] != row['action_keys']):
                    raise ValueError('Native prefix or decision input differs')
                hands = [list(h) for h in env.hands]
                for world in row['counterfactual_worlds']:
                    fork = sample_future(env,world['future_seed'],subcfg.get('wall_scope','live_only'))
                    actual = hashlib.sha256(bytes(fork.wall)).hexdigest()
                    if (actual != world['future_wall_sha256'] or Counter(fork.wall) != Counter(env.wall)
                            or [list(h) for h in fork.hands] != hands):
                        raise ValueError('Future wall, inventory or concealed hands differ')
                    world_hashes[identity,world['future_seed']] = actual; walls += 1
                replayed += 1
                write(out/'status.json',{'status':'auditing_existing_data','prefixes_replayed':replayed,
                    'future_walls_rederived':walls,'new_simulations':0})
    result = {'status':'passed_complete','role':'data consistency audit, not policy strength',
        'pairs':pair_count,'states':sum(map(len,split_states.values())),
        'train_pairs':split_pairs['train'],'held_pairs':split_pairs['held'],
        'train_states':len(split_states['train']),'held_states':len(split_states['held']),
        'worlds_per_state':collection_protocol['config']['worlds'],
        'prefixes_replayed':replayed,'future_walls_rederived':walls,'arm_records_audited':arm_records,
        'whole_wall_disjoint':True,'cached_head_probability_max_error':max_error,
        'new_simulations':0,'independent_strength_proven':False}
    if marker.get('pairs',marker.get('states')) != pair_count:
        raise ValueError('Collection marker pair count differs')
    if marker.get('original_unique_states',result['states']) != result['states']:
        raise ValueError('Collection marker state count differs')
    for key,value in cfg.get('expected',{}).items():
        if key not in result or result[key] != value: raise ValueError('Unexpected audit coverage: '+key)
    verify(protocol); verify(collection_protocol)
    for path,checksum in hashes.items():
        if digest(Path(path)) != checksum: raise ValueError('Audit input changed')
    write(out/'result.json',result); write(out/'status.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    print(json.dumps(audit(json.loads(args.config.read_text()),args.resume)))


if __name__=='__main__': main()
