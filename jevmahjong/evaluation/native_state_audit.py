"""Audit captured defense inputs by native prefix replay without new games."""
import argparse
import json
from pathlib import Path
import torch
from riichienv import GameRule,RiichiEnv
from jevmahjong.core.io import digest,request_hash,write
from jevmahjong.core.seeds import engine_seed
from jevmahjong.core.state import build_request
from jevmahjong.data.native_states import METHOD,validate
from jevmahjong.policy.defense import eligible_indices
from jevmahjong.policy.routing import make_head
from jevmahjong.training.collector import action_key,call_observation_digest
from jevmahjong.training.futures import replay_prefix
from jevmahjong.training.provenance import snapshot,verify
from jevmahjong.training.rewards import hand_outcomes


def audit(cfg):
    torch.set_num_threads(4)
    out=Path(cfg['out']);out.mkdir(parents=True,exist_ok=False)
    root=Path(cfg['collection']).resolve();source_file=root/'protocol.json'
    original=json.loads(source_file.read_text());source_cfg=original['config'];verify(original)
    if original['collection_method']!=METHOD:raise ValueError('Native defense input collection required')
    total=source_cfg['training_seeds_per_stage'];count=cfg.get('prefix_walls',total)
    if not isinstance(count,int) or not 1<=count<=total:raise ValueError('Invalid registered prefix length')
    hashes={str(source_file):digest(source_file)}
    marker=None
    if count==total:
        path=root/'collection-result.json';marker=json.loads(path.read_text());hashes[str(path)]=digest(path)
        if marker['status']!='collection_complete' or marker['method']!=METHOD or marker['walls']!=total:
            raise ValueError('Complete source collection required')
    protocol=snapshot(cfg)
    if protocol['model_files']!=original['model_files']:raise ValueError('Audit frozen models differ')
    head=make_head(torch.load(Path(cfg['discard_checkpoint'])/'head.pt',map_location='cpu',weights_only=True),'cpu').eval()
    checked=0;max_error=0.;splits={'train':0,'held':0}
    for seed in range(source_cfg['training_seed_start'],source_cfg['training_seed_start']+count):
        path=root/'walls-private'/f'{seed}.pt';receipt=path.with_suffix('.audit.json')
        checksum=digest(path);record=json.loads(receipt.read_text())
        if checksum!=record['sha256']:raise ValueError('Native input shard changed')
        hashes[str(path)]=checksum;hashes[str(receipt)]=digest(receipt)
        payload=torch.load(path,map_location='cpu',weights_only=False);rows=validate(payload,seed,source_cfg)
        if record['rows']!=len(rows) or record['controls']!=4:raise ValueError('Native receipt counts differ')
        meta={tuple(d['episode_key']):d for d in payload['decisions']}
        controls={c['seat']:c for c in payload['controls']}
        for row in rows:
            seat=row['episode_key'][1];decision=meta[tuple(row['episode_key'])]
            with torch.no_grad():q=head(row['decide'],row['options']).softmax(-1)
            max_error=max(max_error,float((q-row['reference']).abs().max()))
            if not torch.allclose(q,row['reference'],atol=2e-5,rtol=2e-4):raise ValueError('Frozen cached model probabilities differ')
            env=RiichiEnv(game_mode=source_cfg['game_mode'],rule=GameRule.default_mjsoul(),
                seed=engine_seed(seed,source_cfg['seed_scheme']))
            obs=replay_prefix(env,controls[seat]['mjai_log'],decision['intervention_step'])
            request,actions,features=build_request(obs[seat]);hands=hand_outcomes(env.mjai_log)
            if (call_observation_digest(obs)!=row['source_state_digest']
                    or request_hash(request)!=row['request_sha256']
                    or [action_key(a) for a in actions]!=row['action_keys']
                    or eligible_indices(obs[seat],actions,features)!=row['eligible']
                    or len(hands)-1!=row['hand_index'] or hands[-1]['draws']!=row['draws_before']):
                raise ValueError('Replayed native defense decision differs')
            checked+=1
        splits['held' if (seed-source_cfg['training_seed_start'])%4==3 else 'train']+=len(rows)
        write(out/'status.json',{'status':'auditing_native_inputs','walls_complete':seed-source_cfg['training_seed_start']+1,
            'prefixes_replayed':checked,'new_simulations':0})
    if marker and (marker['rows']!=checked or marker['state_splits']!=splits):raise ValueError('Native collection coverage differs')
    protocol['source_files']=hashes;write(out/'protocol.json',protocol)
    verify(protocol);verify(original)
    for path,h in hashes.items():
        if digest(Path(path))!=h:raise ValueError('Native audit input changed')
    result={'status':'passed_complete' if count==total else 'passed_registered_prefix',
        'walls':count,'control_matches_audited':count*4,'states':checked,'state_splits':splits,
        'prefixes_replayed':checked,'cached_head_probability_max_error':max_error,'whole_wall_disjoint':True,
        'new_simulations':0,'reward_labels_available':False,'independent_strength_proven':False}
    write(out/'result.json',result);write(out/'status.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(audit(json.loads(args.config.read_text()))))


if __name__=='__main__':main()
