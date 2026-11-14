"""Recover closed-hand inputs from existing controls, without new matches."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import torch
from riichienv import GameRule,RiichiEnv
from jevmahjong.core.io import digest,write
from jevmahjong.core.seeds import engine_seed
from jevmahjong.core.state import build_request
from jevmahjong.data.native_states import CLOSED_METHOD,INPUT_METHODS,validate
from jevmahjong.policy.closed import is_closed_request,request_eligible
from .collector import NativeCollector,call_observation_digest
from .futures import replay_step
from .provenance import snapshot,verify
from .rewards import hand_outcomes


def closed_opportunity(request):
    return is_closed_request(request) and len(request_eligible(request))>1


def replay_control(control,config,policy=None):
    """Enumerate before reading the next logged action; no outcome sampling."""
    seed,seat=control['seed'],control['seat']
    env=RiichiEnv(game_mode=config['game_mode'],rule=GameRule.default_mjsoul(),
        seed=engine_seed(seed,config['seed_scheme']))
    observations=env.reset();log=control['mjai_log'];step=0;rows=[];decisions=[]
    if env.mjai_log!=log[:len(env.mjai_log)]:raise ValueError('Closed control initial prefix differs')
    while not env.done():
        if seat in observations:
            request=build_request(observations[seat])[0]
            if closed_opportunity(request):
                # Audit mode independently enumerates all eligible locations.
                captured=[]
                if policy is not None:
                    policy.act(observations[seat],frozen=True,trace=captured)
                    if len(captured)!=1:raise ValueError('Closed opportunity feature capture missing')
                    row=captured[0]
                    if row['eligible']!=request_eligible(request):raise ValueError('Closed public and native guards differ')
                else:row={}
                hands=hand_outcomes(env.mjai_log);key=(seed,seat,len(rows))
                row.update(episode_key=key,category='ordinary',source_state_digest=call_observation_digest(observations),
                    hand_index=len(hands)-1,draws_before=hands[-1]['draws'])
                if policy is not None:row['action']=next(i for i in row['eligible'] if i!=row['teacher_action'])
                rows.append(row);decisions.append(dict(episode_key=key,seed=seed,seat=seat,replica=key[2],
                    intervention_step=step,intervention_state_digest=row['source_state_digest'],
                    hand_index=row['hand_index'],draws_before=row['draws_before']))
        previous=len(env.mjai_log);observations=replay_step(env,observations,log,step);step+=1
        if env.mjai_log[previous:]!=log[previous:len(env.mjai_log)]:raise ValueError('Closed control action prefix differs')
        if step>control['environment_steps']:raise ValueError('Closed control exceeds saved steps')
    if step!=control['environment_steps'] or env.mjai_log!=log or env.scores()[seat]!=control['score']:
        raise ValueError('Closed control terminal replay differs')
    return rows,decisions


def audit_control_coverage(payload,config,hashes):
    origin=Path(config['source_collection']).resolve()/'walls-private'/f"{payload['seed']}.pt"
    if (Path(payload['source_path']).resolve()!=origin or payload['source_sha256']!=hashes[str(origin)]
            or digest(origin)!=hashes[str(origin)]):raise ValueError('Recovered controls differ from registered source')
    parents={c['seat']:c for c in torch.load(origin,map_location='cpu',weights_only=False)['controls']}
    enumerated=[]
    for control in payload['controls']:
        old=dict(parents[control['seat']]);current=dict(control)
        old.pop('available_intervention_decisions');current.pop('available_intervention_decisions')
        if current!=old:raise ValueError('Recovered control changed original log or settlement')
        _,expected=replay_control(control,config);enumerated.extend(expected)
    if enumerated!=payload['decisions']:raise ValueError('Recovered closed opportunity coverage differs')


def collect(cfg,resume=False):
    if cfg.get('behavior_heads') or cfg['train_scopes']!=['ordinary'] or cfg['decision_filter']!='closed_discard':
        raise ValueError('Frozen ordinary closed-hand inputs required')
    source=Path(cfg['source_collection']).resolve();original=json.loads((source/'protocol.json').read_text())
    marker=json.loads((source/'collection-result.json').read_text());verify(original);sc=original['config']
    if original.get('collection_method') not in INPUT_METHODS or original.get('behavior_heads'):
        raise ValueError('Existing frozen native control source required')
    total=cfg['training_seeds_per_stage'];start=cfg['training_seed_start']
    if (marker['status']!='collection_complete' or marker['walls']!=sc['training_seeds_per_stage']
            or start!=sc['training_seed_start'] or not 1<=total<=marker['walls']):
        raise ValueError('Declared completed control prefix required')
    from jevmahjong.policy.call_baseline import identity
    if identity(cfg)!=identity(sc) or any(cfg[k]!=sc[k] for k in ('game_mode','seed_scheme','fast_gamma')):
        raise ValueError('Control rules, partition or calling baseline changed')
    protocol=snapshot(cfg)
    if protocol['model_files']!=original['model_files']:raise ValueError('Control reference models changed')
    hashes={str(source/n):digest(source/n) for n in ('protocol.json','collection-result.json')}
    for seed in range(start,start+total):
        p=source/'walls-private'/f'{seed}.pt';r=p.with_suffix('.audit.json');h=digest(p)
        if json.loads(r.read_text())['sha256']!=h:raise ValueError('Control shard changed')
        hashes[str(p)]=h;hashes[str(r)]=digest(r)
    protocol.update(collection_method=CLOSED_METHOD,source_files=hashes)
    out=Path(cfg['out']);out.mkdir(parents=True,exist_ok=resume)
    lock=(out/'collection.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=protocol:
        raise ValueError('Resume requires identical closed replay inputs')
    write(out/'protocol.json',protocol);shards=out/'walls-private';shards.mkdir(exist_ok=True)
    policy=NativeCollector(cfg);count=0;splits={'train':0,'held':0}
    try:
        for completed,seed in enumerate(range(start,start+total),1):
            target=shards/f'{seed}.pt';receipt=target.with_suffix('.audit.json')
            if target.exists() or receipt.exists():
                if not resume or not target.exists() or not receipt.exists():raise ValueError('Inspect partial closed shard')
                if digest(target)!=json.loads(receipt.read_text())['sha256']:raise ValueError('Closed shard changed')
                saved=torch.load(target,map_location='cpu',weights_only=False)
            else:
                origin=source/'walls-private'/f'{seed}.pt'
                if digest(origin)!=hashes[str(origin)]:raise ValueError('Control shard changed during replay')
                payload=torch.load(origin,map_location='cpu',weights_only=False);validate(payload,seed,sc)
                rows=[];decisions=[];controls=copy.deepcopy(payload['controls'])
                for control in controls:
                    captured,meta=replay_control(control,cfg,policy);rows.extend(captured);decisions.extend(meta)
                    control['available_intervention_decisions']=len(captured)
                    write(out/'status.json',dict(status='recovering_closed_inputs',walls_complete=completed-1,
                        walls_total=total,current_seat=control['seat'],states_complete_walls=count))
                saved=dict(collection_method=CLOSED_METHOD,seed=seed,controls=controls,rows=rows,decisions=decisions,
                    source_path=str(origin),source_sha256=hashes[str(origin)])
                validate(saved,seed,cfg);temp=target.with_suffix('.tmp');torch.save(saved,temp);temp.replace(target)
                write(receipt,dict(sha256=digest(target),rows=len(rows),controls=4))
            validate(saved,seed,cfg);count+=len(saved['rows']);splits['held' if (seed-start)%4==3 else 'train']+=len(saved['rows'])
        verify(protocol);verify(original)
        for name,h in hashes.items():
            if digest(Path(name))!=h:raise ValueError('Closed replay source changed')
        result=dict(status='collection_complete',method=CLOSED_METHOD,walls=total,rows=count,state_splits=splits,
            fresh_control_matches=0,reused_control_matches=total*4,counterfactual_rollouts=0,optimizer_updates=0,
            inputs_only_not_reward_labels=True,human_test_opened=False)
        write(out/'collection-result.json',result);write(out/'status.json',result)
    finally:policy.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args();collect(json.loads(a.config.read_text()),a.resume)


if __name__=='__main__':main()
