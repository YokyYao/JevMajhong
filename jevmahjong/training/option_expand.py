"""Complete eligible-action coverage using an audited common-future parent."""
from collections import Counter,defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import torch
from riichienv import GameRule,RiichiEnv
from jevmahjong.core.io import digest,request_hash,write
from jevmahjong.core.seeds import engine_seed
from jevmahjong.core.state import build_request
from jevmahjong.data.native_pairs import load_sources
from jevmahjong.data.future_pairs import METHOD,mean_credit,validate_rows
from .collector import NativeCollector,action_key,call_observation_digest
from .futures import replay_prefix,sample_future,finish_intervention_hand
from .provenance import snapshot,verify
from .sampling import OPERATIONS

PAIR_SELECTION='all_eligible_alternatives_against_play_v1'


def representatives(rows):
    """Choose once per visible state; never inspect future rewards."""
    chosen={}
    for row in sorted(rows,key=lambda r:tuple(r['episode_key'])):
        key=(row['episode_key'][0],row['episode_key'][1],row['request_sha256'])
        chosen.setdefault(key,row)
    return list(chosen.values())


def category_for(row,alternative):
    kinds=[c['type'] for c in row['public_request']['questions']['action']['criteria'].values()]
    return (kinds[alternative] if kinds[alternative] in OPERATIONS else
        kinds[row['teacher_action']] if kinds[row['teacher_action']] in OPERATIONS else 'ordinary')


def choose_origins(infos,limit=None):
    ordered=sorted(infos,key=lambda v:v['origin']);chosen=[]
    if limit is None:return [v['origin'] for v in ordered]
    if not isinstance(limit,int) or limit<1:raise ValueError('Positive pilot state limit required')
    def take(candidates):
        if len(chosen)>=limit:return
        for item in candidates:
            if item['origin'] not in chosen:chosen.append(item['origin']);break
    for source in sorted({v['source'] for v in ordered}):
        for split in ('train','held'):take(v for v in ordered if v['source']==source and v['split']==split)
    for kind in ('chi','pon','ankan','daiminkan','kakan'):take(v for v in ordered if kind in v['kinds'])
    while len(chosen)<min(limit,len(ordered)):take(ordered)
    return chosen


def expand_state(row,original,policy,cfg):
    origin=tuple(row['episode_key']);seed,seat,replica=origin
    game=next(g for g in original['games'] if (g['seed'],g['seat'],g['replica'])==origin)
    control=next(c for c in original['controls'] if c['seat']==seat)
    env=RiichiEnv(game_mode=cfg['game_mode'],rule=GameRule.default_mjsoul(),seed=engine_seed(seed,cfg['seed_scheme']))
    obs=replay_prefix(env,control['mjai_log'],game['intervention_step'])
    if call_observation_digest(obs)!=row['source_state_digest']:raise ValueError('Expanded state differs')
    request,actions,_=build_request(obs[seat])
    if request_hash(request)!=row['request_sha256'] or [action_key(a) for a in actions]!=row['action_keys']:
        raise ValueError('Expanded legal input changed')
    from jevany.api import validate_distribution
    answer=policy.router.system_one(**request)['answers']['action']['probabilities']
    p,_=validate_distribution(answer,list(request['questions']['action']['criteria']))
    if (not torch.allclose(torch.tensor(p),row['reference'],atol=2e-5,rtol=2e-4)
            or max(row['eligible'],key=lambda i:p[i])!=row['teacher_action']):raise ValueError('Expanded reference differs')
    first={p:policy.opponent(p,o,seat,control['suite']) for p,o in obs.items() if p!=seat}
    alternatives=[i for i in row['eligible'] if i!=row['teacher_action']]
    worlds={i:[] for i in alternatives};new_rollouts=0
    for parent in row['counterfactual_worlds']:
        fork=sample_future(env,parent['future_seed'],cfg['wall_scope'])
        if hashlib.sha256(bytes(fork.wall)).hexdigest()!=parent['future_wall_sha256']:raise ValueError('Parent future differs')
        for alternative in alternatives:
            if alternative==row['action']:branch=parent['branch']
            else:
                selected=dict(first);selected[seat]=actions[alternative]
                branch=finish_intervention_hand(fork.clone(),selected,policy,seat,control['suite'],row['hand_index'])
                new_rollouts+=1
            worlds[alternative].append({'future_seed':parent['future_seed'],'future_wall_sha256':parent['future_wall_sha256'],
                'control':parent['control'],'branch':branch})
    result=[]
    for alternative in alternatives:
        saved={k:v for k,v in row.items() if k!='counterfactual_worlds'}
        saved.update(episode_key=(*origin,alternative),origin_episode_key=origin,action=alternative,
            category=category_for(row,alternative),pair_selection=PAIR_SELECTION,counterfactual_worlds=worlds[alternative])
        means=mean_credit(worlds[alternative],saved,cfg,cfg['reward'])
        saved.update(training_credit=means,training_advantage=means['branch']['utility']-means['control']['utility'],
            control_fast_reward=means['control']['fast_reward'],branch_fast_reward=means['branch']['fast_reward'],
            fast_advantage=means['branch']['fast_reward']-means['control']['fast_reward'])
        result.append(saved)
    return result,new_rollouts


def collect(cfg,resume=False):
    out=Path(cfg['out']);out.mkdir(parents=True,exist_ok=resume)
    lock=(out/'collector.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    parent=Path(cfg['expand_from']).resolve();parent_result=json.loads((parent/'collection-result.json').read_text())
    parent_protocol=json.loads((parent/'protocol.json').read_text());verify(parent_protocol)
    if (parent_result['status']!='collection_complete' or parent_protocol['collection_method']!=METHOD
            or parent_protocol['config'].get('expand_from')):raise ValueError('Complete original future parent required')
    if any(cfg[k]!=parent_protocol['config'][k] for k in ('worlds','seed','wall_scope','fast_gamma','game_mode','seed_scheme','reward')):
        raise ValueError('Expanded alternatives must share the parent future protocol')
    from jevmahjong.data.future_pairs import future_scope
    if future_scope(cfg) != future_scope(parent_protocol['config']):
        raise ValueError('Expanded alternatives must preserve the parent decision scope')
    protocol=snapshot(cfg);protocol['collection_method']=METHOD
    if protocol['model_files']!=parent_protocol['model_files']:raise ValueError('Parent baseline model differs')
    _,hashes,sources=load_sources(parent_result['source_outputs'],protocol['model_files'],cfg['reward'])
    infos=[]
    for source_index,source in enumerate(sources):
        directory=Path(source['path'])
        for name in ('protocol.json','collection-result.json'):hashes[str(directory/name)]=digest(directory/name)
        for path in sorted((directory/'walls-private').glob('*.pt')):
            payload=torch.load(path,map_location='cpu',weights_only=False)
            for row in representatives(payload['rows']):
                infos.append({'origin':tuple(row['episode_key']),'source':source_index,'split':row['source_split'],
                    'kinds':sorted({c['type'] for c in row['public_request']['questions']['action']['criteria'].values()}),
                    'alternatives':[a for a in row['eligible'] if a!=row['teacher_action']],'old_action':row['action']})
    selected=set(choose_origins(infos,cfg.get('max_states')));state_count=len(selected)
    keys=[[*v['origin'],a] for v in infos if v['origin'] in selected for a in v['alternatives']]
    expected_new_rollouts=sum(cfg['worlds']*(a!=v['old_action']) for v in infos if v['origin'] in selected for a in v['alternatives'])
    for name in ('protocol.json','collection-result.json'):hashes[str(parent/name)]=digest(parent/name)
    protocol.update(source_files=hashes,selected_case_keys=keys,selected_origin_keys=[list(v) for v in sorted(selected)])
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('Expansion resume inputs changed')
    write(out/'protocol.json',protocol);policy=NativeCollector(cfg);completed=states_done=0;counts=Counter()
    try:
        for index,source in enumerate(sources):
            directory=Path(source['path']);parent_cfg=json.loads((directory/'protocol.json').read_text())['config']
            target=out/f'source_{index}';shards=target/'walls-private';shards.mkdir(parents=True,exist_ok=True)
            subcfg=dict(cfg,training_seed_start=source['seed_start'],training_seeds_per_stage=source['seeds'],
                source_root=parent_cfg['source_root'],parent_source_root=str(directory))
            subprotocol=dict(protocol,config=subcfg);write(target/'protocol.json',subprotocol);source_rows=0
            for seed in range(source['seed_start'],source['seed_start']+source['seeds']):
                destination=shards/f'{seed}.pt';receipt=destination.with_suffix('.audit.json')
                parent_path=directory/'walls-private'/f'{seed}.pt'
                if digest(parent_path)!=hashes[str(parent_path)]:raise ValueError('Parent shard changed')
                payload=torch.load(parent_path,map_location='cpu',weights_only=False)
                chosen=[r for r in representatives(payload['rows']) if tuple(r['episode_key']) in selected]
                if destination.exists() or receipt.exists():
                    if not resume or not destination.exists() or not receipt.exists():raise ValueError('Inspect incomplete expanded shard')
                    if digest(destination)!=json.loads(receipt.read_text())['sha256']:raise ValueError('Expanded shard changed')
                    saved=torch.load(destination,map_location='cpu',weights_only=False);validate_rows(saved,seed,subcfg,subprotocol)
                else:
                    original_path=Path(parent_cfg['source_root'])/'walls-private'/f'{seed}.pt'
                    original=torch.load(original_path,map_location='cpu',weights_only=False);rows=[]
                    for row in chosen:
                        expanded,_=expand_state(row,original,policy,cfg);rows.extend(expanded)
                        write(out/'status.json',{'status':'expanding_all_eligible_options','pairs_complete':completed+len(rows),
                            'pairs_total':len(keys),'original_unique_states':state_count,'worlds_per_state':cfg['worlds'],'current_source':index})
                    saved={'seed':seed,'rows':rows,'source_path':str(original_path),'source_sha256':hashes[str(original_path)],
                        'parent_source_path':str(parent_path),'parent_source_sha256':hashes[str(parent_path)],'collection_method':METHOD}
                    validate_rows(saved,seed,subcfg,subprotocol)
                    temporary=destination.with_suffix('.tmp');torch.save(saved,temporary);temporary.replace(destination)
                    write(receipt,{'sha256':digest(destination),'rows':len(rows)})
                counts.update(r['category'] for r in saved['rows']);completed+=len(saved['rows']);source_rows+=len(saved['rows']);states_done+=len(chosen)
            write(target/'collection-result.json',{'status':'collection_complete','method':METHOD,'walls':source['seeds'],'rows':source_rows,
                'origin_wall_groups_not_fresh_matches':True})
        verify(protocol);verify(parent_protocol)
        for path,h in hashes.items():
            if digest(Path(path))!=h:raise ValueError('Expansion input changed')
        if completed!=len(keys) or states_done!=state_count:raise ValueError('Incomplete option coverage')
        result={'status':'collection_complete','method':METHOD,'original_unique_states':state_count,'pairs':completed,
            'worlds_per_state':cfg['worlds'],'new_native_hand_rollouts':expected_new_rollouts,'reused_control_rollouts':state_count*cfg['worlds'],
            'reused_branch_rollouts':state_count*cfg['worlds'],'fresh_matches':0,'rows_by_category':dict(counts),
            'source_outputs':[str((out/f'source_{i}').resolve()) for i in range(len(sources))],'optimizer_updates':0,'human_test_opened':False}
        write(out/'collection-result.json',result);write(out/'status.json',result)
    finally:policy.close()
