"""Fresh outcome-blind source games and paired64 terminal supervision."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy
import gc
import json
import multiprocessing
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def validate_config(c):
    expected={'schema':'endgame-targeted-refresh64-v1','out':'runs/endgame-targeted-refresh-v1',
        'parent_step':128,'train_seed_start':2026130000,'train_source_groups':1536,
        'monitor_seed_start':2026133000,'monitor_source_groups':512,'source_workers':2,
        'worlds_per_action':64,'replication_blocks':2,'max_legal_actions':768,'label_workers':2,
        'terminal_step_cap':30000,'label_temperature':.1,'uncertainty_shrink_z':1.96,
        'raw_advantage_fraction':.5,'prior_exploration':.1,'future_hand_rng_resampled':True,
        'current_concealed_hands':'fixed_native_source_not_public_posterior'}
    quotas={'train':{'discard':16,'chi':4,'pon':4,'kan':4,'win':4,'third':8,'hold':8},
        'monitor':{'discard':2,'chi':1,'pon':1,'kan':1,'win':1,'third':1,'hold':1}}
    if (any(c.get(k)!=v for k,v in expected.items()) or c.get('manifest_quotas')!=quotas
            or c.get('continuation_policies')!=['formal_v8_original','parent128_endgame']
            or c.get('opponent_profiles')!=['one_caller','two_callers']
            or any(c.get(k) is not False for k in ('raw_publication','independent_confirmation','export_registered'))
            or abs(sum(c['loss_mass'].values())-1)>1e-12 or sum(c['batch_quotas'].values())!=48
            or set(c['batch_quotas'])!=set(c['loss_mass'])):raise ValueError('Unregistered targeted64 recipe')


def old_assets(c):
    from tools.continue_endgame_fourth_lora import assets
    return assets(c)


def verify_registration(c,reg):
    from tools.train_endgame_fourth_chase import verify
    validate_config(c)
    if reg['configuration']!=c or reg['ci']['conclusion']!='success':raise ValueError('Public source registration missing')
    verify(reg)


def request(obs):
    from jevmahjong.core.state import build_request
    from jevmahjong.endgame.native_public import add_native_context
    from jevmahjong.endgame.context import annotate_record,TRAILING_INPUT_VERSION
    raw,actions,_=build_request(obs);raw['state']['hand_number']=int(obs.kyoku_index)+1
    return annotate_record(add_native_context(raw,obs,actions),TRAILING_INPUT_VERSION),actions


def source_match(seed,split):
    from riichienv import RiichiEnv,GameRule,ActionType
    from jevmahjong.core.seeds import engine_seed
    from jevmahjong.core.state import legal_candidates
    from jevmahjong.evaluation.agents import CallingOpponent,EfficiencyAgent
    from jevmahjong.endgame.coverage import DamaAgent
    from jevmahjong.endgame.collect import digest
    from jevmahjong.endgame.context import placements
    from jevmahjong.endgame.ledger import hand_score_history
    from jevmahjong.core.scoring import settlement_ledger
    from jevmahjong.endgame.targeted_refresh import category,public_cell
    from jevmahjong.endgame.fourth_chase import fourth_chase_action_keys
    env=RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=engine_seed(seed,'spaced-v1'))
    observations=env.reset();agents=[EfficiencyAgent(),DamaAgent(),CallingOpponent()];profile=seed%3;pools={}
    for step in range(30000):
        if env.done():break
        if env.round_wind>0 or env.kyoku_idx==3:
            for seat,obs in observations.items():
                actions=legal_candidates(obs)
                if len(actions)<2:continue
                payload,_=request(obs);rank=placements(payload['state']['scores'])[seat]
                kind=category(payload,rank)
                if kind is None or len(fourth_chase_action_keys(payload))<2:continue
                cell=public_cell(payload,kind,rank);sha=digest(payload)
                row=dict(seed=seed,split=split,step=step,seat=seat,rank=rank,category=kind,cell=cell,
                    request=payload,request_sha256=sha,selection_key=digest([seed,step,seat,sha,'targeted20-source']))
                row['id']='endgame:fresh20:'+digest([seed,step,seat,sha]);row['group_id']='fresh20:'+str(seed)
                if cell not in pools or row['selection_key']<pools[cell]['selection_key']:pools[cell]=row
        chosen={}
        for seat,obs in observations.items():
            actions=legal_candidates(obs);wins=[a for a in actions if a.action_type in (ActionType.RON,ActionType.TSUMO)]
            kans=[a for a in actions if a.action_type in (ActionType.ANKAN,ActionType.DAIMINKAN,ActionType.KAKAN)]
            if wins:action=wins[0]
            elif kans and int(digest([seed,step,seat,'source-kan'])[:8],16)%4==0:action=kans[0]
            else:
                caller=(seat-seed)%4 < (2 if profile==2 else 1)
                action=agents[2 if caller else 1 if profile else 0].act(obs)
            chosen[seat]=action
        observations=env.step(chosen)
    if not env.done():raise RuntimeError('Source match exceeded step cap')
    history=hand_score_history(env.mjai_log)
    if history['final_scores']!=list(env.scores()):raise ValueError('Source hand ledger mismatch')
    return dict(seed=seed,split=split,source_profile=profile,selected=list(pools.values()),mjai_log=env.mjai_log,
                final_scores=list(env.scores()),ledger=settlement_ledger(env.mjai_log,env.scores()))


def _collect_one(out,seed,split):
    from jevmahjong.endgame.collect import write
    path=Path(out)/'sources'/f'{seed}.json'
    if path.exists():raise ValueError('Unexpected pre-existing fresh source')
    value=source_match(seed,split);write(path,value)
    return dict(seed=seed,split=split,counts=dict(Counter(r['category'] for r in value['selected'])))


def collect(c,reg):
    verify_registration(c,reg)
    from jevmahjong.endgame.collect import write
    from jevmahjong.endgame.prepare import checksum
    from jevmahjong.endgame.futures import bundle_identity
    out=Path(c['out'])
    if (out/'collection-protocol-private.json').exists():raise ValueError('Existing collection; inspect before retrying')
    ready,formal,train,val,fourth,ck=old_assets(c)
    train_seeds=set(range(c['train_seed_start'],c['train_seed_start']+c['train_source_groups']))
    monitor_seeds=set(range(c['monitor_seed_start'],c['monitor_seed_start']+c['monitor_source_groups']))
    if train_seeds&monitor_seeds or min(train_seeds|monitor_seeds)<=2026129999:raise ValueError('Fresh blocks overlap reserved previous sources')
    proof=dict(reg,formal_identity=bundle_identity(formal),parent_identity=bundle_identity(ck[128]),
        frozen_old_data_sha256=ready['data_sha256'],train_seed_range=[min(train_seeds),max(train_seeds)],
        monitor_seed_range=[min(monitor_seeds),max(monitor_seeds)])
    write(out/'collection-protocol-private.json',proof)
    complete=[];executor=ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'));tasks=[]
    try:
        for split,seeds in (('train',train_seeds),('monitor',monitor_seeds)):
            tasks.extend(executor.submit(_collect_one,str(out),seed,split) for seed in sorted(seeds))
        for future in as_completed(tasks):
            complete.append(future.result())
            if len(complete)%16==0:
                status=dict(status='collecting_fresh_targeted_sources',complete=len(complete),total=2048)
                write(out/'collection-status-private.json',status);print(json.dumps(status),flush=True)
    finally:
        for task in tasks:task.cancel()
        executor.shutdown(wait=True,cancel_futures=True)
    from tools.train_endgame_fourth_chase import verify
    verify(reg)
    write(out/'collection-summary-private.json',dict(status='fresh_targeted_sources_complete',groups=len(complete),
        groups_by_split=dict(Counter(x['split'] for x in complete)),source_files_sha256={str(s):checksum(out/'sources'/f'{s}.json') for s in sorted(train_seeds|monitor_seeds)}))


def manifest(c,reg):
    verify_registration(c,reg)
    from jevmahjong.endgame.targeted_refresh import select_manifest
    from jevmahjong.endgame.collect import write,digest
    from jevmahjong.endgame.prepare import checksum
    out=Path(c['out']);receipt=json.loads((out/'collection-summary-private.json').read_text())
    if receipt['status']!='fresh_targeted_sources_complete' or receipt['groups']!=2048:raise ValueError('Collection incomplete')
    if (out/'manifest-private.json').exists():raise ValueError('Existing manifest')
    rows=[]
    for seed,sha in receipt['source_files_sha256'].items():
        path=out/'sources'/f'{seed}.json'
        if checksum(path)!=sha:raise ValueError('Source file changed')
        source=json.loads(path.read_text());rows.extend(source['selected'])
    selected=select_manifest(rows,c['manifest_quotas'])
    actions=sum(len(r['request']['questions']['action']['criteria']) for r in selected)
    if actions>768:raise ValueError('All-action budget exceeded; do not prune by future labels')
    bins=[[],[]];cost=[0,0]
    for row in sorted(selected,key=lambda r:(-len(r['request']['questions']['action']['criteria']),r['id'])):
        worker=min(range(2),key=lambda i:(cost[i],i));bins[worker].append(row['id']);cost[worker]+=len(row['request']['questions']['action']['criteria'])
    frozen=dict(status='fresh_targeted_manifest_frozen',positions=selected,positions_sha256=digest(selected),
        worker_ids=bins,worker_actions=cost,legal_actions=actions,terminal_rollouts=actions*64,
        selected_source_sha256={str(r['seed']):receipt['source_files_sha256'][str(r['seed'])] for r in selected})
    write(out/'manifest-private.json',frozen)
    write(out/'manifest-summary-private.json',{k:v for k,v in frozen.items() if k not in ('positions','worker_ids','selected_source_sha256')})
    print(json.dumps({k:v for k,v in frozen.items() if k not in ('positions','worker_ids','selected_source_sha256')}),flush=True)


def replay(c,row,frozen):
    from riichienv import RiichiEnv,GameRule
    from jevmahjong.core.seeds import engine_seed
    from jevmahjong.training.futures import replay_prefix
    from jevmahjong.endgame.prepare import checksum
    from jevmahjong.endgame.collect import digest
    path=Path(c['out'])/'sources'/f"{row['seed']}.json"
    if checksum(path)!=frozen['selected_source_sha256'][str(row['seed'])]:raise ValueError('Selected source changed')
    source=json.loads(path.read_text());env=RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=engine_seed(row['seed'],'spaced-v1'))
    obs=replay_prefix(env,source['mjai_log'],row['step'])[row['seat']]
    actual,actions=request(obs)
    if actual!=row['request'] or digest(actual)!=row['request_sha256']:raise ValueError('Replayed public request changed')
    return env,actions


def finish(env,initial,seat,seed,hero,opponents,cap):
    from jevmahjong.evaluation.agents import CallingOpponent,EfficiencyAgent
    from jevmahjong.endgame.context import placements
    from jevmahjong.endgame.ledger import hand_score_history
    from jevmahjong.core.scoring import settlement_ledger
    ordinary,caller=EfficiencyAgent(),CallingOpponent();obs=env.step(initial)
    for step in range(cap):
        if env.done():break
        chosen={p:(hero if p==seat else caller if (p-seed)%4<(1+opponents) else ordinary).act(o) for p,o in obs.items()}
        obs=env.step(chosen)
    if not env.done():raise RuntimeError('Counterfactual did not reach native match end')
    scores=list(env.scores());ranks=list(env.ranks());history=hand_score_history(env.mjai_log)
    if ranks!=placements(scores) or history['final_scores']!=scores:raise ValueError('Terminal scores/ranks/history mismatch')
    settlement_ledger(env.mjai_log,scores)
    return dict(scores=scores,ranks=ranks,ledger_passed=True,completed=True,steps=step+1)


def worker(c,reg,index):
    verify_registration(c,reg)
    if index not in (0,1):raise ValueError('Expected worker0/1')
    from jevmahjong.endgame.collect import write,digest
    from jevmahjong.endgame.prepare import checksum
    from jevmahjong.endgame.targeted_refresh import world_design,seeded_future,reliable_target
    from jevmahjong.endgame.fourth_chase import fourth_chase_action_keys
    from jevmahjong.endgame.reference import inference_request,restrict_distribution
    from jevmahjong.core.state import signature,legal_candidates
    from jevmahjong.evaluation.agents import CallingOpponent,EfficiencyAgent
    from jevmahjong.endgame.futures import bundle_identity
    from jevmahjong.endgame.policy import ModelAgent
    from jevmahjong.endgame.fourth_policy import FourthChaseAgent
    ready,formal,oldtrain,val,fourth,ck=old_assets(c);sys.path.insert(0,ready['source_config']['upstream_source'])
    from jevmahjong.policy.routing import load_model
    from jevany.api import validate_distribution
    out=Path(c['out']);frozen=json.loads((out/'manifest-private.json').read_text());manifest_sha=checksum(out/'manifest-private.json')
    clients=[load_model(formal),load_model(ck[128])]
    for client in clients:
        client.runtime.model.eval().requires_grad_(False)
        if client.runtime.inference_options.prefix_cache_size!=0:raise ValueError('Inference cache must stay disabled')
    agents=[ModelAgent(clients[0],placement_aware=False),FourthChaseAgent(clients[1])]
    identities=[bundle_identity(formal),bundle_identity(ck[128])];ordinary,caller=EfficiencyAgent(),CallingOpponent();start=time.monotonic()
    chosen=[r for r in frozen['positions'] if r['id'] in set(frozen['worker_ids'][index])]
    completed=0
    for row in chosen:
        filename=row['id'].rsplit(':',1)[1]+'.json';path=out/'labels'/filename
        if path.exists():raise ValueError('Unexpected existing completed label; inspect explicit resume')
        env,actions=replay(c,row,frozen);signatures=[signature(a) for a in actions];keys=list(row['request']['questions']['action']['criteria'])
        answer=clients[1].system_one(**inference_request(row['request']))
        values,_=validate_distribution(answer['answers']['action']['probabilities'],keys);prior=dict(zip(keys,values))
        outcomes={k:[] for k in keys}
        for world in range(64):
            design=world_design(world);wall_seed=int(digest([row['id'],world,'wall64'])[:16],16)
            future_seed=int(digest([row['id'],world,'future-hands64'])[:16],16)
            sampled=seeded_future(env,wall_seed,future_seed)
            initial={p:o for p,o in sampled.clone().get_observations().items() if o.legal_actions()}
            others={p:(caller if (p-row['seed'])%4<(1+design['opponents']) else ordinary).act(o) for p,o in initial.items() if p!=row['seat']}
            for key,sig in zip(keys,signatures):
                branch=sampled.clone();obs=branch.get_observations()[row['seat']]
                menu={signature(a):a for a in legal_candidates(obs)}
                if set(menu)!=set(signatures):raise ValueError('Paired hidden sampling changed legal action support')
                value=finish(branch,{**others,row['seat']:menu[sig]},row['seat'],row['seed'],agents[design['policy']],design['opponents'],30000)
                outcomes[key].append(dict(value,world=world,**design))
            status=dict(status='labeling_fresh64',worker=index,positions_complete=completed,positions_total=len(chosen),
                current_category=row['category'],world_complete=world+1,current_actions=len(keys),elapsed_seconds=time.monotonic()-start)
            write(out/f'worker-{index}-status-private.json',status)
            if (world+1)%8==0:write(out/'partials'/filename,dict(manifest_sha256=manifest_sha,row_id=row['id'],worlds_complete=world+1,outcomes=outcomes))
        utility={k:[int(o['ranks'][row['seat']]<=2) for o in v] for k,v in outcomes.items()}
        target,uncertainty=reliable_target(prior,utility,fourth_chase_action_keys(row['request']))
        record=copy.deepcopy(row['request']);q=record['questions']['action']
        q.update(target=target,label=max(target,key=target.get),src='new_'+('call' if row['category'] in ('chi','pon','kan') else row['category']))
        record['_meta']=dict(id=row['id'],group_id=row['group_id'],split=row['split'],source='targeted_fresh64',variant='clean',
            source_initial_placement=row['rank'],category=row['category'],fresh_top2={k:sum(v)/64 for k,v in utility.items()},
            fresh_escape={k:sum(o['ranks'][row['seat']]<4 for o in outcomes[k])/64 for k in keys},
            label_scope='fixed_current_concealed_hands_random_wall_and_future_hands',paired_uncertainty=uncertainty)
        label=dict(status='fresh_complete_paired64',manifest_sha256=manifest_sha,source=row,record=record,
            prior128=prior,outcomes=outcomes,policy_identities=identities,worlds_per_action=64,all_actions_completed=True)
        write(path,label);completed+=1
        print(json.dumps(dict(worker=index,completed=completed,total=len(chosen),category=row['category'],seconds=time.monotonic()-start)),flush=True)
    if checksum(out/'manifest-private.json')!=manifest_sha:raise ValueError('Frozen manifest changed')
    verify_registration(c,reg)
    write(out/f'worker-{index}-complete-private.json',dict(status='fresh64_worker_complete',worker=index,positions=completed,manifest_sha256=manifest_sha))


def assemble(c,reg):
    verify_registration(c,reg)
    from jevmahjong.endgame.collect import write
    from jevmahjong.endgame.prepare import checksum,jsonl
    from jevmahjong.endgame.context import placements
    from jevmahjong.endgame.reference import inference_request
    from jevmahjong.endgame.targeted_refresh import reliable_target
    from jevmahjong.endgame.fourth_chase import fourth_chase_action_keys
    out=Path(c['out']);frozen=json.loads((out/'manifest-private.json').read_text());sha=checksum(out/'manifest-private.json')
    for i in (0,1):
        r=json.loads((out/f'worker-{i}-complete-private.json').read_text())
        if r['manifest_sha256']!=sha or r['positions']!=len(frozen['worker_ids'][i]):raise ValueError('Label worker incomplete')
    ready,formal,oldtrain,val,fourth,ck=old_assets(c);splits={'train':[],'monitor':[]};hashes={}
    for row in frozen['positions']:
        path=out/'labels'/(row['id'].rsplit(':',1)[1]+'.json');label=json.loads(path.read_text())
        if label['status']!='fresh_complete_paired64' or label['source']!=row or label['manifest_sha256']!=sha:raise ValueError('Label provenance differs')
        keys=list(row['request']['questions']['action']['criteria']);utility={}
        for key in keys:
            values=label['outcomes'][key]
            if len(values)!=64 or [o['world'] for o in values]!=list(range(64)):raise ValueError('Incomplete paired worlds')
            if any(not o['ledger_passed'] or not o['completed'] or placements(o['scores'])!=o['ranks'] for o in values):raise ValueError('Invalid terminal ranks/ledger')
            utility[key]=[int(o['ranks'][row['seat']]<=2) for o in values]
        target,details=reliable_target(label['prior128'],utility,fourth_chase_action_keys(row['request']))
        record=label['record']
        if (record['questions']['action']['target']!=target or record['_meta']['paired_uncertainty']!=details
                or inference_request(record)!=inference_request(row['request'])):raise ValueError('Target reproducibility/input leakage failure')
        splits[row['split']].append(record);hashes[row['id']]=checksum(path)
    if (len(splits['train']),len(splits['monitor']))!=(48,8):raise ValueError('Fixed fresh48/8 membership changed')
    for row in oldtrain:
        q=row['questions']['action'];src=q['src']
        kind=('old_replay' if src=='replay_reference' else 'old_leading' if src=='replay_reference_leading'
              else 'old_fourth' if 'fourth' in src else 'old_hold' if src=='endgame_terminal_hold' else 'old_third')
        row=copy.deepcopy(row);row['questions']['action']['src']=kind;splits['train'].append(row)
    if len(splits['train'])!=874:raise ValueError('Existing826 plus fresh48 expected')
    jsonl(out/'train.jsonl',splits['train']);jsonl(out/'monitor.jsonl',splits['monitor'])
    # Preserve fixed186 validation byte for byte, not a regenerated representation.
    import shutil
    shutil.copyfile(Path(c['prepared_data_run'])/'validation.jsonl',out/'validation.jsonl')
    summary=dict(status='fresh64_verified_training_data_ready',train=874,fresh_train=48,fresh_monitor=8,
        old_positive=826,old_validation=186,loss_mass=c['loss_mass'],source_counts=dict(Counter(r['questions']['action']['src'] for r in splits['train'])),
        labels_sha256=hashes,data_sha256={n:checksum(out/(n+'.jsonl')) for n in ('train','monitor','validation')},
        validation_unchanged=checksum(out/'validation.jsonl')==checksum(Path(c['prepared_data_run'])/'validation.jsonl'),
        raw_publication=False,optimizer_updates=0)
    write(out/'data-ready-private.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='labels_sha256'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['collect','manifest','worker','assemble'])
    p.add_argument('--config',type=Path,required=True);p.add_argument('--registration',type=Path,required=True);p.add_argument('--worker-index',type=int)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith(('lo-','login')):raise RuntimeError('Scheduled GPU allocation required')
    from tools.prepare_native_context_replay import allocation_metadata
    d=subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID'],'--oneliner'],text=True)
    print(json.dumps(dict(node=socket.gethostname(),allocation=allocation_metadata(d,socket.gethostname()))),flush=True)
    c=json.loads(a.config.read_text());reg=json.loads(a.registration.read_text())
    if a.stage=='worker':worker(c,reg,a.worker_index)
    else:{'collect':collect,'manifest':manifest,'assemble':assemble}[a.stage](c,reg)
