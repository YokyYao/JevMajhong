"""Revise audited MC training labels while preserving admitted positions and validation."""
import argparse
from collections import Counter,defaultdict
import copy
import json
import math
from pathlib import Path
import sqlite3
import torch
from jevmahjong.core.io import digest,write
from .base_lora_pool import complete_target,normalized_request,position,soft_target

POLICY='positive_safety_tradeoff_masked_v4'


def bind(inputs,path,expected=None):
    path=Path(path).resolve();checksum=digest(path)
    if expected is not None and checksum!=expected:raise ValueError('Registered target input changed')
    if str(path) in inputs and inputs[str(path)]!=checksum:raise ValueError('Conflicting target input receipt')
    inputs[str(path)]=checksum
    return path


def collect_revisions(cfg,parent,inputs,out):
    """Reproduce every admitted v3 target before applying the explicit v4 policy."""
    audit_file=bind(inputs,cfg['audit_receipt'],cfg['audit_receipt_sha256'])
    audit=json.loads(audit_file.read_text())
    if (audit['status']!='passed_complete_read_only_admitted_train_target_audit'
            or audit['optimizer_updates'] or audit['new_simulations']
            or audit['held_rollout_payloads_opened'] or audit['current_evaluation_game_payloads_opened']):
        raise ValueError('Complete read-only admitted training-target audit required')
    for name,h in audit['inputs'].items():bind(inputs,name,h)
    pool=Path(parent['config']['pool']);pool_result_file=bind(inputs,pool/'result.json',parent['config']['pool_result_sha256'])
    pool_result=json.loads(pool_result_file.read_text())
    bind(inputs,pool/'pool-private.sqlite',pool_result['pool_sha256'])
    extension=pool_result['config']['extension']
    if (extension['reward']!=cfg['reward'] or extension['target_temperature']!=cfg['target_temperature']
            or extension['soft_target_mode']!='outcome_only_win_risk_compatible_masked_v3'):
        raise ValueError('Same reward/temperature and legacy v3 source targets required')
    train=Path(parent['files']['train']['path']);prepared=[json.loads(x) for x in train.read_text().splitlines() if x.strip()]
    wanted={position(r,r['_meta']['group_id']):r for r in prepared if r['_meta']['label_type']=='paired_future_masked'}
    if len(wanted)!=sum(r['_meta']['label_type']=='paired_future_masked' for r in prepared):
        raise ValueError('Admitted MC training positions repeat')
    db=sqlite3.connect('file:'+str((pool/'pool-private.sqlite').resolve())+'?mode=ro',uri=True)
    positions={pos:(phase,priority,json.loads(payload)) for pos,phase,priority,payload in db.execute(
        "SELECT position,phase,priority,payload FROM cases WHERE split='train' AND label_type='paired_future_masked'") if pos in wanted}
    db.close()
    if set(positions)!=set(wanted) or len(wanted)!=audit['states_checked']:raise ValueError('Admitted MC coverage differs')
    for pos,(_,_,saved) in positions.items():
        if saved['state']!=wanted[pos]['state'] or saved['questions']!=wanted[pos]['questions']:
            raise ValueError('Admitted raw position or target differs from audited pool')
    seen=set();revisions={};phases=Counter();changed=Counter();argmax=Counter();clipped=Counter()
    for spec in pool_result['config']['sources']:
        directory=Path(spec['path']);protofile=directory/'protocol.json'
        if not protofile.exists():continue
        proto=json.loads(protofile.read_text())
        if proto.get('collection_method')!='paired_future_ev_v1':continue
        expected=audit['inputs'].get(str(protofile.resolve()))
        if expected is None:raise ValueError('Future source protocol was not in independent target audit')
        bind(inputs,protofile,expected)
        sc=proto['config']
        for shard in sorted((directory/'walls-private').glob('*.pt')):
            seed=int(shard.stem)
            if (seed-sc['training_seed_start'])%4==3:continue
            expected=audit['inputs'].get(str(shard.resolve()))
            if expected is None:raise ValueError('Training shard was not in complete independent target audit')
            bind(inputs,shard,expected);payload=torch.load(shard,map_location='cpu',weights_only=False);groups=defaultdict(list)
            for row in payload['rows']:groups[(row['episode_key'][0],row['episode_key'][1],row['request_sha256'])].append(row)
            for rows in groups.values():
                group='native:'+str(seed);pos=position(rows[0]['public_request'],group)
                if pos not in positions or pos in seen or positions[pos][1]!=spec['priority']:continue
                public=normalized_request(rows[0]['public_request'],group,'paired_future_masked',0);saved=positions[pos][2]
                if public['state']!=saved['state'] or public['questions']['action']['criteria']!=saved['questions']['action']['criteria']:continue
                for row in rows:row.update(_source_reward=sc['reward'],_source_config=sc)
                target,credits=complete_target(rows,cfg['reward'],cfg['target_temperature'],True,True)
                first=rows[0];keys=list(first['public_request']['questions']['action']['criteria']);old={k:0. for k in keys}
                for a,p in zip(first['eligible'],soft_target(credits,cfg['target_temperature'],True)):old[keys[a]]=p
                stored=saved['questions']['action']['target']
                if set(old)!=set(stored) or any(abs(old[k]-stored[k])>1e-12 for k in old):continue
                seen.add(pos);phase=positions[pos][0];phases[phase]+=1
                clipped[phase]+=sum(c['utility']>0 and c['own_win']<-1e-12 and c['dealin']<-1e-12 for c in credits)
                if max(abs(target[k]-old[k]) for k in old)>1e-12:
                    revisions[pos]=target;changed[phase]+=1
                    if max(target,key=target.get)!=max(old,key=old.get):argmax[phase]+=1
            del payload
            write(out/'status.json',dict(status='reconstructing_audited_training_targets',states_checked=len(seen),expected_states=len(wanted)))
    if seen!=set(wanted):raise ValueError('Not every admitted training target was reconstructed')
    if dict(phases)!=audit['phase_states'] or dict(changed)!=audit['affected_states'] or dict(argmax)!=audit['argmax_changes']:
        raise ValueError('Independent target audit and actual relabeling disagree')
    for phase,n in clipped.items():
        if n!=audit['clipped_actions'].get(phase+'/safety_tradeoff_positive_utility_clipped',0):
            raise ValueError('Safety-tradeoff action count differs')
    return revisions,dict(states_checked=len(seen),phase_states=dict(phases),affected_states=dict(changed),argmax_changes=dict(argmax),positive_safety_tradeoff_actions=dict(clipped))


def materialize(parent,revisions,out):
    """Change only training target/label fields; copy every other byte when possible."""
    old=Path(parent['files']['train']['path']);used=set();records=0
    with old.open('rb') as source,(out/'train.jsonl').open('xb') as target:
        for line in source:
            if not line.strip():target.write(line);continue
            r=json.loads(line);records+=1;pos=position(r,r['_meta']['group_id'])
            if pos not in revisions:target.write(line);continue
            if r['_meta']['label_type']!='paired_future_masked' or pos in used:raise ValueError('Revision is not a unique MC training position')
            q=r['questions']['action'];proposed=revisions[pos];previous=q['target']
            if (set(proposed)!=set(previous) or {k for k,v in proposed.items() if v>0}!={k for k,v in previous.items() if v>0}
                    or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in proposed.values())
                    or abs(sum(proposed.values())-1)>1e-10):raise ValueError('Revised target changed labelled support or normalization')
            q['target']=proposed;q['label']=max(proposed,key=proposed.get);used.add(pos)
            target.write((json.dumps(r,ensure_ascii=False)+'\n').encode())
    if used!=set(revisions) or records!=parent['files']['train']['records']:raise ValueError('Revisions or admitted records were omitted')
    (out/'validation.jsonl').write_bytes(Path(parent['files']['validation']['path']).read_bytes())
    return {split:dict(parent['files'][split],path=str((out/(split+'.jsonl')).resolve()),sha256=digest(out/(split+'.jsonl'))) for split in ('train','validation')}


def prepare(cfg):
    if cfg['policy']!=POLICY:raise ValueError('Explicit safety-tradeoff target policy required')
    torch.set_num_threads(4);inputs={};parent_file=bind(inputs,cfg['parent_data_receipt'],cfg['parent_data_receipt_sha256']);parent=json.loads(parent_file.read_text())
    if parent['status']!='balanced_raw_data_ready' or not parent['requires_masked_soft_target_loss']:
        raise ValueError('Completed admitted masked training data required')
    if set(parent['files'])!={'train','validation'}:raise ValueError('Only admitted train and validation files are supported')
    for record in parent['files'].values():bind(inputs,record['path'],record['sha256'])
    out=Path(cfg['out']);out.mkdir(parents=True,exist_ok=False)
    revisions,stats=collect_revisions(cfg,parent,inputs,out);files=materialize(parent,revisions,out)
    if files['validation']['sha256']!=parent['files']['validation']['sha256']:raise ValueError('Frozen full validation changed')
    for name,h in inputs.items():
        if digest(name)!=h:raise ValueError('Target derivation input changed during preparation')
    result=copy.deepcopy(parent);result.update(config=cfg,files=files,training_target_policy=POLICY,risk_tradeoff_relabel=stats,
        training_position_order_and_IDs_preserved=True,expert_labels_changed=False,validation_targets_changed=False,
        unknown_actions_are_not_negative_labels=True,reward_sign_filtering=False,raw_records_public=False,optimizer_updates=0)
    result['fixed_full_validation']=dict(path=parent['files']['validation']['path'],sha256=files['validation']['sha256'],records=files['validation']['records'],bytes_identical=True)
    write(out/'inputs-private.json',inputs);result['target_derivation_inputs_sha256']=digest(out/'inputs-private.json')
    write(out/'result.json',result);write(out/'status.json',dict(status='balanced_raw_data_ready',target_policy=POLICY,**stats))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    result=prepare(json.loads(parser.parse_args().config.read_text()));print(json.dumps({'status':result['status'],'files':result['files'],'risk_tradeoff_relabel':result['risk_tradeoff_relabel']}))


if __name__=='__main__':main()
