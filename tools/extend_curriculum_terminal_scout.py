"""Extend all four fixed conditional scouts to128, reusing their32 worlds."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.collect_curriculum_endgame_cases import sha


def extend(parent,out):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS')
    assert not socket.gethostname().startswith(('lo-','login'))
    out.mkdir(parents=True,exist_ok=False)
    read=lambda p:json.loads(p.read_text())
    registered=read(parent/'registered-private.json');complete=read(parent/'summary-private.json')
    if complete['status']!='conditional_terminal_scout_complete_not_training_labels' or registered['worlds_per_case']!=32:
        raise ValueError('A successful frozen32-world scout is required')
    if not all(sha(p)==h for p,h in registered['files'].items()):raise ValueError('Original scout bindings changed')
    reg=read(ROOT/'runs/curriculum-stage3-attack-recovery-v2/fit-plan/registered.json')
    sys.path.insert(0,reg['source']);sys.path.insert(0,reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from riichienv import HandEvaluator
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.policy.routing import load_model
    from jevmahjong.policy.unified import UnifiedPolicyClient
    from jevmahjong.evaluation.agents import PolicyAgent,EfficiencyAgent,CallingOpponent
    from tools.audit_curriculum_endgame_cases import restore
    from tools.audit_curriculum_history_worlds import derived_seed
    from tools.curriculum_native_decisions import decision_scope,decision_inventory,construct_world
    from tools.curriculum_decision_history import public_history
    from tools.curriculum_hierarchical_prior import sample
    from tools.curriculum_decision_rollout import finish
    from tools.curriculum_terminal_scout_stats import summarize
    checkpoint=ROOT/reg['curriculum']['reference_checkpoint'];identity=bundle(checkpoint)
    if identity['identity_sha256']!=registered['teacher_candidate_identity']:raise ValueError('Frozen v8 changed')
    rows={}
    for root in ('curriculum-stage4-placement-mortal-data-v4','curriculum-stage4-threat-prefix-v1'):
        ready=read(ROOT/'runs'/root/'data/data-ready-private.json')
        for spec in ready['files'].values():
            path=ROOT/spec['path']
            if sha(path)!=spec['sha256']:raise ValueError('Registered input data changed')
            for line in path.open():
                row=json.loads(line);rows[row['_meta']['id']]=row
    cached={case['id']:[] for case in registered['cases']}
    for line in (parent/'paired-continuation-fixtures-private.jsonl').open():
        record=json.loads(line);cached[record['id']].append(record)
    for case,summary in zip(registered['cases'],complete['cases']):
        records=cached[case['id']]
        if [r['world'] for r in records]!=list(range(32)):raise ValueError('Missing or reordered cached paired worlds')
        if summarize([r['arms'] for r in records],summary['accept_key'],summary['decline_key'])!=summary['conditional_summary']:
            raise ValueError('Cached outcomes do not reproduce the complete32 summary')
    paths=[Path(__file__),ROOT/'docs/experiments/plans/curriculum-terminal-scout-extension-20261122.md',
        parent/'registered-private.json',parent/'summary-private.json',parent/'paired-continuation-fixtures-private.jsonl']
    binding={str(p.resolve()):sha(p) for p in paths}
    def write(path,value):
        with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    write(out/'registered-private.json',dict(status='all_four_scouts128_registered_before_new_rollouts',files=binding,
        original_registration=registered,cached_worlds_per_case=32,total_worlds_per_case=128,
        new_world_indices=[32,127],all_four_original_cases_extended=True,labels_generated=False,optimizer_updates=0))
    start=time.monotonic();client=UnifiedPolicyClient(load_model(checkpoint));loading=time.monotonic()-start
    class CountClient:
        def __init__(self):self.calls=0;self.seconds=0.
        def system_one(self,**kw):
            t=time.monotonic();value=client.system_one(**kw);self.calls+=1;self.seconds+=time.monotonic()-t;return value
    counted=CountClient();cases=[];start=time.monotonic()
    with (out/'paired-continuation-fixtures-private.jsonl').open('x') as stream:
        for number,(case,old) in enumerate(zip(registered['cases'],complete['cases'])):
            row=rows[case['id']];env,obs=restore(row);seat=obs.player_id;visible=obs.to_dict()
            scope=decision_scope(env,obs,env.mjai_log[-1]);waits=lambda p,h:list(HandEvaluator(h,list(visible['melds'][p])).get_waits())
            proposal=sample(decision_inventory(obs,scope),public_history(env.mjai_log),visible['riichi_declared'],202611220500+number,128,2000000,waits)
            arms=[old['accept_key'],old['decline_key']];pairs=[r['arms'] for r in cached[case['id']]]
            for record in cached[case['id']]:stream.write(json.dumps(record)+'\n')
            before=counted.calls;t=time.monotonic()
            for world,allocation in enumerate(proposal['allocations'][32:],start=32):
                fork,_=construct_world(env,obs,env.mjai_log[-1],allocation,True);result={}
                for arm in arms:
                    agents=[CallingOpponent() if p==(seat+1)%4 else EfficiencyAgent() for p in range(4)]
                    agents[seat]=PolicyAgent(counted)
                    result[arm]=finish(fork.clone(),seat,arm,derived_seed(202611220500+number,world),agents=agents,record_events=True)
                stream.write(json.dumps(dict(id=case['id'],group=row['_meta']['group_id'],world=world,prior=proposal['method'],arms=result))+'\n');stream.flush()
                pairs.append(result)
                if len(pairs) in (64,128):print(json.dumps(dict(stage='extended_scout_prefix',slot=case['slot'],summary=summarize(pairs,*arms))),flush=True)
            result=dict(slot=case['slot'],accept_key=arms[0],decline_key=arms[1],conditional_summary=summarize(pairs,*arms),
                cached_terminal_arms=64,new_terminal_arms=192,new_model_calls=counted.calls-before,seconds=time.monotonic()-t)
            cases.append(result);print(json.dumps(dict(stage='extended_scout_case_complete',**result)),flush=True)
    if bundle(checkpoint)!=identity or not all(sha(p)==h for p,h in binding.items()):raise ValueError('Extension inputs or v8 changed')
    if not all(sha(p)==h for p,h in registered['files'].items()):raise ValueError('Original immutable scout source changed')
    summary=dict(status='all_four_conditional_scouts128_complete_not_labels',cases=cases,
        cached_terminal_arms=256,new_terminal_arms=768,total_terminal_arms=1024,new_model_calls=counted.calls,
        model_seconds=counted.seconds,rollout_seconds=time.monotonic()-start,loading_seconds=loading,
        conditional_hierarchical_prior_not_calibrated=True,ordinary_future_wins_rule_forced=True,
        teacher_not_admitted=True,labels_generated=0,optimizer_updates=0,unseen_final_grid_unused=True,raw_rollouts_private=True)
    write(out/'summary-private.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();extend(a.parent.resolve(),a.out.resolve())
