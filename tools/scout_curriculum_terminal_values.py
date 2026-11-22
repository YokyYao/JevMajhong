"""Hard-score terminal scout under frozen conventional v8 continuation."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.collect_curriculum_endgame_cases import sha


def benchmark(out):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS')
    assert not socket.gethostname().startswith(('lo-','login'))
    out.mkdir(parents=True,exist_ok=False)
    read=lambda path:json.loads(path.read_text())
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
    from tools.curriculum_hierarchical_prior import sample,METHOD
    from tools.curriculum_decision_rollout import finish
    from tools.curriculum_terminal_scout_stats import summarize
    parent=ROOT/'runs/curriculum-live-worlds-v2/matrix-audit-v1'
    matrix=read(parent/'registered-private.json');passed=read(parent/'summary-private.json')
    if passed['status']!='live_decision_native_matrix_complete':raise ValueError('Complete native scope/threat preflight required')
    roots=[ROOT/'runs/curriculum-stage4-placement-mortal-data-v4/data',ROOT/'runs/curriculum-stage4-threat-prefix-v1/data']
    rows={};data_files=[]
    for root in roots:
        ready=read(root/'data-ready-private.json')
        for spec in ready['files'].values():
            path=ROOT/spec['path']
            if sha(path)!=spec['sha256']:raise ValueError('Registered public input data changed')
            data_files.append(path)
            for line in path.open():
                row=json.loads(line);rows[row['_meta']['id']]=row
    score_path=ROOT/'runs/curriculum-stage4-placement-mortal-data-v4/input-audit-v1/public-score-details-private.jsonl'
    scores={r['id']:r['public_solo_score'] for r in (json.loads(line) for line in score_path.open())}
    pilot_path=ROOT/'runs/curriculum-live-worlds-v2/v8-continuation-pilot-v1/summary-private.json'
    if read(pilot_path)['status']!='frozen_v8_public_continuation_pilot_complete':raise ValueError('Verified continuation pilot required')
    wanted=[('third_ron','third_gap3000','ron'),('third_tsumo_if_available','third_gap3000','tsumo'),
        ('fourth_ron_gap5000','fourth_gap5000','ron'),('fourth_ron_gap13000','fourth_gap13000','ron')]
    selected=[];used=set()
    for name,context,win_kind in wanted:
        def candidates(kind):
            pool=[]
            for row in rows.values():
                m=row['_meta'];state=row['state'];score=scores.get(m['id'],{})
                if (m['split']!='train' or m['score_context']!=context or m['category']!='win'
                    or state['seat']==state['dealer'] or not score.get('supported')
                    or score['own_income']>3000 or score['immediate_rank']<=2):continue
                source=next(iter(state['win_context']['candidates'].values()))
                if source['kind']==kind:pool.append(row)
            return sorted(pool,key=lambda r:(r['_meta']['group_id'] in used,r['_meta']['id']))
        pool=candidates(win_kind)
        if not pool and name=='third_tsumo_if_available':pool=candidates('ron');name='third_second_ron_no_tsumo_available'
        if not pool:raise ValueError('Public TRAIN coverage missing for scout slot '+name)
        chosen=pool[0];selected.append((name,chosen));used.add(chosen['_meta']['group_id'])
    checkpoint=ROOT/reg['curriculum']['reference_checkpoint'];identity=bundle(checkpoint)
    if identity['identity_sha256']!=reg['curriculum']['reference_identity']:raise ValueError('Frozen v8 identity changed')
    paths=[Path(__file__),ROOT/'tools/curriculum_decision_rollout.py',ROOT/'tools/curriculum_native_decisions.py',
        ROOT/'tools/curriculum_hierarchical_prior.py',ROOT/'tools/curriculum_decision_history.py',parent/'summary-private.json',parent/'registered-private.json',
        ROOT/'tools/curriculum_terminal_scout_stats.py',ROOT/'docs/experiments/plans/curriculum-terminal-scout-20261122.md',
        score_path,pilot_path,*data_files]
    binding={str(p.resolve()):sha(p) for p in paths}
    registration=dict(status='conditional_terminal_scout_registered_before_rollouts',files=binding,teacher_candidate_identity=identity['identity_sha256'],
        cases=[dict(slot=b,id=r['_meta']['id'],split=r['_meta']['split'],public_solo=scores[r['_meta']['id']]) for b,r in selected],
        worlds_per_case=32,arms_per_world=2,prior=METHOD,opponent_profile='efficiency_with_calling_at_next_seat',
        descriptive_utility='top2 +0.02*(4-rank)/3 +0.005*tanh(net_from_anchor/8000)',
        later_own_legal_wins_taken_by_frozen_standard_guard=True,nested_pursuit_planner=False,
        teacher_not_admitted_for_training=True,labels_generated=False,engine_version=version('riichienv'))
    with (out/'registered-private.json').open('x') as f:json.dump(registration,f,indent=2);f.write('\n')
    started=time.monotonic();client=UnifiedPolicyClient(load_model(checkpoint));load_seconds=time.monotonic()-started
    class CountClient:
        def __init__(self):self.calls=0;self.seconds=0.
        def system_one(self,**kw):
            t=time.monotonic();value=client.system_one(**kw);self.seconds+=time.monotonic()-t;self.calls+=1;return value
    counted=CountClient();cases=[];all_started=time.monotonic()
    with (out/'paired-continuation-fixtures-private.jsonl').open('x') as stream:
        for n,(slot,row) in enumerate(selected):
            env,obs=restore(row);seat=obs.player_id;visible=obs.to_dict();scope=decision_scope(env,obs,env.mjai_log[-1])
            waits=lambda p,h:list(HandEvaluator(h,list(visible['melds'][p])).get_waits())
            proposal=sample(decision_inventory(obs,scope),public_history(env.mjai_log),visible['riichi_declared'],202611220500+n,32,2000000,waits)
            criteria=row['questions']['action']['criteria'];first=row['_meta']['source_control_key']
            alternatives=[k for k in criteria if k!=first]
            if criteria[first]['type']=='hora':alternatives.sort(key=lambda k:(criteria[k]['type']!='none',criteria[k]['type']!='dahai',k))
            elif row['_meta']['category']=='call_response':alternatives.sort(key=lambda k:(criteria[k]['type']=='none' if criteria[first]['type']=='none' else criteria[k]['type']!='none',k))
            else:alternatives.sort(key=lambda k:(criteria[k]['type']!='dahai',k))
            arms=[first,alternatives[0]];before_calls=counted.calls;before=time.monotonic();paired=[]
            for w,allocation in enumerate(proposal['allocations']):
                fork,_=construct_world(env,obs,env.mjai_log[-1],allocation,True)
                result={}
                for arm in arms:
                    agents=[CallingOpponent() if p==(seat+1)%4 else EfficiencyAgent() for p in range(4)]
                    agents[seat]=PolicyAgent(counted)
                    result[arm]=finish(fork.clone(),seat,arm,derived_seed(202611220500+n,w),agents=agents,record_events=True)
                stream.write(json.dumps(dict(id=row['_meta']['id'],group=row['_meta']['group_id'],world=w,prior=proposal['method'],arms=result))+'\n');stream.flush()
                paired.append(result)
                if len(paired) in (8,16,32):print(json.dumps(dict(stage='conditional_scout_prefix',slot=slot,summary=summarize(paired,*arms))),flush=True)
            case=dict(slot=slot,complete_terminal_arms=64,model_calls=counted.calls-before_calls,seconds=time.monotonic()-before,
                accept_key=first,decline_key=arms[1],source_group=row['_meta']['group_id'],conditional_summary=summarize(paired,*arms))
            cases.append(case);print(json.dumps(dict(stage='conditional_scout_case_complete',**case)),flush=True)
    if bundle(checkpoint)!=identity or not all(sha(p)==h for p,h in binding.items()):raise ValueError('Frozen continuation or inputs changed')
    result=dict(status='conditional_terminal_scout_complete_not_training_labels',teacher_candidate_identity=identity['identity_sha256'],
        cases=cases,total_complete_terminal_arms=256,total_model_calls=counted.calls,model_seconds=counted.seconds,
        rollout_wall_seconds=time.monotonic()-all_started,model_load_seconds=load_seconds,
        ordinary_future_wins_rule_forced=True,nested_pursuit_not_implemented=True,
        conditional_hierarchical_prior_not_calibrated=True,teacher_not_admitted=True,labels_generated=0,optimizer_updates=0,
        unseen_final_grid_unused=True,raw_rollouts_private=True)
    with (out/'summary-private.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();benchmark(a.out.resolve())
