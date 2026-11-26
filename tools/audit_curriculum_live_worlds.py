"""Native scope/threat matrix: public-prior fixtures, never teacher labels."""
import argparse
from collections import Counter
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
from tools.curriculum_public_worlds import SamplingExhausted


def audit(out,worlds=2,max_attempts=250000):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS')
    assert not socket.gethostname().startswith(('lo-','login'))
    if type(worlds) is not int or not 1<=worlds<=4:raise ValueError('Bounded fixture worlds required')
    out.mkdir(parents=True,exist_ok=False)
    read=lambda path:json.loads(path.read_text())
    reg_path=ROOT/'runs/curriculum-optimization-v2/fit-plan/registered.json';reg=read(reg_path)
    sys.path.insert(0,reg['source'])
    from riichienv import HandEvaluator
    import riichienv._riichienv as native
    from jevmahjong.core.state import build_request
    from jevmahjong.training.base_lora import package_digest
    from tools.audit_curriculum_endgame_cases import restore
    from tools.audit_curriculum_history_worlds import derived_seed,reactions
    from tools.curriculum_decision_history import public_history,history_flags
    from tools.curriculum_hierarchical_prior import sample,METHOD
    from tools.curriculum_native_decisions import decision_scope,decision_inventory,construct_world
    from tools.curriculum_decision_rollout import finish
    if version('riichienv')!='0.4.11' or package_digest(Path(reg['source'])/'jevmahjong')!=reg['source_sha256']:
        raise ValueError('Pinned native/project implementation changed')
    paths=[Path(__file__),reg_path,Path(native.__file__),ROOT/'docs/experiments/plans/curriculum-live-worlds-20261122.md']
    paths += [ROOT/'tools'/p for p in ('curriculum_decision_history.py','curriculum_hierarchical_prior.py',
        'curriculum_native_decisions.py','curriculum_decision_rollout.py','curriculum_native_worlds.py',
        'curriculum_public_inventory.py','audit_curriculum_endgame_cases.py','audit_curriculum_history_worlds.py','curriculum_endgame_ledger.py')]
    inputs=[]
    for root in (ROOT/'runs/curriculum-stage4-placement-mortal-data-v4',ROOT/'runs/curriculum-stage4-threat-prefix-v1'):
        ready_path=root/'data/data-ready-private.json';ready=read(ready_path);paths.append(ready_path)
        if not (root/'preparation-execution-private.json').exists():raise ValueError('Unverified source inputs')
        paths += [root/'preparation-execution-private.json',root/'input-audit-v1/summary-private.json']
        for split,spec in ready['files'].items():
            path=ROOT/spec['path']
            if sha(path)!=spec['sha256']:raise ValueError('An admitted source-input file changed')
            paths.append(path)
            inputs.extend(row for row in (json.loads(line) for line in path.open()) if row['_meta']['score_context']=='hold_first')
    selected={};unsupported=Counter()
    for row in sorted(inputs,key=lambda r:r['_meta']['id']):
        env,obs=restore(row)
        try:scope=decision_scope(env,obs,env.mjai_log[-1])
        except ValueError:unsupported[env.mjai_log[-1]['type']]+=1;continue
        threats=sum(p!=obs.player_id for p in row['state']['riichi'])
        bucket=(scope,row['_meta']['category'],threats)
        if bucket not in selected:selected[bucket]=row
    if len(selected)>24:raise ValueError('A changed matrix needs a new budget')
    binding={str(p.resolve()):sha(p) for p in paths}
    registration=dict(status='live_native_fixture_matrix_registered_before_worlds',files=binding,
        selected=[dict(bucket=list(k),id=r['_meta']['id'],group=r['_meta']['group_id'],split=r['_meta']['split']) for k,r in selected.items()],
        worlds_per_case=worlds,max_native_hand_draws_per_case=max_attempts,prior=METHOD,
        hierarchical_prior_is_not_uniform_joint_for_multi_riichi=True,world_weights_reference_declared_prior_only=True,
        source_selection_by_public_scope_threat_and_ID_not_outcome=True,labels_generated=False,teacher_admitted=False)
    with (out/'registered-private.json').open('x') as f:json.dump(registration,f,indent=2);f.write('\n')
    counts=Counter();failures=[];cases=[]
    def truth_allocation(env,obs,waiting,history):
        known={4+2*i-env.rinshan_draw_count for i in range(len(env.dora_indicators))}
        proposal=dict(opponent_hands={p:list(h) for p,h in enumerate(env.hands) if p!=obs.player_id},
            unknown_nonhand_tiles=[t for i,t in enumerate(env.wall) if i not in known])
        flags=history_flags(history,obs.player_id,obs.to_dict()['riichi_declared'],proposal,waiting)
        if not flags['compatible']:raise ValueError('Truth control violates the declared no-missed-opponent history restriction')
        proposal['opponent_temporary_furiten']=flags['opponent_temporary_furiten'];return proposal
    with (out/'paired-full-fixtures-private.jsonl').open('x') as stream:
        for number,(bucket,row) in enumerate(selected.items()):
            env,obs=restore(row);seat=obs.player_id;visible=obs.to_dict();history=public_history(env.mjai_log)
            waiting=lambda p,h:list(HandEvaluator(h,list(visible['melds'][p])).get_waits())
            source_cache=reactions(env)
            control,truth_receipt=construct_world(env,obs,env.mjai_log[-1],truth_allocation(env,obs,waiting,history),True)
            if reactions(control)!=source_cache:raise ValueError('Reconstructing the true physical world changed its reaction cache')
            if list(control.missed_agari_doujun)!=list(env.missed_agari_doujun):
                raise ValueError('Native truth temporary-furiten flags differ after reconstruction')
            counts['truth_menu_cache_furiten_controls']+=1
            request=build_request(obs)[0];q=request['questions']['action']['criteria'];keys=list(q)
            first=row['_meta']['source_control_key']
            if first not in keys:raise ValueError('Source control action absent')
            other=[k for k in keys if k!=first]
            if q[first]['type']=='hora':
                other.sort(key=lambda k:(q[k]['type']!='none',q[k]['type']!='dahai',k))
            elif row['_meta']['category']=='call_response':
                other.sort(key=lambda k:(q[k]['type']=='none' if q[first]['type']=='none' else q[k]['type']!='none',k))
            else:other.sort(key=lambda k:(q[k]['type']!='dahai',k))
            if not other:raise ValueError('A fixture needs two legal alternatives')
            arms=[first,other[0]];start=time.monotonic()
            try:
                proposal=sample(decision_inventory(obs,bucket[0]),history,visible['riichi_declared'],202611220300+number,
                    worlds,max_attempts,waiting)
            except SamplingExhausted as error:
                failure=dict(bucket=list(bucket),source_id=row['_meta']['id'],attempts=error.attempts,partial_worlds_not_used=error.accepted)
                failures.append(failure);print(json.dumps(dict(stage='prior_budget_exhausted',**failure)),flush=True);continue
            elapsed=time.monotonic()-start;counts['native_hand_draws']+=proposal['attempts']
            counts['history_rejections']+=proposal['history_rejections']
            for index,allocation in enumerate(proposal['allocations']):
                fork,receipt=construct_world(env,obs,env.mjai_log[-1],allocation,True)
                counts['sampled_worlds']+=1;counts['changed_reaction_caches']+=reactions(fork)!=source_cache
                seed=derived_seed(202611220300+number,index)
                results={key:finish(fork.clone(),seat,key,seed,record_events=True) for key in arms}
                for result in results.values():
                    counts['complete_terminal_arms']+=1;counts['later_hand_reseeds']+=result['future_hand_reseeds']
                stream.write(json.dumps(dict(source_id=row['_meta']['id'],group=row['_meta']['group_id'],bucket=list(bucket),
                    world=index,prior=proposal['method'],allocation=allocation,future_seed=seed,receipt=receipt,arms=results))+'\n');stream.flush()
            cases.append(dict(bucket=list(bucket),source_id=row['_meta']['id'],attempts=proposal['attempts'],sampling_seconds=elapsed,
                history_rejections=proposal['history_rejections'],completed_worlds=worlds,completed_terminal_arms=2*worlds))
            print(json.dumps(dict(stage='native_matrix_case_completed',**cases[-1])),flush=True)
    if not all(sha(p)==h for p,h in binding.items()):raise ValueError('Registered preflight code/data/runtime changed')
    result=dict(status='live_decision_native_matrix_complete' if not failures else 'live_decision_native_matrix_prior_coverage_incomplete',
        node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'],counts=dict(counts),selected_cases=len(selected),
        cases=cases,prior_exhaustions=failures,unsupported_current_triggers=dict(unsupported),
        independent_source_families=len({r['_meta']['group_id'] for r in selected.values()}),
        public_hierarchical_prior_not_calibrated=True,equal_weights_do_not_target_uniform_joint_for_multi_riichi=True,
        four_public_efficiency_continuations_are_integration_fixtures=True,independent_han_fu_scoring=False,
        actual_known_observer_decline_behavior_controls_pending=True,validated_continuation_teacher_pending=True,
        new_training_labels=0,stage4_fit_admitted=False,unseen_final_grid_unused=True,raw_worlds_events_private=True)
    with (out/'summary-private.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)
    if failures:raise RuntimeError('The sampler did not cover every preregistered native fixture bucket')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--worlds',type=int,default=2);p.add_argument('--max-attempts',type=int,default=250000)
    a=p.parse_args();audit(a.out.resolve(),a.worlds,a.max_attempts)
