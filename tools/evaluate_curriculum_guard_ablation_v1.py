"""Two fixed-weight, paired development serving routes; no training or admission."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read,sha


def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def evaluate(out):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS')
    assert not socket.gethostname().startswith(('lo-','login'))
    out.mkdir(parents=True,exist_ok=False)
    registered=ROOT/'runs/curriculum-basic-repair-v1/fit-plan/registered.json'
    reg=read(registered);c=reg['curriculum']
    sys.path.insert(0,reg['source']);sys.path.insert(0,reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.policy.routing import load_model
    from jevmahjong.policy.unified import UnifiedPolicyClient
    from jevmahjong.evaluation.agents import PolicyAgent,guarded_indices
    from jevmahjong.core.state import build_request,signature
    from jevmahjong.training.metric_journal import aggregate_complete_matches
    from tools.evaluate_curriculum import play
    from tools.curriculum_all_legal_policy_v1 import AllLegalPolicyAgent
    from tools.audit_curriculum_basic_teacher import visible
    from tools.audit_curriculum_endgame_cases import restore
    from types import SimpleNamespace
    if package_digest(Path(reg['source'])/'jevmahjong')!=reg['source_sha256']:
        raise ValueError('Pinned model/game implementation changed')
    files=dict(reg['inputs']);files[str(registered)]=sha(registered)
    for name in ('tools/evaluate_curriculum_guard_ablation_v1.py','tools/curriculum_all_legal_policy_v1.py',
                 'tests/core/test_curriculum_all_legal_policy_v1.py',
                 'docs/experiments/plans/curriculum-guard-ablation-20261123.md'):
        files[str(ROOT/name)]=sha(ROOT/name)
    if not all(sha(path)==digest for path,digest in files.items()):
        raise ValueError('Frozen registered input changed')
    checkpoint=Path(c['reference_checkpoint']);identity=bundle(checkpoint)
    assert identity['identity_sha256']==c['reference_identity']=='5ebce6af3777854cbec55ba29db2dc90be0f959193114bb3154179c5552370fa'
    seeds=list(range(c['curve_seed_start'],c['curve_seed_start']+c['curve_seed_groups']))
    assert seeds==[2026200000,2026200001] and c['opponent_profiles']==['efficiency','calling']
    expected={(seed,profile,seat) for seed in seeds for profile in c['opponent_profiles'] for seat in range(4)}
    cache=ROOT/c['reference_evaluation_cache'];cached={}
    for name in ('protocol.json','summary.json','matches-private.jsonl'):
        files[str(cache/name)]=sha(cache/name)
    assert read(cache/'protocol.json')['checkpoint']==identity
    for line in (cache/'matches-private.jsonl').open():
        row=json.loads(line);key=(row['seed'],row['profile'],row['seat'])
        if key in expected:
            if key in cached:raise ValueError('Duplicate cached v8 match')
            cached[key]=row
    assert set(cached)==expected
    write(out/'registered-private.json',dict(status='frozen_v8_two_route_development_diagnostic_registered_before_inference',
        checkpoint=identity,seeds=seeds,profiles=c['opponent_profiles'],matches_per_arm=16,
        new_optimizer_updates=0,no_final_reserved_seeds=True,files=files,
        job_id=os.environ['SLURM_JOB_ID'],node=socket.gethostname()))

    # Interface preflight uses real native states and explicitly synthetic
    # preferences. No Qwen inference/teacher label generation occurs here.
    raw_root=ROOT/'runs/curriculum-risk-teacher-data-v1/data'
    risk=read(raw_root/'summary-private.json');files[str(raw_root/'summary-private.json')]=sha(raw_root/'summary-private.json')
    starts={};source=raw_root/'source-matches-private.jsonl';files[str(source)]=sha(source)
    for line in source.open():
        match=json.loads(line)
        for index,event in enumerate(e for e in match['mjai'] if e['type']=='start_kyoku'):
            starts[(match['source']['seed'],index)]=event
    coverage=Counter();seen=set()
    for split,spec in risk['files'].items():
        path=ROOT/spec['path'];assert sha(path)==spec['sha256'];files[str(path)]=spec['sha256']
        for line in path.open():
            row=json.loads(line);meta=row['_meta'];diag=meta['guard_diagnostic']
            cell=(split,diag['reason'],diag['opponent_riichi'])
            if cell in seen:continue
            seen.add(cell);start=starts[(meta['source_logical_seed'],meta['source_hand_index'])]
            replay=visible(row);replay['_meta']=dict(meta,initial_scores=start['scores'],initial_sticks=start['kyotaku'],
                category='diagnostic_nonwin',source_control_key=row['questions']['action']['label'])
            _,obs=restore(replay);request,actions,features=build_request(obs)
            keys=list(request['questions']['action']['criteria']);label=row['questions']['action']['label'];target=keys.index(label)
            probabilities={key:float(key==label) for key in keys}
            calls=[]
            def fake_answer(**value):
                if value!=request:raise ValueError('Ablation changed the ordinary native request')
                calls.append(value)
                return {'answers':{'action':{'probabilities':probabilities}}}
            client=SimpleNamespace(system_one=fake_answer)
            experimental=AllLegalPolicyAgent(client)
            chosen=experimental.act(obs)
            if signature(chosen)!=signature(actions[target]) or len(calls)!=1:
                raise ValueError('All-legal route did not execute the legal unique synthetic preference with one forward')
            allowed=set(guarded_indices(obs,actions,features))|{i for i,f in enumerate(features) if not f}
            if (target in allowed)!=diag['teacher_target_allowed']:
                raise ValueError('Native guard diagnostic changed')
            original=PolicyAgent(client).act(obs)
            expected_index=max(sorted(allowed),key=lambda i:probabilities[keys[i]])
            if signature(original)!=signature(actions[expected_index]) or len(calls)!=2:
                raise ValueError('Original helper changed its guarded argmax')
            coverage[':'.join(map(str,cell))]+=1
    assert {reason for _,reason,_ in seen}=={'allowed','blocked_rule_risk','blocked_shanten_increase'}
    assert {split for split,_,_ in seen}=={'train','validation'}
    write(out/'native-preflight-private.json',dict(status='actual_native_guard_route_interface_preflight_passed',
        coverage=dict(coverage),synthetic_preferences_only=True,no_model_inference_or_new_labels=True))

    client=UnifiedPolicyClient(load_model(checkpoint));results={};summaries={};shadow={}
    for mode in ('original_guard','all_legal'):
        agent=PolicyAgent(client) if mode=='original_guard' else AllLegalPolicyAgent(client)
        rows=[]
        with (out/f'{mode}-matches-private.jsonl').open('x') as stream:
            for seed in seeds:
                for profile in c['opponent_profiles']:
                    for seat in range(4):
                        row=play(agent,seed,seat,profile)
                        if mode=='original_guard':
                            before=cached[(seed,profile,seat)]
                            fields=('final_scores','rank','net_points','hands','wins','dealin_hands','mjai_log')
                            if any(before[field]!=row[field] for field in fields):
                                write(out/'cached-baseline-mismatch-private.json',dict(actual=row,expected=before))
                                raise ValueError('Fresh unchanged-guard v8 does not reproduce the cached trajectory')
                        stream.write(json.dumps(row,separators=(',',':'))+'\n');stream.flush();rows.append(row)
                        (out/'status-private.json').write_text(json.dumps(dict(status='running',mode=mode,completed=len(rows),total=16))+'\n')
        if len(rows)!=16 or {(r['seed'],r['profile'],r['seat']) for r in rows}!=expected:
            raise ValueError('Incomplete or duplicate fixed evaluation grid')
        results[mode]=rows;summaries[mode]=aggregate_complete_matches(rows)
        shadow[mode]=dict(getattr(agent,'shadow',{}))
    bykey=lambda rows:{(r['seed'],r['profile'],r['seat']):r for r in rows}
    before=bykey(results['original_guard']);after=bykey(results['all_legal'])
    metrics=dict(net_points=lambda r:r['net_points'],top2_rate=lambda r:float(r['rank']<=2),
        hand_win_rate=lambda r:r['wins']/r['hands'],dealin_hand_rate=lambda r:r['dealin_hands']/r['hands'])
    gains={field:statistics.mean(fn(after[key])-fn(before[key]) for key in expected) for field,fn in metrics.items()}
    seed_gains={str(seed):{field:statistics.mean(fn(after[key])-fn(before[key]) for key in expected if key[0]==seed)
        for field,fn in metrics.items()} for seed in seeds}
    if bundle(checkpoint)!=identity or not all(sha(path)==digest for path,digest in files.items()):
        raise ValueError('Frozen checkpoint/cache/source input changed')
    result=dict(status='frozen_v8_guard_ablation_complete_development_only',summaries=summaries,
        paired_mean_gains_all_legal_minus_original=gains,paired_seed_cluster_gains=seed_gains,
        shadow_guard_counters=shadow,checkpoint_identity=identity['identity_sha256'],
        seed_groups=2,matches_per_arm=16,unchanged_guard_exact_cache_reproduction=True,
        ci95=None,insufficient_clusters_for_independent_strength_claim=True,
        no_policy_deployment_or_training_admission=True,no_model_updates=True,
        current_basic_and_conditional_control_unchanged=True,final_reserved_grid_unused=True,
        private_full_match_logs=True)
    write(out/'summary-private.json',result)
    write(out/'binding-private.json',dict(files=files,checkpoint=identity,job_id=os.environ['SLURM_JOB_ID']))
    (out/'status-private.json').write_text(json.dumps(dict(status='complete',matches=32))+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();evaluate(args.out.resolve())
