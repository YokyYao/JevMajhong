"""Serving-guard coverage and fixed class-posterior diagnostics on cached RI scores."""
import argparse
from collections import Counter,defaultdict
import json
import math
import os
from pathlib import Path
import socket
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.collect_curriculum_endgame_cases import sha


def auc(pairs):
    positive=sum(y for _,y in pairs);negative=len(pairs)-positive
    if not positive or not negative:return None
    ranked=sorted(pairs);score=0.;below=0;i=0
    while i<len(ranked):
        j=i+1
        while j<len(ranked) and ranked[j][0]==ranked[i][0]:j+=1
        positives=sum(y for _,y in ranked[i:j]);negatives=j-i-positives
        score+=positives*(below+.5*negatives);below+=negatives;i=j
    return score/(positive*negative)


def audit(registered,basic,out,step=76):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS')
    assert not socket.gethostname().startswith(('lo-','login'))
    read=lambda p:json.loads(p.read_text());reg=read(registered);fit=ROOT/reg['curriculum']['out']
    sys.path.insert(0,reg['source']);sys.path.insert(0,reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from jevmahjong.evaluation.agents import guarded_indices
    from jevmahjong.core.state import build_request
    from jevmahjong.evaluation.unified_contract import bundle
    from tools.audit_curriculum_basic_teacher import visible
    from tools.audit_curriculum_endgame_cases import restore
    if not all(sha(p)==h for p,h in reg['inputs'].items()):raise ValueError('Active fit inputs changed')
    out.mkdir(parents=True,exist_ok=False)
    data=read(ROOT/reg['curriculum']['data_root']/'data-ready.json');validation=ROOT/data['stages']['stage1']['validation']['path']
    predictions=fit/'training-run/training_eval'/f'step-{step:06d}'/'development/rows.json'
    cached={r['id']:r for r in read(predictions)};rows={r['_meta']['id']:r for r in (json.loads(s) for s in validation.open())}
    if set(cached)!=set(rows):raise ValueError('Cached validation coverage differs')
    checkpoint=fit/'training-run/checkpoints'/f'step-{step:06d}';identity=bundle(checkpoint)
    summary=read(basic/'summary-private.json');paths=[Path(__file__),registered,validation,predictions,basic/'summary-private.json',basic/'source-matches-private.jsonl',
        ROOT/'docs/experiments/plans/curriculum-riichi-execution-audit-20261122.md']
    for spec in summary['files'].values():
        path=ROOT/spec['path']
        if sha(path)!=spec['sha256']:raise ValueError('Teacher raw source changed')
        paths.append(path)
    binding={str(p.resolve()):sha(p) for p in paths}
    def write(path,value):
        with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    write(out/'registered-private.json',dict(files=binding,checkpoint=identity,step=step,
        fixed_threshold=.5,no_threshold_search=True,no_model_inference_or_updates=True,job_id=os.environ['SLURM_JOB_ID']))
    confusion=defaultdict(Counter);scored=defaultdict(list)
    def count(source,method,truth,predicted):confusion[source+':'+method]['tp' if truth and predicted else 'fn' if truth else 'fp' if predicted else 'tn']+=1
    def probabilities(row):
        p=cached[row['_meta']['id']];q=row['questions']['action'];prob=dict(zip(p['keys'],p['p']))
        if set(prob)!=set(q['criteria']) or not all(math.isfinite(v) and v>=0 for v in prob.values()) or abs(sum(prob.values())-1)>1e-5:
            raise ValueError('Invalid cached action distribution')
        return prob
    for row in rows.values():
        q=row['questions']['action'];types={v['type'] for v in q['criteria'].values()}
        if 'reach' not in types:continue
        prob=probabilities(row);source=row['questions']['action']['src'].removeprefix('curriculum1:').split(':')[0]
        truth=q['criteria'][q['label']]['type']=='reach';reach=sum(v for k,v in prob.items() if q['criteria'][k]['type']=='reach')
        chosen=max(prob,key=prob.get);count(source,'raw_argmax',truth,q['criteria'][chosen]['type']=='reach')
        count(source,'fixed_marginal_full_0_5',truth,reach>=.5);scored[source].append((reach,truth))
    point=next(r for r in read(fit/'validation-curve.json') if r['step']==step)
    for source,expected in point['riichi_by_source'].items():
        if any(confusion[source+':raw_argmax'].get(k,0)!=expected.get(k,0) for k in ('tp','fn','tn','fp')):raise ValueError('Raw RI counts do not reproduce actual validation')
    starts={}
    for line in (basic/'source-matches-private.jsonl').open():
        match=json.loads(line)
        for index,event in enumerate(e for e in match['mjai'] if e['type']=='start_kyoku'):starts[(match['source']['seed'],index)]=event
    coverage={s:Counter() for s in ('train','validation')};blocked=[]
    for split,spec in summary['files'].items():
        for line in (ROOT/spec['path']).open():
            row=json.loads(line);m=row['_meta'];q=row['questions']['action'];start=starts[(m['source_logical_seed'],m['source_hand_index'])]
            replay=visible(row);replay['_meta']=dict(m,initial_scores=start['scores'],initial_sticks=start['kyotaku'],category='diagnostic_nonwin',source_control_key=q['label'])
            env,obs=restore(replay);request,actions,features=build_request(obs);keys=list(request['questions']['action']['criteria'])
            eligible=sorted(set(guarded_indices(obs,actions,features))|{i for i,f in enumerate(features) if not f})
            allowed={keys[i] for i in eligible};kind=m['curriculum_stratum'];threats=sum(p!=m['seat'] for p in row['state']['riichi'])
            c=coverage[split];c['rows']+=1;c[kind+':rows']+=1
            if q['label'] not in allowed:
                c['teacher_target_excluded']+=1;c[kind+':teacher_target_excluded']+=1;c['excluded_vs_'+str(threats)+'_riichi']+=1
                blocked.append(dict(id=m['id'],group=m['group_id'],stratum=kind,teacher_key=q['label'],eligible_keys=sorted(allowed)))
            if split=='validation' and 'reach' in {v['type'] for v in q['criteria'].values()}:
                prob=probabilities(rows[m['id']]);truth=q['criteria'][q['label']]['type']=='reach'
                best=max(allowed,key=lambda k:prob[k]);count('teacher','actual_guarded_argmax',truth,q['criteria'][best]['type']=='reach')
                reach=sum(prob[k] for k in allowed if q['criteria'][k]['type']=='reach');total=sum(prob[k] for k in allowed)
                count('teacher','fixed_marginal_eligible_0_5',truth,reach/total>=.5)
    with (out/'excluded-targets-private.jsonl').open('x') as stream:
        for row in blocked:stream.write(json.dumps(row)+'\n')
    if bundle(checkpoint)!=identity or not all(sha(p)==h for p,h in binding.items()) or not all(sha(p)==h for p,h in reg['inputs'].items()):raise ValueError('Frozen checkpoint/cache/inputs changed')
    result=dict(status='cached_riichi_class_and_actual_serving_eligibility_diagnostic_complete',step=step,
        confusion={k:dict(v) for k,v in confusion.items()},riichi_probability_auc={k:auc(v) for k,v in scored.items()},
        actual_guard_source_target_coverage={k:dict(v) for k,v in coverage.items()},
        no_threshold_selected_or_policy_changed=True,no_new_inference_labels_updates_or_games=True,
        imitation_not_win_probability_or_strength=True,raw_case_details_private=True)
    write(out/'summary-private.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('registered','basic','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--step',type=int,default=76);a=p.parse_args();audit(a.registered.resolve(),a.basic.resolve(),a.out.resolve(),a.step)
