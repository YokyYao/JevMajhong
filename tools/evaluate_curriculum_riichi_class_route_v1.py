"""Fixed-weight RI route preflight, native trial and independent cached audit."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read,sha
from tools.curriculum_riichi_control import require_registered
from tools.diagnostics.riichi_final_decoder_contrast_v1 import require_final
from tools.curriculum_riichi_class_policy_v1 import METHOD,select


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def rows(path):return [json.loads(line) for line in path.open()]


def context(config):
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS')
        or socket.gethostname().startswith(('lo-','login'))):raise RuntimeError('Scheduled GPU required')
    c=read(config);registered=ROOT/c['source_registration'];reg=read(registered);require_registered(reg)
    if (c['method']!=METHOD or c['threshold']!=.5 or c['seed_groups']!=8 or c['seats']!=4
        or c['candidate_matches']!=64 or c['baseline_reproduction_matches']!=16
        or c['parameter_updates']!=0 or c['automatic_parent_admission'] or c['automatic_release']
        or c['game_mode']!='4p-red-east' or c['profiles']!=reg['curriculum']['opponent_profiles']
        or c['logical_seed_start']!=reg['curriculum']['curve_seed_start']):raise ValueError('Fixed registered route recipe required')
    fit=ROOT/reg['curriculum']['out'];evidence=fit.parent/'final-evidence-private.json';completion=fit/'completion.json'
    require_final(read(completion),read(evidence))
    if read(fit.parent/'post-audit-private.json')['status']!='control_audit_terminal_and_real_scalar_curves_archived':raise ValueError('Actual final terminal capture required')
    sys.path.insert(0,reg['source']);sys.path.insert(0,reg['training_config']['upstream_source'])
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.training.base_lora import package_digest
    if package_digest(Path(reg['source'])/'jevmahjong')!=reg['source_sha256']:raise ValueError('Frozen runtime package changed')
    checkpoint=ROOT/c['checkpoint'];identity=bundle(checkpoint)
    if identity['identity_sha256']!=c['parameter_identity']:raise ValueError('Frozen final parameter identity required')
    paths=[config,registered,evidence,completion,Path(__file__),ROOT/'tools/curriculum_riichi_class_policy_v1.py',
        ROOT/'tests/core/test_curriculum_riichi_class_policy_v1.py',ROOT/'tools/evaluate_curriculum.py',
        ROOT/'docs/experiments/plans/curriculum-riichi-class-route-20261123.md']
    files=dict(reg['inputs']);files.update({str(p.resolve()):sha(p) for p in paths})
    references={}
    for name,folder in [('raw_current',fit/'evaluations/step-000608'),('v8',ROOT/reg['curriculum']['reference_evaluation_cache'])]:
        for filename in ('matches-private.jsonl','summary.json','protocol.json'):files[str(folder/filename)]=sha(folder/filename)
        summary=read(folder/'summary.json');protocol=read(folder/'protocol.json')
        expected=identity['identity_sha256'] if name=='raw_current' else reg['curriculum']['reference_identity']
        if (summary['checkpoint_sha256']!=expected or protocol['checkpoint']['identity_sha256']!=expected
            or summary['matches']!=64 or summary['seed_groups']!=8
            or protocol['seeds']!=list(range(c['logical_seed_start'],c['logical_seed_start']+8))
            or protocol['profiles']!=c['profiles'] or protocol['seats']!=4):raise ValueError('Reference policy/grid mismatch')
        references[name]=folder
    if any(sha(p)!=h for p,h in files.items()):raise ValueError('Frozen source/data/cache changed')
    return c,reg,fit,checkpoint,identity,files,references


def preflight(config,out):
    c,reg,fit,checkpoint,identity,files,refs=context(config)
    from tools.curriculum_calls_repair import prediction_index
    data=read(ROOT/reg['curriculum']['data_root']/'data-ready.json');spec=data['stages']['stage1']['validation'];path=ROOT/spec['path']
    if sha(path)!=spec['sha256']:raise ValueError('Actual validation changed')
    records=rows(path);prediction=fit/'training-run/training_eval/step-000608/development/rows.json'
    saved=prediction_index(records,read(prediction));counts={s:Counter() for s in ('legacy','teacher')};plain=0;riichi=0
    for row in records:
        q=row['questions']['action'];p=saved[row['_meta']['id']];keys=p['keys'];types=[q['criteria'][k]['type'] for k in keys]
        chosen,applied=select(types,p['p'],list(range(len(keys))))
        if applied:
            source=q['src'].removeprefix('curriculum1:').split(':')[0];truth=q['criteria'][q['label']]['type']=='reach';actual=types[chosen]=='reach'
            counts[source]['tp' if truth and actual else 'fn' if truth else 'fp' if actual else 'tn']+=1;riichi+=1
        else:
            assert chosen==max(range(len(keys)),key=p['p'].__getitem__);plain+=1
    contrast_path=ROOT/'runs/curriculum-riichi-final-decoder-contrast-v1/measurement-v1/summary-private.json';contrast=read(contrast_path)
    for source,value in counts.items():assert {k:value[k] for k in ('tp','fn','tn','fp')}==contrast['models']['max_class']['sources'][source]['confusion']['fixed_full_class_0_5']
    assert len(records)==3266 and riichi==632 and plain==2634
    for p in (path,prediction,contrast_path):files[str(p)]=sha(p)
    out.mkdir(parents=True,exist_ok=False)
    write(out/'summary-private.json',dict(status='fixed_RI_route_actual_cached_menu_preflight_passed',menus=3266,RI_menus=632,
        unchanged_plain_choices=plain,source_confusion={s:dict(v) for s,v in counts.items()},new_model_inference_games_updates=0,parent_admitted=False))
    write(out/'binding-private.json',dict(files=files,job_id=os.environ['SLURM_JOB_ID'],node=socket.gethostname()))


def evaluate(config):
    c,reg,fit,checkpoint,identity,files,refs=context(config)
    pre=ROOT/Path(c['out']).parent/'preflight-v1'
    assert read(pre/'summary-private.json')['status']=='fixed_RI_route_actual_cached_menu_preflight_passed'
    files.update(read(pre/'binding-private.json')['files']);assert all(sha(p)==h for p,h in files.items())
    for p in (pre/'summary-private.json',pre/'binding-private.json'):files[str(p)]=sha(p)
    import torch
    torch.set_num_threads(4)
    from jevmahjong.policy.routing import load_model
    from jevmahjong.policy.unified import UnifiedPolicyClient
    from jevmahjong.evaluation.agents import PolicyAgent
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.training.metric_journal import aggregate_complete_matches
    from tools.evaluate_curriculum import play
    from tools.curriculum_riichi_class_policy_v1 import RiichiClassPolicyAgent
    out=ROOT/c['out'];out.mkdir(parents=True,exist_ok=False)
    policy_spec=dict(parameter_identity=identity['identity_sha256'],runtime_source_sha256=reg['source_sha256'],method=METHOD,route_sha256=sha(ROOT/'tools/curriculum_riichi_class_policy_v1.py'),config_sha256=sha(config))
    policy_identity=hashlib.sha256(json.dumps(policy_spec,sort_keys=True).encode()).hexdigest()
    write(out/'protocol.json',dict(policy=policy_spec,policy_identity=policy_identity,seeds=list(range(c['logical_seed_start'],c['logical_seed_start']+8)),
        profiles=c['profiles'],seats=4,development_not_confirmation=True,parameter_updates=0))
    write(out/'binding-private.json',dict(files=files,job_id=os.environ['SLURM_JOB_ID'],node=socket.gethostname()))
    client=UnifiedPolicyClient(load_model(checkpoint));original=PolicyAgent(client)
    cached={(r['seed'],r['profile'],r['seat']):r for r in rows(refs['raw_current']/'matches-private.jsonl')}
    count=0
    with (out/'baseline16-private.jsonl').open('x') as stream:
        for seed in range(c['logical_seed_start'],c['logical_seed_start']+2):
            for profile in c['profiles']:
                for seat in range(4):
                    actual=play(original,seed,seat,profile);assert actual==cached[seed,profile,seat],'Raw baseline native reproduction failed'
                    stream.write(json.dumps(actual,separators=(',',':'))+'\n');stream.flush();count+=1
                    write(out/'status.json',dict(status='baseline_reproduction',complete=count,total=16))
    assert count==16;agent=RiichiClassPolicyAgent(client);games=[];started=time.monotonic()
    with (out/'matches-private.jsonl').open('x') as stream:
        for seed in range(c['logical_seed_start'],c['logical_seed_start']+8):
            for profile in c['profiles']:
                for seat in range(4):
                    row=play(agent,seed,seat,profile);games.append(row);stream.write(json.dumps(row,separators=(',',':'))+'\n');stream.flush()
                    write(out/'status.json',dict(status='candidate_matches',complete=len(games),total=64,seconds=time.monotonic()-started))
    assert bundle(checkpoint)==identity and all(sha(p)==h for p,h in files.items())
    summary=aggregate_complete_matches(games);summary.update(policy_identity=policy_identity,parameter_identity=identity['identity_sha256'],
        diagnostics=dict(agent.diagnostics),no_parameter_updates=True,not_parent_admitted=True)
    write(out/'summary.json',summary);write(out/'completion.json',dict(status='fixed_RI_route64_complete_not_audited',matches=64,baseline16_exact=True,parameter_identity_unchanged=True))
    write(out/'status.json',dict(status='complete',matches=64));print(json.dumps(summary),flush=True)


def audit(config,out):
    c,reg,fit,checkpoint,identity,files,refs=context(config)
    from jevmahjong.core.scoring import settlement_ledger
    from jevmahjong.evaluation.metrics import hand_dealin_count
    from jevmahjong.training.metric_journal import aggregate_complete_matches
    trial=ROOT/c['out'];assert read(trial/'completion.json')['status']=='fixed_RI_route64_complete_not_audited'
    protocol=read(trial/'protocol.json');files.update(read(trial/'binding-private.json')['files'])
    expected_policy=dict(parameter_identity=identity['identity_sha256'],runtime_source_sha256=reg['source_sha256'],method=METHOD,route_sha256=sha(ROOT/'tools/curriculum_riichi_class_policy_v1.py'),config_sha256=sha(config))
    assert protocol['policy']==expected_policy
    assert protocol['policy_identity']==hashlib.sha256(json.dumps(expected_policy,sort_keys=True).encode()).hexdigest()
    assert protocol['seeds']==list(range(c['logical_seed_start'],c['logical_seed_start']+8)) and protocol['profiles']==c['profiles'] and protocol['seats']==4
    grid={(seed,p,s) for seed in protocol['seeds'] for p in c['profiles'] for s in range(4)};arms={}
    for name,folder in [('candidate',trial),*refs.items()]:
        records=rows(folder/'matches-private.jsonl');assert len(records)==64 and {(r['seed'],r['profile'],r['seat']) for r in records}==grid
        for row in records:
            seat=row['seat'];ledger=settlement_ledger(row['mjai_log'],row['final_scores'])
            assert row['completed'] and row['ledger_passed'] and ledger['hands']==row['hands']
            assert row['net_points']==row['final_scores'][seat]-25000==sum(ledger['components'][seat].values())
            assert sorted(range(4),key=lambda p:(-row['final_scores'][p],p)).index(seat)+1==row['rank']
            assert hand_dealin_count(row['mjai_log'],seat)==row['dealin_hands']
            assert sum(e['type']=='hora' and e.get('actor')==seat for e in row['mjai_log'])==row['wins']
            counts=Counter(e['type'] for e in row['mjai_log'] if e.get('actor')==seat)
            assert row['choices']=={k:counts[k] for k in ('chi','pon','ankan','daiminkan','kakan','reach')}
        actual=aggregate_complete_matches(records);stored=read(folder/'summary.json')
        assert all(actual[k]==stored[k] for k in actual);arms[name]=records
        for filename in ('matches-private.jsonl','summary.json','protocol.json'):files[str(folder/filename)]=sha(folder/filename)
    raw={(r['seed'],r['profile'],r['seat']):r for r in arms['raw_current']};baseline=rows(trial/'baseline16-private.jsonl')
    assert len(baseline)==16 and all(r==raw[r['seed'],r['profile'],r['seat']] for r in baseline)
    assert {(r['seed'],r['profile'],r['seat']) for r in baseline}=={(s,p,a) for s in protocol['seeds'][:2] for p in c['profiles'] for a in range(4)}
    differences={}
    for name in refs:
        reference={(r['seed'],r['profile'],r['seat']):r for r in arms[name]};values={}
        for field,fn in [('net_points',lambda r:r['net_points']),('top2',lambda r:float(r['rank']<=2)),('hand_win',lambda r:r['wins']/r['hands']),('dealin',lambda r:r['dealin_hands']/r['hands']),('fourth',lambda r:float(r['rank']==4))]:
            clusters=[statistics.mean(fn(r)-fn(reference[r['seed'],r['profile'],r['seat']]) for r in arms['candidate'] if r['seed']==seed) for seed in protocol['seeds']]
            mean=statistics.mean(clusters);radius=2.3646242510102993*statistics.stdev(clusters)/math.sqrt(8)
            values[field]=dict(gain=mean,ci95=[mean-radius,mean+radius],seed_groups=8)
        differences[name]=values
    assert all(sha(p)==h for p,h in files.items())
    diagnostics=read(trial/'summary.json')['diagnostics']
    assert diagnostics['RI_scope_windows']==diagnostics.get('RI_selected',0)+diagnostics.get('RI_declined',0)
    assert diagnostics['different_from_raw_guarded']<=diagnostics['model_decisions'] and diagnostics['RI_scope_windows']<=diagnostics['model_decisions']
    out.mkdir(parents=True,exist_ok=False)
    result=dict(status='fixed_RI_route_native64_independent_cached_audit_passed',summaries={n:aggregate_complete_matches(r) for n,r in arms.items()},
        comparisons=differences,diagnostics=diagnostics,policy=protocol['policy'],policy_identity=protocol['policy_identity'],
        baseline16_exact=True,checked_complete_match_ledgers=192,raw_parameters_unchanged=True,source_grade_not_overridden=True,
        parent_admitted=False,new_optimizer_updates=0,final_reserved_grid_unused=True,consumed_development_not_independent_confirmation=True)
    write(out/'summary-private.json',result);write(out/'binding-private.json',dict(files=files,job_id=os.environ['SLURM_JOB_ID'],node=socket.gethostname()))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--phase',choices=('preflight','evaluate','audit'),required=True);p.add_argument('--out',type=Path)
    a=p.parse_args()
    if a.phase=='evaluate':evaluate(a.config.resolve())
    elif a.phase=='preflight':preflight(a.config.resolve(),a.out.resolve())
    else:audit(a.config.resolve(),a.out.resolve())
