"""Count genuinely recorded multi-action discards immediately after source reach.

This native audit creates no new labels, rows, fit or changes to frozen sources.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read,sha


def audit(out):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS')
    assert not socket.gethostname().startswith(('lo-','login'))
    out.mkdir(parents=True,exist_ok=False)
    checked=read(ROOT/'runs/curriculum-risk-teacher-data-v1/native-coverage-v1/summary-private.json')
    assert checked['status']=='all_teacher_native_turn_guard_and_reach_followups_audited' and checked['checked_rows']==44964
    reg_path=ROOT/'runs/curriculum-optimization-v2/fit-plan/registered.json';reg=read(reg_path)
    sys.path.insert(0,reg['source']);sys.path.insert(0,reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from riichienv import RiichiEnv,GameRule
    from jevmahjong.core.state import build_request,legal_candidates
    from jevmahjong.policy.unified import serving_request
    from tools.audit_curriculum_basic_teacher import visible
    files={str(Path(__file__).resolve()):sha(__file__),str(reg_path):sha(reg_path)}
    context=lambda m:(m['source_logical_seed'],m['source_hand_index'],m['source_step'],m['seat'])
    existing={};reaches=[]
    for cohort in ('basic','advanced','risk'):
        path=ROOT/f'runs/curriculum-{cohort}-teacher-data-v1/audit-v1/summary-private.json'
        summary=read(path);files[str(path)]=sha(path)
        for split,spec in summary['files'].items():
            data=ROOT/spec['path'];assert sha(data)==spec['sha256'];files[str(data)]=spec['sha256']
            for line in data.open():
                row=json.loads(line);m=row['_meta'];key=context(m)
                if key in existing:raise ValueError('Existing source context repeated')
                existing[key]=row
                q=row['questions']['action']
                if q['criteria'][q['label']]['type']=='reach':reaches.append((cohort,split,row))
    assert len(existing)==44964 and len(reaches)==1393
    source=ROOT/'runs/curriculum-basic-teacher-data-v1/data/source-matches-private.jsonl'
    files[str(source)]=sha(source);starts={}
    for line in source.open():
        match=json.loads(line)
        for index,event in enumerate(e for e in match['mjai'] if e['type']=='start_kyoku'):
            starts[(match['source']['seed'],index)]=event
    counts={};total=Counter()
    with (out/'missing-contexts-private.jsonl').open('x') as cases:
        for cohort,split,row in reaches:
            c=counts.setdefault(cohort+':'+split,Counter());m=row['_meta'];state=row['state']
            start=starts[(m['source_logical_seed'],m['source_hand_index'])]
            env=RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=m['source_engine_seed']+m['source_hand_index'])
            observations=env.reset(oya=state['dealer'],round_wind=state['round_wind'],honba=state['honba'],scores=start['scores'],kyotaku=start['kyotaku'])
            identity=lambda a:(int(a.action_type),a.tile,tuple(sorted(a.consume_tiles)))
            for batch in m['source_trace']:
                assert {a['player'] for a in batch}==set(observations)
                selected={}
                for saved in batch:
                    menu={identity(a):a for a in legal_candidates(observations[saved['player']])}
                    selected[saved['player']]=menu[(saved['type'],saved['tile'],tuple(sorted(saved['consumed'])))]
                observations=env.step(selected)
            assert set(observations)=={m['seat']}
            request,actions,_=build_request(observations[m['seat']]);assert serving_request(request)==visible(row)
            index=list(request['questions']['action']['criteria']).index(row['questions']['action']['label'])
            observations=env.step({m['seat']:actions[index]});assert set(observations)=={m['seat']}
            next_obs=observations[m['seat']];menu=legal_candidates(next_obs)
            assert {json.loads(a.to_mjai())['type'] for a in menu}=={'dahai'}
            c['source_selected_reach']+=1;total['source_selected_reach']+=1
            if len(menu)==1:
                c['following_discard_forced']+=1;total['following_discard_forced']+=1
                continue
            next_key=(m['source_logical_seed'],m['source_hand_index'],m['source_step']+1,m['seat'])
            next_request,_,_=build_request(next_obs)
            if next_key in existing:
                saved=existing[next_key]
                if saved['_meta']['split']!=split or visible(saved)!=serving_request(next_request):
                    raise ValueError('Alleged recorded next context is not the actual following native input')
                field='following_multi_action_already_recorded'
            else:
                field='following_multi_action_missing_from_all_three_teacher_cohorts'
                cases.write(json.dumps(dict(source_reach_id=m['id'],context=next_key,split=split,actions=len(menu)),separators=(',',':'))+'\n')
            c[field]+=1;total[field]+=1
    assert total['source_selected_reach']==1393
    assert total['following_discard_forced']==846
    assert all(sha(path)==digest for path,digest in files.items())
    result=dict(status='all1393_actual_source_reach_followup_training_coverage_audited',
        counts={key:dict(value) for key,value in counts.items()},total=dict(total),
        all_existing_three_teacher_contexts_checked=len(existing),
        no_new_teacher_labels_or_training_rows=True,no_model_loaded_or_updated=True,
        missing_coverage_not_proof_of_causal_playing_strength_failure=True,
        current_data_fit_serving_and_source_bindings_unchanged=True,raw_cases_private=True)
    for name,value in (('summary-private.json',result),('binding-private.json',dict(files=files,node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID']))):
        with (out/name).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();audit(args.out.resolve())
