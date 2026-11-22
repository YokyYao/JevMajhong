"""Representative true-history Mortal legal flags and daiminkan Q audit."""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import socket
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.collect_curriculum_endgame_cases import sha


def audit(basic,advanced,out):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS')
    assert not socket.gethostname().startswith(('lo-','login'))
    read=lambda p:json.loads(p.read_text());br=read(basic/'registered-private.json');ar=read(advanced/'registered-private.json')
    bs=read(basic/'summary-private.json');ars=read(advanced/'summary-private.json')
    if bs['stats']['complete_sources']!=256 or ars['stats']['identical_parent_trajectories']!=256 or br['sources']!=ar['sources']:
        raise ValueError('Both complete identical256-family source cohorts required')
    for registration in (br,ar):
        if not all(sha(p)==h for p,h in registration['files'].items()):raise ValueError('A source binding changed')
    reg_path=ROOT/br['config']['source_registration'];reg=read(reg_path);sys.path.insert(0,reg['source'])
    import torch
    torch.set_num_threads(4)
    from tools.curriculum_mortal_source import MortalSource,public_events
    from tools.audit_curriculum_basic_teacher import visible
    from tools.audit_curriculum_endgame_cases import restore
    out.mkdir(parents=True,exist_ok=False)
    actor=MortalSource((ROOT/br['config']['teacher_root']).resolve())
    selected={};allowed={s['seed'] for s in br['sources'][:64]};paths=[Path(__file__),reg_path]
    for role,root,ready in (('basic',basic,bs),('advanced',advanced,ars)):
        paths.extend((root/'registered-private.json',root/'summary-private.json'))
        for spec in ready['files'].values():
            path=ROOT/spec['path']
            if sha(path)!=spec['sha256']:raise ValueError('Completed source decision file changed')
            paths.append(path)
            for line in path.open():
                row=json.loads(line);m=row['_meta']
                if m['source_logical_seed'] not in allowed:continue
                q=row['questions']['action'];types={v['type'] for v in q['criteria'].values()};chosen=q['criteria'][q['label']]['type']
                for kind in sorted(types&{'chi','pon','ankan','daiminkan','kakan','reach'}):
                    slot=(role,m['split'],kind,'selected' if kind==chosen else 'other',sum(p!=m['seat'] for p in row['state']['riichi']))
                    selected.setdefault(slot,row)
    unique={r['_meta']['id']:r for r in selected.values()}
    if not 1<=len(unique)<=256:raise ValueError('Unexpected diagnostic fixture count')
    snapshot=out/'selected-fixtures-private.jsonl'
    with snapshot.open('x') as f:
        for identity in sorted(unique):f.write(json.dumps(unique[identity])+'\n')
    raw=basic/'source-matches-private.jsonl'
    if sha(raw)!=bs['full_source_matches_sha256'] or sha(raw)!=ars['full_source_matches_sha256']:raise ValueError('Shared complete source history differs')
    starts={}
    for line in raw.open():
        match=json.loads(line)
        for index,event in enumerate(e for e in match['mjai'] if e['type']=='start_kyoku'):starts[(match['source']['seed'],index)]=event
    paths.extend((raw,snapshot,ROOT/'tools/curriculum_mortal_source.py',ROOT/'tools/audit_curriculum_endgame_cases.py'))
    binding={str(p.resolve()):sha(p) for p in paths};binding.update(actor.bindings)
    def write(p,value):
        with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    write(out/'registered-private.json',dict(files=binding,fixtures=len(unique),selection='first_by_role_split_offered_selectedclass_threat_within_first64_families',
        source_is_true_history_not_resampled=True,labels_generated=0,job_id=os.environ['SLURM_JOB_ID']))
    fields={'chi':'can_chi','pon':'can_pon','ankan':'can_ankan','daiminkan':'can_daiminkan','kakan':'can_kakan','reach':'can_riichi'}
    mapped_index=lambda i:'dahai' if i<=36 else {37:'reach',38:'chi',39:'chi',40:'chi',41:'pon',42:'kan',43:'hora',44:'ryukyoku',45:'none'}[i]
    counts=Counter();mismatches=[];q_mismatches=[]
    with (out/'details-private.jsonl').open('x') as stream:
        for identity in sorted(unique):
            row=unique[identity];m=row['_meta'];q=row['questions']['action'];start=starts[(m['source_logical_seed'],m['source_hand_index'])]
            replay=visible(row);replay['_meta']=dict(m,initial_scores=start['scores'],initial_sticks=start['kyotaku'],
                category='diagnostic_nonwin',source_control_key=q['label'])
            env,obs=restore(replay);seat=m['seat'];events=[]
            for original in env.mjai_log:
                event=copy.deepcopy(original)
                if event['type']=='start_kyoku':event['tehais']=[h if p==seat else ['?']*13 for p,h in enumerate(event['tehais'])]
                if event['type']=='tsumo' and event['actor']!=seat:event['pai']='?'
                events.append(event)
            mirror=actor.PlayerState(seat)
            for event in public_events(events,seat):mirror.update(json.dumps(event))
            if list(mirror.tehai)!=[sum(t//4==i for t in obs.hand) for i in range(34)]:raise ValueError('Teacher history/own-hand mismatch')
            types={c['type'] for c in q['criteria'].values()};cans=mirror.last_cans;detail=dict(id=identity)
            for kind,name in fields.items():
                expected=kind in types;actual=bool(getattr(cans,name))
                if expected:counts['native_offered_'+kind]+=1
                if expected!=actual:mismatches.append(dict(id=identity,kind=kind,native=expected,mortal=actual))
            if 'daiminkan' in types:
                encoded,mask=mirror.encode_obs(4,False);indices,values,_,_=actor.engine.react_batch([encoded],[mask],None)
                source=q['criteria'][q['label']]['type'];expected='kan' if source in ('ankan','kakan','daiminkan') else source
                direct=mapped_index(indices[0]);counts['daiminkan_q_queries']+=1;counts['encoded_kan42_available']+=bool(mask[42])
                counts['daiminkan_source_'+source]+=1;counts['daiminkan_direct_'+direct]+=1
                detail.update(kan42_available=bool(mask[42]),source=source,direct=direct,q_kan_minus_best=float(values[0][42]-max(values[0])))
                if expected!=direct:q_mismatches.append(detail)
            counts['fixtures']+=1;stream.write(json.dumps(detail)+'\n')
    if not all(sha(p)==h for p,h in binding.items()):raise ValueError('Diagnostic sources changed')
    result=dict(status='representative_true_history_mortal_mask_and_daiminkan_q_audit_complete',counts=dict(counts),
        eligibility_mismatch_count=len(mismatches),mismatches=mismatches,daiminkan_q_disagreement_count=len(q_mismatches),q_disagreements=q_mismatches,
        representative_first64_not_full256_eligibility_sweep=True,labels_generated=0,student_fit_changed=False,raw_details_private=True)
    write(out/'summary-private.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('basic','advanced','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();audit(a.basic.resolve(),a.advanced.resolve(),a.out.resolve())
