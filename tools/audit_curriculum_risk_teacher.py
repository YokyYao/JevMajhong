"""Reconstruct new risk choices, encoder inputs, replay splits and guard diagnostics."""
import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tomllib
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.collect_curriculum_endgame_cases import sha


def visible(row):
    return dict(state=row['state'],questions={k:{a:b for a,b in q.items() if a in ('type','instructions','criteria')} for k,q in row['questions'].items()})


def fingerprint(row):return hashlib.sha256(json.dumps(visible(row),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def audit(source,out):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS')
    assert not socket.gethostname().startswith(('lo-','login'))
    out.mkdir(parents=True,exist_ok=False)
    read=lambda p:json.loads(p.read_text())
    registration=read(source/'registered-private.json');ready=read(source/'summary-private.json')
    if ready['status']!='risk_true_trajectory_teacher_sources_complete_not_fit_admitted' or ready['stats']['complete_sources']!=256 or ready['stats']['identical_parent_trajectories']!=256:
        raise ValueError('The complete prospective256-family cohort is required')
    if not all(sha(p)==h for p,h in registration['files'].items()):raise ValueError('Collected dependencies changed')
    reg=read(ROOT/registration['config']['source_registration']);sys.path.insert(0,reg['source']);sys.path.insert(0,reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from riichienv import RiichiEnv,GameRule
    from jevmahjong.core.state import build_request,legal_candidates
    from jevmahjong.data.base_lora_pool import normalized_request
    from jevmahjong.policy.unified import serving_request
    from jevmahjong.evaluation.agents import guarded_indices,heuristic_index
    from tools.curriculum_risk_teacher import category,guard_diagnostic
    from jevany.api import SystemOneRequest,to_record
    from jevany.data import api_request
    from jevany.model import DecisionModel,load_preprocessor
    from tools.curriculum_endgame_ledger import ledger
    files=dict(registration['files']);files[str(Path(__file__).resolve())]=sha(__file__)
    files[str(source/'summary-private.json')]=sha(source/'summary-private.json')
    raw=source/'source-matches-private.jsonl'
    if sha(raw)!=ready['full_source_matches_sha256']:raise ValueError('Raw source matches changed')
    files[str(raw)]=sha(raw)
    families={r['seed']:r for r in registration['sources']};starts={};complete_sources=set()
    for line in raw.open():
        match=json.loads(line);family=match['source'];seed=family['seed']
        if family!=families[seed] or seed in complete_sources:raise ValueError('Missing, duplicate or mis-split source match')
        complete_sources.add(seed);ledger(match['mjai'],match['terminal_scores'])
        for index,event in enumerate(e for e in match['mjai'] if e['type']=='start_kyoku'):starts[(seed,index)]=event
    if complete_sources!=set(families):raise ValueError('Some registered source matches are absent')
    ancestral_path=ROOT/'runs/curriculum-reproduction-v1/clean-data-v1/data-ready.json';ancestral=read(ancestral_path)
    reserved_path=ancestral_path.parent/'reserved-public-groups.json';reserved=set(read(reserved_path))
    files[str(ancestral_path)]=sha(ancestral_path);files[str(reserved_path)]=sha(reserved_path)
    ancestor_groups={p:set() for p in ('train','validation')};ancestor_inputs={p:set() for p in ancestor_groups};ancestor_ids={p:set() for p in ancestor_groups}
    for specs in ancestral['stages'].values():
        for split,spec in specs.items():
            path=ROOT/spec['path']
            if sha(path)!=spec['sha256']:raise ValueError('Ancestral public data changed')
            files[str(path)]=spec['sha256']
            for line in path.open():
                row=json.loads(line);ancestor_groups[split].add(row['_meta']['group_id']);ancestor_ids[split].add(row['_meta']['id']);ancestor_inputs[split].add(fingerprint(row))
    basic_path=ROOT/'runs/curriculum-basic-teacher-data-v1/audit-v1/summary-private.json';basic=read(basic_path)
    if basic['status']!='basic_teacher_native_normalized_encoder_and_available_ancestral_audit_passed':raise ValueError('Verified basic-course replay required')
    files[str(basic_path)]=sha(basic_path)
    for split,spec in basic['files'].items():
        path=ROOT/spec['path']
        if sha(path)!=spec['sha256']:raise ValueError('Basic-course teacher replay changed')
        files[str(path)]=spec['sha256']
        for line in path.open():
            row=json.loads(line);ancestor_groups[split].add(row['_meta']['group_id']);ancestor_ids[split].add(row['_meta']['id']);ancestor_inputs[split].add(fingerprint(row))
    calls_path=ROOT/'runs/curriculum-calls-repair-data-v1/data/data-ready-private.json';calls=read(calls_path)
    if calls['status']!='calls_repair12_source_data_prepared_not_parent_or_fit_admitted':raise ValueError('Complete prepared calls replay required')
    files[str(calls_path)]=sha(calls_path)
    for split,spec in calls['files'].items():
        p=ROOT/spec['path']
        if sha(p)!=spec['sha256']:raise ValueError('Prepared calls replay changed')
        files[str(p)]=spec['sha256']
        for line in p.open():
            r=json.loads(line);ancestor_groups[split].add(r['_meta']['group_id']);ancestor_ids[split].add(r['_meta']['id']);ancestor_inputs[split].add(fingerprint(r))
    recipe=tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    tokenizer=load_preprocessor(reg['training_config']['base_load_path'],backbone_adapter='text');proxy=SimpleNamespace(decision_mode='pointer',option_isolation=False)
    groups={p:set() for p in ancestor_groups};inputs={p:set() for p in groups};ids=set();counts={p:Counter() for p in groups};tokens={p:Counter() for p in groups};outputs={}
    for split,spec in ready['files'].items():
        path=ROOT/spec['path']
        if sha(path)!=spec['sha256']:raise ValueError('Teacher decision file changed')
        files[str(path)]=spec['sha256'];destination=out/f'{split}.jsonl'
        with destination.open('x') as stream:
            for line in path.open():
                row=json.loads(line);meta=row['_meta'];state=row['state'];q=row['questions']['action'];family=families[meta['source_logical_seed']]
                if meta['split']!=split or family['split']!=split or meta['group_id']!=family['group_id'] or meta['id'] in ids:
                    raise ValueError('Teacher split or decision identity changed')
                ids.add(meta['id']);groups[split].add(meta['group_id']);start=starts[(family['seed'],meta['source_hand_index'])]
                if start['scores']!=meta['source_initial_current_hand_scores']:raise ValueError('Recorded initial scores differ from actual source')
                env=RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=meta['source_engine_seed']+meta['source_hand_index'])
                observations=env.reset(oya=state['dealer'],round_wind=state['round_wind'],honba=state['honba'],scores=start['scores'],kyotaku=start['kyotaku'])
                identity=lambda a:(int(a.action_type),a.tile,tuple(sorted(a.consume_tiles)))
                for batch in meta['source_trace']:
                    if {a['player'] for a in batch}!=set(observations):raise ValueError('A source batch omitted an active actor')
                    actions={}
                    for saved in batch:
                        menu={identity(a):a for a in legal_candidates(observations[saved['player']])}
                        actions[saved['player']]=menu[(saved['type'],saved['tile'],tuple(sorted(saved['consumed'])))]
                    observations=env.step(actions)
                obs=observations[meta['seat']];rebuilt,menu,feats=build_request(obs)
                if rebuilt!=visible(row):raise ValueError('Actual native public input or action menu differs from saved teacher row')
                item=normalized_request(row,meta['group_id'],'expert_replay',meta['source_step'])
                if serving_request(rebuilt)!=visible(item):raise ValueError('Normalized training/serving input mismatch')
                if item['questions']['action']['label']!=q['label']:raise ValueError('Normalization relabelled teacher choice')
                expected=category({c['type'] for c in q['criteria'].values()},state)
                if meta['curriculum_stratum']!=expected:raise ValueError('Risk source category differs from its actual native menu')
                diagnostic=guard_diagnostic(obs,menu,feats,rebuilt,q['label'],guarded_indices,heuristic_index)
                if diagnostic!=meta['guard_diagnostic']:raise ValueError('Actual serving guard/feature diagnostic differs')
                item['_meta']=copy.deepcopy(meta);item['questions']['action']['src']='curriculum3:teacher:'+meta['curriculum_stratum']
                key=fingerprint(item)
                if key in inputs[split]:raise ValueError('Duplicate observed input in new teacher split')
                inputs[split].add(key)
                if key in ancestor_inputs['validation' if split=='train' else 'train']:
                    raise ValueError('New teacher input overlaps opposite ancestral split')
                if key in ancestor_inputs[split]:raise ValueError('Additional risk data duplicate an existing same-split model input')
                record,mapping=to_record(SystemOneRequest.model_validate(api_request(item)))
                encoded=DecisionModel.encode(proxy,tokenizer,record,max_state=recipe['max_state'],max_branch=recipe['max_branch'],strict=True)
                if encoded['state_truncated'] or len(encoded['ids'])>recipe['max_packed'] or mapping[0]['keys']!=list(q['criteria']):
                    raise ValueError('Teacher input truncated or legal menu reordered by encoder')
                counts[split][meta['curriculum_stratum']]+=1;counts[split]['rows']+=1
                counts[split]['guard:'+diagnostic['reason']]+=1
                counts[split]['opponent_riichi_'+str(diagnostic['opponent_riichi'])]+=1
                tokens[split]['max_packed']=max(tokens[split]['max_packed'],len(encoded['ids']))
                tokens[split]['max_state']=max(tokens[split]['max_state'],encoded['seg'].count(0))
                stream.write(json.dumps(item,ensure_ascii=False,separators=(',',':'))+'\n')
        outputs[split]=dict(path=str(destination.relative_to(ROOT)),sha256=sha(destination),records=counts[split]['rows'],groups=len(groups[split]))
    if groups['train']&(groups['validation']|ancestor_groups['validation']|reserved) or groups['validation']&ancestor_groups['train']:
        raise ValueError('Ancestral held-out source family enters the opposite new split')
    if inputs['train']&inputs['validation'] or ids&(ancestor_ids['train']|ancestor_ids['validation']):raise ValueError('Cross-split input or ancestral decision ID overlap')
    if not all(sha(p)==h for p,h in files.items()):raise ValueError('Bound collection/audit inputs changed')
    summary=dict(status='risk_teacher_native_normalized_encoder_replay_and_guard_audit_passed',files=outputs,
        counts={p:dict(c) for p,c in counts.items()},tokens={p:dict(c) for p,c in tokens.items()},
        checked_complete_source_games=len(complete_sources),checked_native_decisions=len(ids),
        normalization_matches_live_unified_serving=True,source_labels_unchanged=True,source_splits_unchanged=True,
        cross_split_observed_input_overlap=0,available_ancestral_groups_inputs_and_ids_checked=True,
        reserved_ancestral_groups_excluded=len(reserved),raw_private=True,student_fit_admitted=False,student_optimizer_updates=0,
        original_basic_families_and_source_trajectories_reused=True,additional_independent_families=0,basic_replay_cross_split_exclusions_checked=True,
        no_duplicate_same_split_model_inputs=True,full_calls_replay_split_exclusions_checked=True,
        every_actual_guard_diagnostic_reproduced=True,current_policy_and_guard_unchanged=True,
        note='Checks all available earlier-course public datasets/group reservations; does not infer unrecorded historical concealed-wall identities. New source seed ranges were registered before generation.')
    with (out/'audit-binding-private.json').open('x') as f:json.dump(dict(files=files,node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID']),f,indent=2);f.write('\n')
    with (out/'summary-private.json').open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit(a.source.resolve(),a.out.resolve())
