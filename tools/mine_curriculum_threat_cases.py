"""Mine missed threat strata from registered physical prefixes; no new games."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.collect_curriculum_endgame_cases import sha,category
from tools.curriculum_case_order import canonical_trace,ordered_seats


def threat_stratum(request,phase):
    state=request['state'];seat=state['seat']
    threats=sum(p!=seat for p in state['riichi'])
    kind=category(request['questions']['action']['criteria'],phase,True)
    if kind not in ('progression_draw','call_response') or threats not in (1,2,3):return None
    return kind,threats


def mine(parent_root,out):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS')
    assert not socket.gethostname().startswith(('lo-','login'))
    out.mkdir(parents=True,exist_ok=False)
    read=lambda path:json.loads(path.read_text())
    ready=read(parent_root/'data/data-ready-private.json');registration=read(parent_root/'data/source-registration-private.json')
    proof=read(parent_root/'preparation-execution-private.json')
    if not proof['repeated_all_rows_identical']:raise ValueError('A verified deterministic source cohort is required')
    reg=read(ROOT/registration['config']['source_registration'])
    sys.path.insert(0,reg['source'])
    from riichienv import RiichiEnv,GameRule,Phase,ActionType
    from jevmahjong.core.state import build_request,legal_candidates,signature
    from tools.curriculum_endgame_contexts_v2 import score_contexts,replay_score_context
    from tools.curriculum_endgame import in_endgame_scope,public_placement
    bindings=dict(registration['files'])
    for path in (Path(__file__),ROOT/'tests/core/test_curriculum_threat_mining.py',parent_root/'preparation-execution-private.json',parent_root/'data/data-ready-private.json'):
        bindings[str(path.resolve())]=sha(path)
    pool=[];parent_ids=set()
    for split,spec in ready['files'].items():
        path=ROOT/spec['path']
        if sha(path)!=spec['sha256']:raise ValueError('Parent source inputs changed')
        bindings[str(path.resolve())]=spec['sha256']
        for line in path.open():
            row=json.loads(line);parent_ids.add(row['_meta']['id'])
            if row['_meta']['score_context']=='hold_first':pool.append(row)
    if not all(sha(path)==digest for path,digest in bindings.items()):raise ValueError('A source binding changed')
    anchors={}
    for row in pool:
        m=row['_meta'];key=(m['source_logical_seed'],m['source_hand_index'])
        previous=anchors.get(key)
        if previous is not None:
            left=canonical_trace(previous['_meta']['source_trace']);right=canonical_trace(m['source_trace'])
            if left[:min(len(left),len(right))]!=right[:min(len(left),len(right))]:raise ValueError('Same source hand contains different physical prefixes')
        if previous is None or len(m['source_trace'])>len(previous['_meta']['source_trace']):anchors[key]=row
    source_reg=dict(status='threat_prefix_rule_registered_before_mining',
        config=dict(source_registration=registration['config']['source_registration'],method='source_prefix_threat_strata_v1'),
        sources=registration['sources'],files=bindings,
        rule='first_public_progression_or_call_per_family_per_opponent_riichi_count1_2_3_sorted_time_then_seat',
        cap_one_per_family_category_threat=True,preserve_parent_whole_family_split=True,
        source_prefixes_censored_at_registered_case_windows=True,additional_source_games=0,
        teacher_selected=False,labels_generated=False,training_parent_admitted=False)
    with (out/'source-registration-private.json').open('x') as f:json.dump(source_reg,f,indent=2);f.write('\n')
    streams={p:(out/f'{p}-unlabelled-private.jsonl').open('x') for p in ('train','validation')}
    counts=Counter();strata=Counter();seen=set();ids=set();groups={p:set() for p in streams}
    try:
        for (logical,index),anchor in sorted(anchors.items()):
            meta=anchor['_meta'];state=anchor['state'];seat=meta['seat'];native_seed=meta['source_engine_seed']
            env=RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=native_seed+index)
            observations=env.reset(oya=state['dealer'],round_wind=state['round_wind'],honba=state['honba'],
                scores=meta['initial_scores'],kyotaku=meta['initial_sticks'])
            trace=[];offset=meta['source_step']-len(meta['source_trace'])
            for position,batch in enumerate(meta['source_trace']):
                key=lambda a:(int(a.action_type),a.tile,tuple(sorted(a.consume_tiles)))
                if {saved['player'] for saved in batch}!=set(observations):raise ValueError('Incomplete saved joint source action')
                selected={}
                for saved in batch:
                    menu={key(a):a for a in legal_candidates(observations[saved['player']])}
                    wanted=(saved['type'],saved['tile'],tuple(sorted(saved['consumed'])))
                    if wanted not in menu:raise ValueError('Source-prefix action became illegal')
                    selected[saved['player']]=menu[wanted]
                for actor,obs in ordered_seats(observations):
                    request,menu,_=build_request(obs)
                    bucket=threat_stratum(request,'draw' if env.phase==Phase.WaitAct else 'response')
                    if bucket is None:continue
                    kind,threats=bucket;family=(logical,kind,threats)
                    if family in seen:continue
                    if not in_endgame_scope(request['state']):raise ValueError('Saved late prefix left the declared scope')
                    seen.add(family);contexts=score_contexts(actor,env.riichi_sticks)
                    chosen=signature(selected[actor]);control=next(k for k,a in zip(request['questions']['action']['criteria'],menu) if signature(a)==chosen)
                    prefix=canonical_trace([[dict(player=p,type=int(a.action_type),tile=a.tile,consumed=list(a.consume_tiles)) for p,a in b.items()] for b in trace])
                    for name,context in contexts.items():
                        rid=f's4:{logical}:{index}:{offset+position}:{actor}:{kind}:{name}'
                        if rid in parent_ids:counts['existing_parent_score_views_not_duplicated']+=1;continue
                        if rid in ids:raise ValueError('Duplicate mined decision')
                        fork,receipt=replay_score_context(env,native_seed,index,trace,actor,context['scores'])
                        row=build_request(fork.get_obs_py([actor])[actor])[0]
                        placement=public_placement(row['state'])
                        if placement['rank']!=context['rank']:raise ValueError('Augmented rank differs')
                        row['_meta']=dict(id=rid,group_id=meta['group_id'],split=meta['split'],
                            label_type='unlabelled_terminal_context',source_logical_seed=logical,source_engine_seed=native_seed,
                            source_hand_index=index,source_step=offset+position,seat=actor,category=kind,score_context=name,
                            source_profile=meta['source_profile'],source_control_key=control,source_trace=prefix,
                            placement=placement,initial_scores=receipt['initial_scores'],initial_sticks=receipt['initial_sticks'],
                            same_wall_score_only_replay_verified=True,current_native_world_adapter_supported=False,
                            source_mining='registered_prefix_threat_strata_v1',opponent_riichi_count=threats,labels_generated=False)
                        streams[meta['split']].write(json.dumps(row,ensure_ascii=False)+'\n')
                        ids.add(rid);groups[meta['split']].add(meta['group_id']);counts['rows']+=1
                        strata[f'{meta["split"]}:{kind}:opponent_riichi_{threats}']+=1
                for actor,obs in observations.items():
                    if any(a.action_type in (ActionType.RON,ActionType.TSUMO) for a in legal_candidates(obs)) and selected[actor].action_type not in (ActionType.RON,ActionType.TSUMO):
                        raise ValueError('Saved prefix would decline a legal win')
                trace.append(selected);observations=env.step(selected)
                if env.done():raise ValueError('Prefix settled before its registered anchor')
    finally:
        for stream in streams.values():stream.close()
    if groups['train']&groups['validation'] or not ids:raise ValueError('Missing new cases or split overlap')
    if not all(sha(path)==digest for path,digest in bindings.items()):raise ValueError('Source inputs changed during mining')
    files={p:dict(path=str((out/f'{p}-unlabelled-private.jsonl').relative_to(ROOT)),sha256=sha(out/f'{p}-unlabelled-private.jsonl')) for p in streams}
    result=dict(status='unlabelled_threat_prefix_cases_mined',counts=dict(counts),strata=dict(strata),files=files,
        actual_groups={p:len(v) for p,v in groups.items()},registered_source_groups=ready['registered_source_groups'],
        additional_source_games=0,parent_rows_not_duplicated=True,source_prefixes_censored_at_registered_case_windows=True,
        source_split_registered_before_generation=True,all_variants_share_source_group=True,source_legal_win_declines=0,
        labels_generated=0,teacher_selected=False,training_parent_admitted=False,stage4_fit_admitted=False,
        complete_ancestral_source_audit_pending=True,actual_fixed_tokenizer_admission_pending=True)
    with (out/'data-ready-private.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();mine(a.parent_root.resolve(),a.out.resolve())
