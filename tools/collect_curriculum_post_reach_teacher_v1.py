"""Missing immediate post-reach discard choices from the same actual source families."""
import argparse
from collections import Counter
from importlib.metadata import version
import json
import os
from pathlib import Path
import socket
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.collect_curriculum_endgame_cases import sha
from tools.curriculum_post_reach_teacher_v1 import category,guard_diagnostic


def families(config,parent):
    from tools.collect_curriculum_basic_teacher import sources
    if (config['method']!='same_basic_families_additional_post_reach_mortal_choices_v1' or config['source_games']!=256
        or config['preserve_all_parent_family_splits'] is not True or config['require_identical_complete_parent_trajectories'] is not True
        or config['capture']!=['post_reach_discard'] or config['filter_by_success_or_selected_action'] is not False
        or config['exclude_existing_teacher_contexts'] is not True or config['keep_current_guard_unchanged'] is not True
        or len(config['existing_teacher_audits'])!=3 or len(set(config['existing_teacher_audits']))!=3
        or config['require_previous_actual_action_reach'] is not True
        or config['expected_new_records_from_independent_context_audit']!={'train':312,'validation':75}
        or config['automatic_student_fit'] is not False):raise ValueError('Exact prospective data-only risk recipe required')
    expected=sources(parent['config'])
    if parent['sources']!=expected:raise ValueError('Original whole-family split changed')
    return expected


def collect(config_path):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS')
    assert not socket.gethostname().startswith(('lo-','login'))
    read=lambda p:json.loads(p.read_text());config=read(config_path);parent_root=ROOT/config['parent_source']
    parent=read(parent_root/'registered-private.json');ready=read(parent_root/'summary-private.json');cohort=families(config,parent)
    if ready['status']!='basic_true_trajectory_teacher_sources_complete_not_fit_admitted' or ready['stats']['complete_sources']!=256:
        raise ValueError('Completed basic source required')
    if not all(sha(p)==h for p,h in parent['files'].items()):raise ValueError('Original source dependencies changed')
    old_raw=parent_root/'source-matches-private.jsonl'
    if sha(old_raw)!=ready['full_source_matches_sha256']:raise ValueError('Original complete trajectories changed')
    originals={}
    for line in old_raw.open():
        item=json.loads(line);seed=item['source']['seed']
        if seed in originals:raise ValueError('Duplicate original source match')
        originals[seed]=item
    if set(originals)!={s['seed'] for s in cohort}:raise ValueError('Original cohort is incomplete')
    reg_path=ROOT/config['source_registration'];reg=read(reg_path);sys.path.insert(0,reg['source']);sys.path.insert(0,reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from riichienv import RiichiEnv,GameRule
    import riichienv._riichienv as native
    from jevmahjong.core.state import build_request,legal_candidates,signature
    from jevmahjong.core.seeds import engine_seed
    from jevmahjong.data.base_lora_pool import phase
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.agents import guarded_indices,heuristic_index
    from tools.curriculum_mortal_source import MortalSource
    from tools.curriculum_case_order import ordered_seats,canonical_trace
    from tools.curriculum_endgame_ledger import ledger
    if version('riichienv')!='0.4.11' or package_digest(Path(reg['source'])/'jevmahjong')!=reg['source_sha256']:raise ValueError('Pinned native/project changed')
    out=ROOT/config['out'];out.mkdir(parents=True,exist_ok=False)
    actor=MortalSource((ROOT/config['teacher_root']).resolve())
    existing=set();existing_paths=[]
    for directory in config['existing_teacher_audits']:
        summary_path=ROOT/directory/'summary-private.json';summary=read(summary_path);existing_paths.append(summary_path)
        if summary['status'] not in ('basic_teacher_native_normalized_encoder_and_available_ancestral_audit_passed',
            'advanced_teacher_native_normalized_encoder_basic_and_ancestral_audit_passed',
            'risk_teacher_native_normalized_encoder_replay_and_guard_audit_passed'):
            raise ValueError('Actual audited basic/advanced source records required')
        for spec in summary['files'].values():
            p=ROOT/spec['path'];existing_paths.append(p)
            if sha(p)!=spec['sha256']:raise ValueError('Existing audited teacher records changed')
            for line in p.open():
                m=json.loads(line)['_meta'];existing.add((m['source_logical_seed'],m['source_hand_index'],m['source_step'],m['seat']))
    paths=[Path(__file__),config_path,reg_path,parent_root/'registered-private.json',parent_root/'summary-private.json',old_raw,
        Path(native.__file__),ROOT/'tools/curriculum_case_order.py',ROOT/'tools/curriculum_endgame_ledger.py',
        ROOT/'tools/collect_curriculum_basic_teacher.py',ROOT/'tools/curriculum_post_reach_teacher_v1.py',
        ROOT/'docs/experiments/plans/curriculum-post-reach-teacher-data-20261123.md',*existing_paths]
    binding={str(p.resolve()):sha(p) for p in paths};binding.update(actor.bindings)
    def write(p,value):
        with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    write(out/'registered-private.json',dict(status='same_family_post_reach_teacher_choices_registered_before_generation',config=config,
        sources=cohort,files=binding,node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'],additional_independent_families=0))
    (out/'source-matches-private.jsonl').symlink_to(old_raw.resolve())
    paths={split:out/f'{split}-teacher-private.jsonl' for split in ('train','validation')}
    streams={split:p.open('x') for split,p in paths.items()};counts={s:Counter() for s in paths};stats=Counter();ids=set()
    try:
        for source in cohort:
            seed=engine_seed(source['seed'],'spaced-v1');old=originals[source['seed']]
            if old['source']!=source or old['engine_seed']!=seed:raise ValueError('Original source metadata differs')
            env=RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=seed);observations=env.reset()
            agents=[actor.player(p) for p in range(4)];trace=[];hand_index=0;marker=(env.round_wind,env.kyoku_idx,env.honba)
            for step in range(30000):
                if env.done():break
                current=(env.round_wind,env.kyoku_idx,env.honba)
                if current!=marker:trace=[];marker=current;hand_index+=1
                selected={p:agents[p].act(obs) for p,obs in ordered_seats(observations)}
                for seat,obs in ordered_seats(observations):
                    actions=legal_candidates(obs)
                    if len(actions)<2:continue
                    kinds={json.loads(a.to_mjai())['type'] for a in actions}
                    public_state=obs.to_dict();context=(source['seed'],hand_index,step,seat)
                    if kinds!={'dahai'} or context in existing:continue
                    # Select only from actual public state, before inspecting
                    # the teacher's chosen discard or the hand's outcome.
                    previous_reach=bool(trace) and seat in trace[-1] and json.loads(trace[-1][seat].to_mjai())['type']=='reach'
                    if not previous_reach or not public_state['riichi_declared'][seat]:continue
                    request,menu,features=build_request(obs)
                    slot=category(kinds,request['state'],previous_reach=previous_reach,existing=context in existing)
                    if slot is None:continue
                    keys=list(request['questions']['action']['criteria'])
                    chosen=signature(selected[seat]);label=next(k for k,a in zip(keys,menu) if signature(a)==chosen)
                    chosen_type=request['questions']['action']['criteria'][label]['type']
                    identifier=f'postreach:{source["seed"]}:{hand_index}:{step}:{seat}'
                    if identifier in ids:raise ValueError('Duplicate additional risk teacher decision')
                    ids.add(identifier);request['questions']['action'].update(label=label,src='post_reach_teacher:'+slot)
                    request['_meta']=dict(id=identifier,group_id=source['group_id'],split=source['split'],phase=phase(request),
                        label_type='expert_replay',source='mortal_true_trajectory_additional_post_reach_choices',curriculum_stratum=slot,
                        source_logical_seed=source['seed'],source_engine_seed=seed,source_hand_index=hand_index,source_step=step,seat=seat,
                        source_trace=canonical_trace([[dict(player=p,type=int(a.action_type),tile=a.tile,consumed=list(a.consume_tiles)) for p,a in batch.items()] for batch in trace]),
                        source_initial_current_hand_scores=list(next(e['scores'] for e in reversed(env.mjai_log) if e['type']=='start_kyoku')),
                        teacher_actual_choice_not_expert_optimality=True,previous_actual_own_action='reach',
                        guard_diagnostic=guard_diagnostic(obs,menu,features,request,label,guarded_indices,heuristic_index))
                    streams[source['split']].write(json.dumps(request,ensure_ascii=False)+'\n')
                    c=counts[source['split']];c['rows']+=1;c['stratum_'+slot]+=1;c[slot+':selected_'+chosen_type]+=1
                    c['opponent_riichi_'+str(sum(p!=seat for p in request['state']['riichi']))]+=1
                    diagnostic=request['_meta']['guard_diagnostic'];c['guard:'+diagnostic['reason']]+=1
                    c['teacher_safe_to_all_threats']+=diagnostic['feature_safe_to_all_threats']
                    c['own_discard_number_'+str(len(public_state['discards'][seat])+1)]+=1
                    for kind in sorted(kinds&{'chi','pon','ankan','daiminkan','kakan'}):
                        c['offered_'+kind]+=1;c[kind+':selected_same' if kind==chosen_type else kind+':selected_pass' if chosen_type=='none' else kind+':selected_other']+=1
                trace.append(selected);observations=env.step(selected)
            if not env.done():raise RuntimeError('Bounded advanced source did not complete')
            ledger(env.mjai_log,env.scores())
            if env.mjai_log!=old['mjai'] or list(env.scores())!=old['terminal_scores']:
                write(out/'trajectory-mismatch-private.json',dict(source=source,new_mjai=env.mjai_log,new_scores=list(env.scores()),old_source_reference=str(old_raw)))
                raise ValueError('Regenerated full source differs; preserve failure, do not admit a different trajectory')
            stats['complete_sources']+=1;stats['identical_parent_trajectories']+=1
            for stream in streams.values():stream.flush()
            if stats['complete_sources']%16==0:print(json.dumps(dict(stage='post_reach_teacher_collection',sources=stats['complete_sources'],counts={k:dict(v) for k,v in counts.items()})),flush=True)
    except BaseException as error:
        write(out/'error-private.json',dict(type=type(error).__name__,error=str(error),stats=dict(stats)));raise
    finally:
        for stream in streams.values():stream.close()
    if not all(sha(p)==h for p,h in binding.items()):raise ValueError('Advanced source bindings changed')
    if {split:counts[split]['rows'] for split in counts}!=config['expected_new_records_from_independent_context_audit']:
        raise ValueError('Actual missing post-reach count differs from the independent native audit')
    result=dict(status='post_reach_true_trajectory_teacher_sources_complete_not_fit_admitted',stats=dict(stats),
        counts={k:dict(v) for k,v in counts.items()},files={split:dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for split,p in paths.items()},
        full_source_matches_sha256=sha(old_raw),source_matches_reuse_parent_reference=True,additional_independent_families=0,
        actual_teacher_labels_not_censored_event_inferences=True,source_ledger_and_full_parent_trajectory_equality=True,
        selection_uses_actual_previous_reach_not_outcomes=True,existing_teacher_contexts_excluded=True,current_serving_guard_unchanged=True,
        student_fit_admitted=False,student_optimizer_updates=0,raw_private=True,complete_native_normalized_encoder_ancestral_audit_pending=True)
    write(out/'summary-private.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args();collect(a.config.resolve())
