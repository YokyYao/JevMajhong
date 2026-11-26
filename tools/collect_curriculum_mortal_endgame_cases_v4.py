"""Replay the same Mortal cohort with deterministic simultaneous seat order.

Independent coverage supplement after v1's closed-hand gap; no labels/fit.
"""
import argparse
from collections import Counter
import datetime
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def cohort(config):
    start,count=config['seed_start'],config['source_games']
    period,residue=config['validation_period'],config['validation_residue']
    if (type(start) is not int or type(count) is not int or not 5<=count<=512
        or type(period) is not int or period<2 or type(residue) is not int or residue not in range(period)):
        raise ValueError('Explicit bounded whole-source cohort/split required')
    seeds=range(start,start+count)
    for protected in config['protected_source_ranges']:
        if set(seeds)&set(range(protected['start'],protected['start']+protected['count'])):
            raise ValueError('A protected development/final/diagnostic family enters collection')
    result=[dict(seed=seed,group_id=f'stage4-native-family:{seed}',
        split='validation' if offset%period==residue else 'train',hero_seat=offset%4,
        profile='calling' if offset%2 else 'efficiency') for offset,seed in enumerate(seeds)]
    if {row['split'] for row in result}!={'train','validation'}:raise ValueError('Both whole-source splits required')
    return result


def category(criteria,phase,is_hero):
    kinds={value['type'] for value in criteria.values()}
    if 'hora' in kinds:return 'win'
    if not is_hero or len(criteria)<2:return None
    if phase=='draw' and 'dahai' in kinds:return 'progression_draw'
    if phase=='response' and kinds&{'chi','pon','daiminkan'} and 'none' in kinds:return 'call_response'
    return None


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def collect(config_path):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-','login'))
    config=json.loads(config_path.read_text());sources=cohort(config)
    old_config_path=ROOT/config['replays_failed_cohort_config']
    old_config=json.loads(old_config_path.read_text())
    if sources!=cohort(old_config) or old_config['source_behavior']!=config['source_behavior']:
        raise ValueError('Retry must preserve the failed registered source cohort/split/behavior')
    if (config['method']!='stage4_mortal_grouped_score_context_inputs_v4'
        or config['simultaneous_seat_order']!='ascending_absolute_seat'
        or config['source_registration_is_code_reference_not_training_parent'] is not True
        or config['source_split_registered_before_generation'] is not True
        or config['per_source_category_caps']!={'progression_draw':1,'call_response':1,'win':1}
        or config['labels_generated'] is not False or config['automatic_fit'] is not False
        or config['training_parent_admitted'] is not False
        or config['source_behavior']!='mortal_all_seats_forced_legal_wins_input_collection_v1'
        or config['game_mode']!='4p-red-east' or config['seed_scheme']!='spaced-v1'
        or config['late_scope']!='scheduled_east4_and_extensions'
        or config['group_scope']!='entire_logical_source_family_all_hands_actors_and_score_contexts'
        or config['score_contexts']!=['hold_first','hold_second','third_gap3000','fourth_gap5000','fourth_gap13000']):
        raise ValueError('Explicit unlabelled collection recipe required')
    out=ROOT/config['out'];out.mkdir(parents=True,exist_ok=False)
    reg_path=ROOT/config['source_registration'];reg=json.loads(reg_path.read_text())
    sys.path.insert(0,reg['source']);sys.path.insert(0,reg['training_config']['upstream_source'])
    from riichienv import RiichiEnv,GameRule,Phase,ActionType
    import riichienv._riichienv as native
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.core.state import legal_candidates,build_request,signature
    from jevmahjong.core.seeds import engine_seed
    from jevmahjong.core.scoring import settlement_ledger
    from tools.curriculum_mortal_source import MortalSource
    from tools.curriculum_case_order import ordered_seats,canonical_trace
    from tools.curriculum_endgame import in_endgame_scope,public_placement
    from tools.curriculum_endgame_contexts_v2 import score_contexts,replay_score_context
    from tools.curriculum_win_context import build_win_request
    if version('riichienv')!=config['engine_version'] or package_digest(Path(reg['source'])/'jevmahjong')!=reg['source_sha256']:
        raise ValueError('Pinned native/source package changed')
    paths=[config_path,reg_path,Path(__file__),Path(native.__file__),ROOT/'tools/curriculum_endgame.py',
        ROOT/'tools/curriculum_endgame_contexts_v2.py',ROOT/'tools/curriculum_win_context.py',
        ROOT/'tests/core/test_curriculum_endgame_cases.py',ROOT/'tests/core/test_curriculum_mortal_source.py',
        ROOT/'tests/core/test_curriculum_endgame_contexts_v2.py',old_config_path,
        ROOT/'tools/curriculum_case_order.py',ROOT/'tests/core/test_curriculum_case_order.py']
    binding={str(path.resolve()):sha(path) for path in paths}
    actor=MortalSource((ROOT/config['teacher_root']).resolve())
    binding.update(actor.bindings)
    for row in sources:row['profile']='mortal-all-seats-source-only'
    registration=dict(status='unlabelled_source_cohort_registered_before_generation',config=config,
        sources=sources,files=binding,source_package_sha256=reg['source_sha256'],engine_version=version('riichienv'),
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'],
        training_parent_admitted=False,teacher_selected=False,
        source_actor='pinned_mortal_298k_public_events_forced_legal_wins',
        source_actor_is_not_a_terminal_label_teacher=True)
    def write(path,value):
        with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    write(out/'source-registration-private.json',registration)
    stats=Counter();per_split={'train':Counter(),'validation':Counter()};groups={'train':set(),'validation':set()}
    identifiers=set();rows_per_source=[]
    files={split:out/f'{split}-unlabelled-private.jsonl' for split in ('train','validation')}
    streams={split:path.open('x') for split,path in files.items()}
    try:
        for source in sources:
            logical=source['seed'];native_seed=engine_seed(logical,config['seed_scheme'])
            env=RiichiEnv(game_mode=config['game_mode'],rule=GameRule.default_mjsoul(),seed=native_seed)
            observations=env.reset();hero=source['hero_seat']
            agents=[actor.player(p) for p in range(4)]
            seen=set();trace=[];marker=(env.round_wind,env.kyoku_idx,env.honba);source_rows=0
            for step in range(30000):
                if env.done():break
                current=(env.round_wind,env.kyoku_idx,env.honba)
                if current!=marker:trace=[];marker=current
                selected={player:agents[player].act(obs) for player,obs in ordered_seats(observations)}
                if env.round_wind>0 or env.oya==3:
                    for seat,obs in ordered_seats(observations):
                        request,menu,_=build_request(obs)
                        kind=category(request['questions']['action']['criteria'],'draw' if env.phase==Phase.WaitAct else 'response',seat==hero)
                        if kind is None or kind in seen:continue
                        if not in_endgame_scope(request['state']):raise ValueError('Scope differs from public hand fields')
                        contexts=score_contexts(seat,env.riichi_sticks)
                        if not contexts:stats['no_valid_score_context_windows']+=1;continue
                        trigger=env.mjai_log[-1]
                        index=sum(e['type']=='start_kyoku' for e in env.mjai_log)-1
                        control_signature=signature(selected[seat])
                        control_key=next(key for key,action in zip(request['questions']['action']['criteria'],menu) if signature(action)==control_signature)
                        replay_trace=canonical_trace([[dict(player=p,type=int(a.action_type),tile=a.tile,consumed=list(a.consume_tiles))
                            for p,a in batch.items()] for batch in trace])
                        for name,context in contexts.items():
                            if name not in config['score_contexts']:raise ValueError('Unregistered context name')
                            fork,proof=replay_score_context(env,native_seed,index,trace,seat,context['scores'])
                            own=fork.get_obs_py([seat])[seat]
                            enriched=build_win_request(own,fork.mjai_log[-1])[0] if kind=='win' else build_request(own)[0]
                            if any(key in enriched['state'] for key in ('wall','opponent_hands','future','target','label')):
                                raise ValueError('A hidden/target field entered the public request')
                            if enriched['questions']!=request['questions']:raise ValueError('Augmented legal criteria changed')
                            placement=public_placement(enriched['state'])
                            if placement['rank']!=context['rank']:raise ValueError('Declared rank differs after replay')
                            identifier=f's4:{logical}:{index}:{step}:{seat}:{kind}:{name}'
                            if identifier in identifiers:raise ValueError('Duplicate source/context input')
                            identifiers.add(identifier)
                            row=dict(enriched)
                            row['_meta']=dict(id=identifier,group_id=source['group_id'],split=source['split'],
                                label_type='unlabelled_terminal_context',source_logical_seed=logical,source_engine_seed=native_seed,
                                source_hand_index=index,source_step=step,seat=seat,category=kind,score_context=name,
                                source_profile=source['profile'],source_control_key=control_key,source_trace=replay_trace,
                                placement=placement,initial_scores=proof['initial_scores'],initial_sticks=proof['initial_sticks'],
                                same_wall_score_only_replay_verified=True,
                                current_native_world_adapter_supported=kind=='win' and trigger['type'] in ('dahai','tsumo'),
                                labels_generated=False)
                            streams[source['split']].write(json.dumps(row,ensure_ascii=False)+'\n')
                            groups[source['split']].add(source['group_id']);stats['rows']+=1;source_rows+=1
                            per_split[source['split']][kind]+=1;per_split[source['split']]['rank_'+str(placement['rank'])]+=1
                        seen.add(kind);stats['base_'+kind]+=1
                for player,choice in selected.items():
                    win=any(a.action_type in (ActionType.RON,ActionType.TSUMO) for a in legal_candidates(observations[player]))
                    if win and choice.action_type not in (ActionType.RON,ActionType.TSUMO):raise ValueError('Source generator declined a legal win')
                trace.append(selected);observations=env.step(selected)
            if not env.done():raise RuntimeError('Bounded source game did not finish')
            settlement_ledger(env.mjai_log,env.scores());stats['complete_source_games']+=1
            rows_per_source.append(dict(seed=logical,split=source['split'],base_categories=sorted(seen),rows=source_rows))
            for stream in streams.values():stream.flush()
            if stats['complete_source_games']%8==0:
                print(json.dumps(dict(source_games=stats['complete_source_games'],rows=stats['rows'],per_split={p:dict(c) for p,c in per_split.items()})),flush=True)
    except BaseException as error:
        write(out/'collection-error-private.json',dict(type=type(error).__name__,error=str(error),counts=dict(stats)))
        raise
    finally:
        for stream in streams.values():stream.close()
    if groups['train']&groups['validation']:raise ValueError('A logical source family crosses splits')
    if not all(sha(path)==value for path,value in binding.items()):raise ValueError('Registered collection inputs changed')
    ready=dict(status='stage4_grouped_score_inputs_collected_unlabelled',method=config['method'],counts=dict(stats),
        per_split={p:dict(v) for p,v in per_split.items()},actual_groups={p:len(v) for p,v in groups.items()},
        registered_source_groups={p:sum(s['split']==p for s in sources) for p in groups},source_receipts=rows_per_source,
        files={p:dict(path=str(path.relative_to(ROOT)),sha256=sha(path)) for p,path in files.items()},
        source_registration=str((out/'source-registration-private.json').relative_to(ROOT)),
        source_split_registered_before_generation=True,all_variants_share_source_group=True,source_legal_win_declines=0,
        source_and_engine_hashes_unchanged=True,source_ledgers_passed=True,all_score_replays_verified=True,
        actual_fixed_tokenizer_admission_pending=True,complete_ancestral_source_audit_pending=True,
        source_actor_counts=dict(actor.counts),source_actor_is_not_a_terminal_label_teacher=True,
        labels_generated=0,teacher_selected=False,training_parent_admitted=False,stage4_fit_admitted=False,
        nonwin_and_current_kan_label_constructor_extensions_pending=True,unseen_final_seeds_unused=True,
        raw_case_inputs_private=True)
    write(out/'data-ready-private.json',ready)
    print(json.dumps({k:v for k,v in ready.items() if k not in ('source_receipts','files')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();collect(a.config.resolve())
