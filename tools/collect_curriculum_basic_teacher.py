"""Fresh true-trajectory Mortal data for ordinary discard and riichi SFT."""
import argparse
from collections import Counter
import datetime
from importlib.metadata import version
import json
import os
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.collect_curriculum_endgame_cases import sha


def sources(config):
    if (config['method']!='fresh_basic_mortal_true_trajectory_v1' or config['source_games']!=256
        or config['validation_period']!=5 or config['validation_residue']!=4
        or config['plain_own_discard_slots']!=[1,4,8,12]
        or config['all_eligible_riichi_windows'] is not True or config['no_outcome_filter'] is not True
        or config['student_fit_automatic'] is not False or config['game_mode']!='4p-red-east'
        or config['seed_scheme']!='spaced-v1' or config['engine_version']!='0.4.11'):
        raise ValueError('Explicit prospective basic-course source recipe required')
    seeds=range(config['seed_start'],config['seed_start']+config['source_games'])
    for protected in config['protected_source_ranges']:
        if set(seeds)&set(range(protected['start'],protected['start']+protected['count'])):
            raise ValueError('A protected source family enters the new basic course')
    return [dict(seed=seed,group_id=f'basic-mortal-family:{seed}',split='validation' if n%5==4 else 'train') for n,seed in enumerate(seeds)]


def collect(config_path):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS')
    assert not socket.gethostname().startswith(('lo-','login'))
    config=json.loads(config_path.read_text());families=sources(config)
    reg_path=ROOT/config['source_registration'];reg=json.loads(reg_path.read_text())
    sys.path.insert(0,reg['source']);sys.path.insert(0,reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from riichienv import RiichiEnv,GameRule,ActionType,Phase
    import riichienv._riichienv as native
    from jevmahjong.core.state import build_request,legal_candidates,signature
    from jevmahjong.core.seeds import engine_seed
    from jevmahjong.training.base_lora import package_digest
    from tools.curriculum_mortal_source import MortalSource
    from tools.curriculum_case_order import ordered_seats,canonical_trace
    from tools.audit_curriculum_endgame_cases import public_solo_win
    from tools.curriculum_endgame_ledger import ledger
    if version('riichienv')!=config['engine_version'] or package_digest(Path(reg['source'])/'jevmahjong')!=reg['source_sha256']:
        raise ValueError('Pinned engine/project changed')
    out=ROOT/config['out'];out.mkdir(parents=True,exist_ok=False)
    actor=MortalSource((ROOT/config['teacher_root']).resolve())
    paths=[config_path,reg_path,Path(__file__),Path(native.__file__),ROOT/'tools/curriculum_case_order.py',
        ROOT/'tools/audit_curriculum_endgame_cases.py',ROOT/'tools/curriculum_endgame_ledger.py',
        ROOT/'docs/experiments/plans/curriculum-basic-teacher-data-20261122.md']
    binding={str(p.resolve()):sha(p) for p in paths};binding.update(actor.bindings)
    def write(path,value):
        with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    write(out/'registered-private.json',dict(status='basic_teacher_sources_registered_before_generation',config=config,
        sources=families,files=binding,source_package_sha256=reg['source_sha256'],node=socket.gethostname(),
        job_id=os.environ['SLURM_JOB_ID'],utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),student_optimizer_updates=0))
    stats=Counter();coverage={p:Counter() for p in ('train','validation')};identities=set()
    files={p:out/f'{p}-teacher-private.jsonl' for p in coverage};streams={p:path.open('x') for p,path in files.items()}
    raw=(out/'source-matches-private.jsonl').open('x')
    try:
        for family in families:
            seed=engine_seed(family['seed'],config['seed_scheme'])
            env=RiichiEnv(game_mode=config['game_mode'],rule=GameRule.default_mjsoul(),seed=seed);observations=env.reset()
            agents=[actor.player(p) for p in range(4)];seen=set();trace=[];marker=(env.round_wind,env.kyoku_idx,env.honba);hand_index=0
            for step in range(30000):
                if env.done():break
                current=(env.round_wind,env.kyoku_idx,env.honba)
                if current!=marker:trace=[];marker=current;hand_index+=1
                selected={p:agents[p].act(obs) for p,obs in ordered_seats(observations)}
                for seat,obs in ordered_seats(observations):
                    actions=legal_candidates(obs);kinds={a.action_type for a in actions}
                    if selected[seat].action_type in (ActionType.RON,ActionType.TSUMO):
                        score=public_solo_win(env,obs);c=coverage[family['split']];c['source_wins']+=1
                        if score['supported']:
                            c['source_wins_public_no_ura_supported']+=1
                            for yaku in score['yaku']:c['source_win_yaku:'+yaku]+=1
                        else:c['source_win_public_score_unsupported']+=1
                    if (env.phase!=Phase.WaitAct or len(actions)<2 or obs.to_dict()['melds'][seat]
                        or ActionType.DISCARD not in kinds or not kinds<={ActionType.DISCARD,ActionType.RIICHI}):continue
                    discard_number=len(obs.to_dict()['discards'][seat])+1
                    reach=ActionType.RIICHI in kinds
                    slot=(seat,discard_number)
                    if not reach and (discard_number not in config['plain_own_discard_slots'] or slot in seen):continue
                    request,menu,_=build_request(obs)
                    choice=signature(selected[seat]);keys=list(request['questions']['action']['criteria'])
                    label=next(k for k,a in zip(keys,menu) if signature(a)==choice)
                    kind='riichi_yes' if reach and selected[seat].action_type==ActionType.RIICHI else 'riichi_no' if reach else 'plain'
                    identifier=f'basic:{family["seed"]}:{hand_index}:{step}:{seat}'
                    if identifier in identities:raise ValueError('Duplicate teacher decision ID')
                    identities.add(identifier)
                    request['questions']['action'].update(label=label,src='basic_teacher:'+kind)
                    request['_meta']=dict(id=identifier,group_id=family['group_id'],split=family['split'],
                        phase='closed_discard',label_type='expert_replay',source='mortal_true_trajectory',
                        curriculum_stratum=kind,source_logical_seed=family['seed'],source_engine_seed=seed,
                        source_hand_index=hand_index,source_step=step,seat=seat,
                        source_trace=canonical_trace([[dict(player=p,type=int(a.action_type),tile=a.tile,consumed=list(a.consume_tiles)) for p,a in batch.items()] for batch in trace]),
                        source_initial_current_hand_scores=list(next(e['scores'] for e in reversed(env.mjai_log) if e['type']=='start_kyoku')),
                        source_is_actual_teacher_action_not_terminal_optimality=True)
                    streams[family['split']].write(json.dumps(request,ensure_ascii=False)+'\n')
                    c=coverage[family['split']];c[kind]+=1;c['rows']+=1;c['seat_'+str(seat)]+=1
                    c['opponent_riichi_'+str(sum(p!=seat for p in request['state']['riichi']))]+=1
                    c['own_discard_number_'+str(discard_number)]+=1
                    if not reach:seen.add(slot)
                trace.append(selected);observations=env.step(selected)
            if not env.done():raise RuntimeError('Bounded teacher source did not finish')
            checked=ledger(env.mjai_log,env.scores())
            raw.write(json.dumps(dict(source=family,engine_seed=seed,mjai=env.mjai_log,terminal_scores=list(env.scores()),ledger_hands=checked['hands']))+'\n');raw.flush()
            stats['complete_sources']+=1
            for stream in streams.values():stream.flush()
            if stats['complete_sources']%16==0:print(json.dumps(dict(stage='basic_teacher_collection',sources=stats['complete_sources'],coverage={p:dict(c) for p,c in coverage.items()})),flush=True)
    except BaseException as error:
        write(out/'error-private.json',dict(type=type(error).__name__,error=str(error),stats=dict(stats)));raise
    finally:
        raw.close()
        for stream in streams.values():stream.close()
    if not all(sha(p)==h for p,h in binding.items()):raise ValueError('Prospectively bound source changed')
    summary=dict(status='basic_true_trajectory_teacher_sources_complete_not_fit_admitted',stats=dict(stats),
        coverage={p:dict(c) for p,c in coverage.items()},registered_families={p:sum(s['split']==p for s in families) for p in coverage},
        files={p:dict(path=str(path.relative_to(ROOT)),sha256=sha(path)) for p,path in files.items()},
        full_source_matches_sha256=sha(out/'source-matches-private.jsonl'),teacher_actor_counts=dict(actor.counts),
        source_settlement_ledgers_passed=True,raw_physical_trajectories_private=True,student_optimizer_updates=0,
        data_native_encoder_ancestral_audit_pending=True,student_fit_admitted=False,final_reserved_grid_unused=True)
    write(out/'summary-private.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();collect(a.config.resolve())
