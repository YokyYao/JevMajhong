"""Replay and encode unlabelled endgame cases; report public coverage gaps."""
import argparse
import base64
from collections import Counter
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


def restore(row):
    """Recreate an exact input from native actions, not from a saved oracle."""
    from riichienv import RiichiEnv,GameRule
    from jevmahjong.core.state import legal_candidates,build_request
    from tools.curriculum_win_context import build_win_request
    meta,state=row['_meta'],row['state']
    env=RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),
        seed=meta['source_engine_seed']+meta['source_hand_index'])
    observations=env.reset(oya=state['dealer'],round_wind=state['round_wind'],honba=state['honba'],
        scores=meta['initial_scores'],kyotaku=meta['initial_sticks'])
    key=lambda a:(int(a.action_type),a.tile,tuple(sorted(a.consume_tiles)))
    for batch in meta['source_trace']:
        if {saved['player'] for saved in batch}!=set(observations):raise ValueError('Replay omitted an active player')
        actions={}
        for saved in batch:
            wanted=(saved['type'],saved['tile'],tuple(sorted(saved['consumed'])))
            menu={key(a):a for a in legal_candidates(observations[saved['player']])}
            if wanted not in menu:raise ValueError('Stored physical source action is no longer legal')
            actions[saved['player']]=menu[wanted]
        observations=env.step(actions)
        if env.done():raise ValueError('Stored trace settled before the decision')
    obs=env.get_obs_py([meta['seat']])[meta['seat']]
    rebuilt=(build_win_request(obs,env.mjai_log[-1]) if meta['category']=='win' else build_request(obs))[0]
    original={k:v for k,v in row.items() if k!='_meta'}
    if rebuilt!=original:raise ValueError('Stored input differs from full native reconstruction')
    if meta['source_control_key'] not in original['questions']['action']['criteria']:
        raise ValueError('Recorded source choice is absent')
    return env,obs


def public_solo_win(env,obs):
    """Narrow independent, no-ura transfer calculation; no future/outcome input.

    Existing scorer explicitly excludes kan and first-turn exceptional hands.
    This is a solo immediate transfer and never a terminal-placement target.
    """
    from riichienv import ActionType
    from jevmahjong.core.state import legal_candidates
    from jevmahjong.core.scoring import independent_win_score
    action=next(a for a in legal_candidates(obs) if a.action_type in (ActionType.RON,ActionType.TSUMO))
    seat=obs.player_id
    target=seat if action.action_type==ActionType.TSUMO else env.mjai_log[-1]['actor']
    # The scorer needs event order/actors only; private starts/draws are removed.
    history=[{k:v for k,v in e.items() if k in ('type','actor')} for e in env.mjai_log]
    snapshot=dict(observation=json.loads(base64.b64decode(obs.serialize_to_base64())),win_tile=action.tile)
    if any(hand for p,hand in enumerate(snapshot['observation']['hands']) if p!=seat):
        raise ValueError('An opponent hand entered the public score check')
    event=dict(actor=seat,target=target,deltas=[0]*4,ura_markers=[])
    result=independent_win_score(history,event,snapshot)
    if 'expected' not in result:return dict(supported=False,reason=result['reason'])
    scores=[s+d for s,d in zip(env.scores(),result['expected'])]
    rank=sorted(range(4),key=lambda p:(-scores[p],p)).index(seat)+1
    return dict(supported=True,own_income=result['expected'][seat],immediate_rank=rank,
        no_unknown_ura=True,sole_winner_assumption=True,terminal_rank_target=False,
        han=result['han'],fu=result['fu'],yaku=result['yaku'],open=result['open'])


def audit(data_root,out):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS')
    assert not socket.gethostname().startswith(('lo-','login'))
    out.mkdir(parents=True,exist_ok=False)
    read=lambda path:json.loads(path.read_text())
    ready=read(data_root/'data-ready-private.json')
    registration=read(data_root/'source-registration-private.json')
    config=registration['config'];reg=read(ROOT/config['source_registration'])
    sys.path.insert(0,reg['source']);sys.path.insert(0,reg['training_config']['upstream_source'])
    from jevmahjong.training.base_lora import package_digest
    from jevany.api import SystemOneRequest,to_record
    from jevany.data import api_request
    from jevany.model import DecisionModel,load_preprocessor
    import torch
    torch.set_num_threads(2)
    if package_digest(Path(reg['source'])/'jevmahjong')!=reg['source_sha256']:raise ValueError('Pinned source changed')
    if package_digest(Path(reg['training_config']['upstream_source'])/'jevany')!=reg['training_config']['upstream_source_sha256']:
        raise ValueError('Pinned encoder source changed')
    for path,digest in registration['files'].items():
        if sha(path)!=digest:raise ValueError('Registered collection dependency changed')
    bindings=dict(registration['files'])
    bindings[str(Path(__file__).resolve())]=sha(__file__)
    recipe=tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    tokenizer=load_preprocessor(reg['training_config']['base_load_path'],backbone_adapter='text')
    proxy=SimpleNamespace(decision_mode='pointer',option_isolation=False)
    registered={row['seed']:row for row in registration['sources']}
    counts={p:Counter() for p in ('train','validation')};groups={p:set() for p in counts}
    base_seen=set();base_counts=Counter();ids=set();lengths={p:[] for p in counts}
    case_details=[]
    for split in counts:
        file=ROOT/ready['files'][split]['path']
        if sha(file)!=ready['files'][split]['sha256']:raise ValueError('Collected inputs changed')
        for line in file.open():
            row=json.loads(line);meta=row['_meta'];state=row['state'];q=row['questions']['action']
            source=registered[meta['source_logical_seed']]
            if source['split']!=split or meta['split']!=split or meta['group_id']!=source['group_id']:
                raise ValueError('A whole-family split was changed')
            if meta['id'] in ids or any(k in q for k in ('label','target')):raise ValueError('Duplicate or already labelled case')
            ids.add(meta['id']);groups[split].add(meta['group_id'])
            env,obs=restore(row)
            rec,mapping=to_record(SystemOneRequest.model_validate(api_request(row)))
            enc=DecisionModel.encode(proxy,tokenizer,rec,max_state=recipe['max_state'],max_branch=recipe['max_branch'],strict=True)
            if enc['state_truncated'] or len(enc['ids'])>recipe['max_packed']:raise ValueError('Unlabelled public input exceeds tokenizer limits')
            if mapping[0]['keys']!=list(q['criteria']):raise ValueError('Actual encoder reordered legal actions')
            lengths[split].append((len(enc['ids']),enc['seg'].count(0)))
            c=counts[split];c['rows']+=1;c['category_'+meta['category']]+=1
            c['rank_'+str(meta['placement']['rank'])]+=1
            base=(meta['source_logical_seed'],meta['source_hand_index'],meta['source_step'],meta['seat'],meta['category'])
            if base not in base_seen:
                base_seen.add(base);base_counts['distinct_decisions']+=1
                base_counts['opponent_riichi_'+str(sum(p!=meta['seat'] for p in state['riichi']))]+=1
                for value in {v['type'] for v in q['criteria'].values()}:base_counts['offered_'+value]+=1
            if meta['category']=='win':
                own=meta['seat'];c['win_dealer' if own==state['dealer'] else 'win_nondealer']+=1
                c['win_open' if any(m.opened for m in obs.to_dict()['melds'][own]) else 'win_closed']+=1
                score=public_solo_win(env,obs)
                if score['supported']:
                    c['public_solo_score_checked']+=1;c['immediate_top2_after_solo_win' if score['immediate_rank']<=2 else 'still_outside_top2_after_solo_win']+=1
                    if score['own_income']<=3000:c['public_solo_income_le3000']+=1
                    for yaku in score['yaku']:c['win_yaku_'+yaku]+=1
                else:c['public_solo_score_skipped_'+score['reason']]+=1
                case_details.append(dict(id=meta['id'],group_id=meta['group_id'],split=split,public_solo_score=score))
    if groups['train']&groups['validation']:raise ValueError('Source families crossed splits')
    if len(ids)!=ready['counts']['rows']:raise ValueError('Collection row count differs')
    if not all(sha(path)==digest for path,digest in bindings.items()):raise ValueError('Audit inputs changed')
    with (out/'public-score-details-private.jsonl').open('x') as stream:
        for row in case_details:stream.write(json.dumps(row)+'\n')
    result=dict(status='unlabelled_native_replay_token_and_public_coverage_audited',
        node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'],data_ready_sha256=sha(data_root/'data-ready-private.json'),
        files=bindings,rows=len(ids),distinct_decisions=len(base_seen),distinct_decision_coverage=dict(base_counts),
        counts={p:dict(c) for p,c in counts.items()},actual_groups={p:len(v) for p,v in groups.items()},
        token_limits={k:recipe[k] for k in ('max_state','max_branch','max_packed')},
        tokens={p:dict(max_packed=max(v[0] for v in values),max_state=max(v[1] for v in values)) for p,values in lengths.items()},
        exact_native_input_replays=len(ids),all_score_views_grouped=True,source_no_legal_win_declines=True,
        own_input_no_opponent_hands=True,actual_encoder_without_backbone_loaded=True,
        encoder_default_label_index_unused_for_training=True,labels_generated=0,optimizer_updates=0,
        complete_ancestral_source_audit_pending=True,teacher_selected=False,stage4_fit_admitted=False,
        current_nonwin_kan_world_adapters_pending=True,raw_details_private=True)
    with (out/'summary-private.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit(a.data_root.resolve(),a.out.resolve())
