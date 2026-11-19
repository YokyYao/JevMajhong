"""Public native contexts for retaining v8 ordinary decisions under new inputs."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

from riichienv import ActionType,GameRule,RiichiEnv
from jevmahjong.core.seeds import engine_seed
from jevmahjong.core.state import build_request,legal_candidates
from jevmahjong.evaluation.agents import guarded_indices
from desktop.jev_sidecar.unified_policy import serving_request as original_request
from .collect import digest,write
from .context import annotate_record,placements,serving_request,TRAILING_INPUT_VERSION
from .native_public import add_native_context
from .public_policy import original_policy_keys,action_keys

SCHEMA='native-ordinary-context-v1'
QUOTAS={'train':{3:32,4:32},'validation':{3:8,4:8}}


def replay_decisions(env,log,cap=30000):
    """Exact event replay, yielding current observations before each action."""
    observations=env.reset();current=env.mjai_log
    if current!=log[:len(current)]:raise ValueError('Initial source log differs')
    for step in range(cap):
        if env.done():
            if current!=log:raise ValueError('Completed source log differs')
            return
        yield step,observations
        pending=log[len(current):]
        while pending and pending[0]['type'] in ('dora','reach_accepted'):pending=pending[1:]
        if not pending:raise ValueError('Source decision events ended early')
        events=[]
        for event in pending:
            if events and (event['type']!='hora' or events[0]['type']!='hora'):break
            events.append(event)
            if event['type']!='hora':break
        selected={}
        for player,obs in observations.items():
            action=None
            for event in events:
                if event.get('actor')==player or (len(observations)==1 and event['type']=='ryukyoku'):
                    action=obs.select_action_from_mjai(json.dumps(event))
                    if action is not None:break
            if action is None:
                legal=obs.legal_actions()
                action=next((a for a in legal if a.action_type==ActionType.PASS),None)
                if action is None and len(legal)==1:action=legal[0]
            if action is None:raise ValueError('Could not reconstruct ordinary-context source action')
            selected[player]=action
        observations=env.step(selected);previous=len(current);current=env.mjai_log
        if current[previous:]!=log[previous:len(current)]:raise ValueError('Source event prefix changed')
    raise RuntimeError('Source replay exceeded registered cap')


def phase_for(kinds):
    if kinds & {'ankan','kakan','daiminkan'}:return 'kan'
    if kinds & {'chi','pon'}:return 'call'
    if 'reach' in kinds:return 'riichi'
    return 'discard'


def select_context_rows(rows):
    groups={s:{r['seed'] for r in rows if r['split']==s} for s in QUOTAS}
    if groups['train'] & groups['validation']:raise ValueError('Native context source split leakage')
    selected=[]
    for split,quota in QUOTAS.items():
        pool=[r for r in rows if r['split']==split];used=set();remaining=dict(quota)
        def feasible(excluded,needed):
            available={rank:{r['seed'] for r in pool if r['rank']==rank and r['seed'] not in excluded} for rank in (3,4)}
            return (all(len(available[r])>=needed[r] for r in (3,4))
                    and len(available[3]|available[4])>=sum(needed.values()))
        if not feasible(used,remaining):raise ValueError('Insufficient distinct sources for rank-balanced ordinary contexts')
        cells={}
        for row in pool:cells.setdefault(row['cell'],[]).append(row)
        for values in cells.values():values.sort(key=lambda r:r['selection_key'])
        while sum(remaining.values()):
            added=False
            for cell in sorted(cells):
                for index,row in enumerate(cells[cell]):
                    if row['seed'] in used or not remaining[row['rank']]:continue
                    needed=dict(remaining);needed[row['rank']]-=1
                    if not feasible(used|{row['seed']},needed):continue
                    selected.append(row);used.add(row['seed']);remaining=needed
                    cells[cell].pop(index);added=True;break
            if not added:raise ValueError('Public cell sampler could not meet registered quotas')
    return selected


def prepare_contexts(config):
    if (config.get('schema')!=SCHEMA or config.get('fit_registered') is not False
            or config.get('teacher_registered') is not False
            or config.get('input_version')!=TRAILING_INPUT_VERSION):
        raise ValueError('Only the registered public-context preparation stage is enabled')
    source_root=Path(config['source_root']);coverage=Path(config['coverage_out']);out=Path(config['out'])
    manifest=json.loads((source_root/'selected-positions-private.json').read_text())
    if len(manifest)!=80 or digest(manifest)!=config['source_manifest_sha256']:
        raise ValueError('Fixed80 source-group manifest differs')
    source_hashes={};candidates=[];checked=0;observed=Counter()
    for anchor in manifest:
        seed=anchor['seed'];split=anchor['split']
        expected='validation' if (seed-2026116900)%4==3 else 'train'
        if split!=expected:raise ValueError('Original native split differs')
        for profile in ('standard','dama'):
            path=coverage/profile/'sources'/f'{seed}.json'
            before=path.read_bytes();source_hashes[str(path)]=hashlib.sha256(before).hexdigest()
            source=json.loads(before)
            if source['seed']!=seed or source['profile']!=profile:raise ValueError('Cached native source identity differs')
            env=RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=engine_seed(seed,'spaced-v1'))
            best={};anchor_checked=False
            for step,observations in replay_decisions(env,source['mjai_log']):
                if profile==anchor['source_profile'] and step==anchor['step']:
                    obs=observations[anchor['seat']];raw,_,_=build_request(obs);raw['state']['hand_number']=int(obs.kyoku_index)+1
                    if digest(serving_request(raw))!=anchor['request_sha256']:raise ValueError('Known win prefix request differs')
                    anchor_checked=True
                if not (int(env.round_wind)>0 or int(env.kyoku_idx)==3):continue
                for seat,obs in observations.items():
                    data=obs.to_dict();rank=placements(data['scores'])[seat]
                    if rank not in (3,4):continue
                    actions=legal_candidates(obs)
                    if len(actions)<2:continue
                    kinds={json.loads(a.to_mjai())['type'] for a in actions}
                    if 'hora' in kinds:continue
                    phase=phase_for(kinds)
                    closed=all(int(m.meld_type)==3 for m in data['melds'][seat])
                    style=('closed_riichi' if data['riichi_declared'][seat] else 'closed_nonriichi') if closed else 'open'
                    stage='extension' if int(env.round_wind)>0 else 'east4'
                    key=digest([SCHEMA,seed,profile,step,seat,rank,phase])
                    # Track the best public hash per rank/phase before forming
                    # full feature requests, then admit only actual NN menus.
                    bucket=(rank,phase,style,stage)
                    if bucket in best and key>=best[bucket]['selection_key']:continue
                    raw,actions,features=build_request(obs);raw['state']['hand_number']=int(obs.kyoku_index)+1
                    teacher=original_request(raw)
                    student=annotate_record(add_native_context(raw,obs,actions),TRAILING_INPUT_VERSION)
                    keys=list(teacher['questions']['action']['criteria'])
                    native=sorted(set(guarded_indices(obs,actions,features))|{i for i,f in enumerate(features) if not f})
                    support=[keys[i] for i in native]
                    if support!=original_policy_keys(teacher) or support!=action_keys(student):
                        raise ValueError('Original/native/student ordinary action supports differ')
                    if len(support)<2:continue
                    state=student['state']
                    closed=all(m['type']==3 for m in state['melds'][seat])
                    style=('closed_riichi' if seat in state['riichi'] else 'closed_nonriichi') if closed else 'open'
                    stage='extension' if state['round_wind']>0 else 'east4'
                    row={'id':'native-context:'+key,'seed':seed,'step':step,'seat':seat,'rank':rank,
                        'split':split,'group_id':'endgame-native:'+str(seed),'source_profile':profile,
                        'phase':phase,'style':style,'stage':stage,'cell':f'{rank}:{phase}:{style}:{stage}',
                        'selection_key':key,'raw_request':copy.deepcopy(raw),'teacher_request':teacher,
                        'student_request':student,'effective_action_keys':support,
                        'teacher_request_sha256':digest(teacher),'student_request_sha256':digest(student)}
                    best[bucket]=row
            if env.mjai_log!=source['mjai_log'] or list(env.scores())!=source['final_scores']:
                raise ValueError('Full native source replay changed final ledger')
            if profile==anchor['source_profile'] and not anchor_checked:raise ValueError('Known source prefix was not visited')
            if path.read_bytes()!=before:raise ValueError('Read-only source changed during extraction')
            candidates.extend(best.values());checked+=1
            observed.update(f'{split}:{r["rank"]}:{r["phase"]}' for r in best.values())
            write(out/'status-private.json',{'status':'extracting_public_ordinary_contexts','source_replays_complete':checked,
                'source_replays_total':160,'candidate_cells':len(candidates),'observed':dict(observed),
                'model_queries':0,'optimizer_updates':0})
    selected=select_context_rows(candidates)
    if len(selected)!=80 or len({r['seed'] for r in selected})!=80:raise ValueError('Ordinary context source cap failed')
    path=out/'paired-contexts-private.json'
    if path.exists() and json.loads(path.read_text())!=selected:raise ValueError('Existing context manifest differs')
    write(path,selected)
    summary={'status':'native_ordinary_context_manifest_ready_no_teacher_queries','positions':80,
        'splits':{s:dict(Counter(str(r['rank']) for r in selected if r['split']==s)) for s in QUOTAS},
        'phases':dict(Counter(r['split']+':'+r['phase'] for r in selected)),
        'styles':dict(Counter(r['split']+':'+r['style'] for r in selected)),
        'stages':dict(Counter(r['split']+':'+r['stage'] for r in selected)),
        'positions_sha256':digest(selected),'source_hashes':source_hashes,'exact_source_replays':160,
        'teacher_kind':'frozen_original_v8_policy_retention_not_expert','model_queries':0,'optimizer_updates':0,
        'raw_future_outcomes_used_for_selection':False,'original_student_native_supports_equal':True}
    write(out/'summary-private.json',summary);write(out/'status-private.json',{k:v for k,v in summary.items() if k!='source_hashes'})
    return {k:v for k,v in summary.items() if k!='source_hashes'}
