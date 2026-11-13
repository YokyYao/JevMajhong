"""Revalue declared decision states with paired same-hand future-wall rollouts."""
import argparse
from collections import Counter, defaultdict
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import statistics
import torch
from riichienv import GameRule, RiichiEnv
from jevmahjong.core.io import digest, request_hash, write
from jevmahjong.core.seeds import engine_seed
from jevmahjong.core.state import build_request, legal_candidates
from jevmahjong.data.native_pairs import load_sources, select_decisions
from jevmahjong.data.native_states import decision_records,INPUT_METHODS
from jevmahjong.policy.routing import make_head
from .collector import NativeCollector, action_key, call_observation_digest, derived_seed
from .futures import replay_prefix, sample_future, finish_intervention_hand
from .provenance import snapshot, verify
from .utility import hand_utility

METHOD = 'paired_future_ev_v1'


def continuation_policy(config):
    """Keep the expanded defense admission identical in controls and futures."""
    if config.get('decision_filter') == 'open_tenpai_riichi':
        from .defense_states import DefenseStateCollector
        return DefenseStateCollector(config)
    return NativeCollector(config)


def case_key(row):
    return tuple(row['episode_key'])


def choose_cases(partitions, head, limit, canonical_state_only=False, stratify_call_openness=False,baseline_context=None):
    if not isinstance(stratify_call_openness,bool):raise ValueError('Boolean call-openness sampling flag required')
    buckets = defaultdict(list); seen = set()
    for split, rows in partitions.items():
        for original in sorted(rows, key=case_key) if canonical_state_only else rows:
            row = copy.deepcopy(original)
            current_head = head[row['scope']] if isinstance(head, dict) else head
            with torch.no_grad(): reference = current_head(row['decide'], row['options']).softmax(-1)
            from jevmahjong.policy.call_baseline import transform
            reference=transform(row['public_request'],reference,baseline_context)
            teacher = max(row['eligible'], key=lambda i: float(reference[i]))
            alternative = row['action'] if row['action'] != teacher else row['teacher_action']
            identity = (row['episode_key'][0], row['episode_key'][1], row['request_sha256'], teacher, alternative)
            if canonical_state_only: identity = identity[:3]
            if identity in seen: continue
            seen.add(identity)
            kinds = {v['type'] for v in row['public_request']['questions']['action']['criteria'].values()}
            kan = '+'.join(sorted(kinds & {'ankan', 'daiminkan', 'kakan'}))
            if row['scope'] == 'calling': kan = '+'.join(sorted(kinds & {'chi', 'pon'}))
            state = row['public_request']['state']; threat = bool(set(state['riichi']) - {state['seat']})
            row.update(reference=reference, teacher_action=teacher, action=alternative, source_split=split)
            row.pop('alternative_probability', None)
            row['pair_selection'] = ('original_legal_pair_against_registered_policy_v1' if baseline_context is not None
                else 'original_legal_pair_against_play_v1')
            bucket=(split,kan,threat)
            if stratify_call_openness:
                if row['scope']!='calling':raise ValueError('Call-openness sampling requires pure calling states')
                opened=any(m['type']!=3 for m in state['melds'][state['seat']])
                bucket+= (opened,)
            buckets[bucket].append(row)
    for rows in buckets.values(): rows.sort(key=case_key, reverse=True)
    selected = []
    while any(buckets.values()) and (limit is None or len(selected) < limit):
        for key in sorted(buckets):
            if buckets[key] and (limit is None or len(selected) < limit): selected.append(buckets[key].pop())
    return selected


def select_case_plan(partitions,head,config,baseline_context=None):
    limits=config.get('max_states_by_split')
    options=(config.get('canonical_state_only',False),config.get('stratify_call_openness',False))
    if config.get('call_context_baseline') and config.get('train_scopes')==['calling'] and baseline_context is None:
        raise ValueError('Pure-call selection requires the actual pinned calling context')
    reserve=config.get('call_route_reserve_by_split')
    if reserve is not None and (limits is None or not options[0]
            or config.get('train_scopes')!=['calling'] or config.get('decision_filter')!='all'
            or not isinstance(reserve,dict) or set(reserve)!={'train','held'}):
        raise ValueError('Call-route reserves require canonical pure-call states and separate split budgets')
    if limits is None:return choose_cases(partitions,head,config.get('max_states'),*options,baseline_context)
    if (not isinstance(limits,dict) or set(limits)!={'train','held'}
            or any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in limits.values())
            or config.get('max_states')!=sum(limits.values())):
        raise ValueError('Explicit positive train/held quotas must sum to the state budget')
    selected=[]
    for split in ('train','held'):
        if reserve is None:
            selected.extend(choose_cases({split:partitions[split]},head,limits[split],*options,baseline_context))
            continue
        requested=reserve[split];allowed={'closed_chi_tanyao_gain','closed_pon_yakuhai','open'}
        if (not isinstance(requested,dict) or set(requested)-allowed
                or any(isinstance(n,bool) or not isinstance(n,int) or n<1 for n in requested.values())
                or sum(requested.values())>limits[split]):
            raise ValueError('Positive public call-route reserves must fit each split budget')
        candidates=choose_cases({split:partitions[split]},head,None,*options,baseline_context)
        from .call_head import route_groups
        routes=route_groups([{'row':r} for r in candidates]);chosen={}
        for name,count in sorted(requested.items()):
            pool=routes.get(name,[])
            if len(pool)<count:raise ValueError('Insufficient public call-route coverage: '+split+'/'+name)
            for g in pool[:count]:chosen[case_key(g['row'])]=g['row']
        for row in candidates:
            if len(chosen)>=limits[split]:break
            chosen.setdefault(case_key(row),row)
        selected.extend(chosen.values())
    return selected


def collect_case(row, source, payload, policy, config):
    seed, seat, replica = case_key(row)
    source_cfg = source['config']
    game = next(g for g in decision_records(payload) if (g['seed'], g['seat'], g['replica']) == (seed, seat, replica))
    control = next(g for g in payload['controls'] if g['seat'] == seat)
    env = RiichiEnv(game_mode=source_cfg['game_mode'], rule=GameRule.default_mjsoul(),
        seed=engine_seed(seed, source_cfg['seed_scheme']))
    observations = replay_prefix(env, control['mjai_log'], game['intervention_step'])
    if call_observation_digest(observations) != game['intervention_state_digest']:
        raise ValueError('Replayed intervention observation differs')
    request, actions, _ = build_request(observations[seat])
    if request_hash(request) != row['request_sha256'] or [action_key(a) for a in actions] != row['action_keys']:
        raise ValueError('Replayed public request or legal actions differ')
    from jevany.api import validate_distribution
    answer = policy.router.system_one(**request)['answers']['action']['probabilities']
    p, _ = validate_distribution(answer, list(request['questions']['action']['criteria']))
    if (not torch.allclose(torch.tensor(p), row['reference'], atol=2e-5, rtol=2e-4)
            or max(row['eligible'], key=lambda i:p[i]) != row['teacher_action']):
        raise ValueError('Cached and actual deployed reference differ')
    first = {p:policy.opponent(p, o, seat, control['suite']) for p,o in observations.items() if p != seat}
    worlds = []
    for index in range(config['worlds']):
        future_seed = derived_seed(config['seed'], seed, seat, replica, f'future-{index}')
        fork = sample_future(env, future_seed, config.get('wall_scope','live_only'))
        wall_hash = hashlib.sha256(bytes(fork.wall)).hexdigest()
        arms = {}
        for name, chosen in (('control', row['teacher_action']), ('branch', row['action'])):
            current = fork.clone(); selected = dict(first); selected[seat] = actions[chosen]
            arms[name] = finish_intervention_hand(current, selected, policy, seat, control['suite'], row['hand_index'])
        worlds.append({'future_seed':future_seed, 'future_wall_sha256':wall_hash, **arms})
    credits = {name:[hand_utility(w[name]['mjai_log'], row['hand_index'], seat, row['draws_before'],
        config['fast_gamma'], config['reward']) for w in worlds] for name in ('control','branch')}
    means = {name:{k:statistics.mean(v[k] for v in values) for k in values[0]} for name,values in credits.items()}
    result = dict(row, counterfactual_worlds=worlds, prefix_events=len(env.mjai_log),
        source_state_digest=game['intervention_state_digest'],
        training_credit=means, training_target=METHOD,
        training_advantage=means['branch']['utility'] - means['control']['utility'],
        control_fast_reward=means['control']['fast_reward'], branch_fast_reward=means['branch']['fast_reward'],
        fast_advantage=means['branch']['fast_reward'] - means['control']['fast_reward'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();cfg=json.loads(args.config.read_text())
    if not isinstance(cfg['worlds'],int) or cfg['worlds'] < 2:raise ValueError('At least two paired futures required')
    if cfg.get('expand_from'):
        from .option_expand import collect
        collect(cfg,args.resume);return
    out=Path(cfg['out']);out.mkdir(parents=True,exist_ok=args.resume)
    lock=(out/'collector.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=snapshot(cfg);protocol['collection_method']=METHOD
    parts,hashes,sources=load_sources(cfg['data'],protocol['model_files'],cfg['reward'],allow_unlabeled=True,
        expected_call_baseline=cfg.get('call_context_baseline'))
    if any(json.loads((Path(s['path'])/'protocol.json').read_text()).get('collection_method') == METHOD for s in sources):
        raise ValueError('Revalue original native states, not an already revalued dataset')
    from jevmahjong.data.future_pairs import future_scope
    scope, mode = future_scope(cfg)
    if mode=='open_tenpai_riichi' or any(json.loads((Path(s['path'])/'protocol.json').read_text()).get('collection_method') in INPUT_METHODS for s in sources):
        audit_root=Path(cfg['required_native_audit']).resolve()
        audited=json.loads((audit_root/'result.json').read_text())
        audit_protocol=json.loads((audit_root/'protocol.json').read_text());verify(audit_protocol)
        if (audited['status']!='passed_complete' or len(sources)!=1
                or Path(audit_protocol['config']['collection']).resolve()!=Path(sources[0]['path'])
                or audited['walls']!=sources[0]['seeds'] or audited['states']!=sources[0]['rows']
                or audit_protocol['model_files']!=protocol['model_files']):
            raise ValueError('Complete matching native input audit required before future labels')
        for path,h in audit_protocol['source_files'].items():
            if digest(Path(path))!=h:raise ValueError('Audited native input changed')
            if path in hashes and hashes[path]!=h:raise ValueError('Native audit and selected inputs differ')
            hashes[path]=h
        for name in ('protocol.json','result.json'):hashes[str(audit_root/name)]=digest(audit_root/name)
    parts=select_decisions(parts,mode,[scope]);torch.set_num_threads(4)
    role = 'discard' if scope == 'ordinary' else 'call'
    meta=torch.load(Path(cfg[role+'_checkpoint'])/'head.pt',map_location='cpu',weights_only=True)
    head=make_head(meta,'cpu').eval().requires_grad_(False)
    from jevmahjong.policy.call_baseline import load as load_baseline
    calling_baseline=load_baseline(cfg,protocol['model_files'])
    cases=select_case_plan(parts,head,cfg,calling_baseline);by_wall=defaultdict(list)
    for row in cases:by_wall[row['episode_key'][0]].append(row)
    protocol['source_files']=hashes;protocol['selected_case_keys']=[list(case_key(r)) for r in cases]
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text()) != protocol:
        raise ValueError('Resume requires unchanged inputs, cases and source')
    write(out/'protocol.json',protocol)
    policy=continuation_policy(cfg);completed=0;counts=Counter()
    try:
        for source_index, source in enumerate(sources):
            source_proto=json.loads((Path(source['path'])/'protocol.json').read_text())
            directory=out/f'source_{source_index}';shards=directory/'walls-private';shards.mkdir(parents=True,exist_ok=True)
            subcfg=dict(cfg,training_seed_start=source['seed_start'],training_seeds_per_stage=source['seeds'],
                game_mode=source_proto['config']['game_mode'],seed_scheme=source_proto['config']['seed_scheme'],source_root=source['path'])
            subprotocol=dict(protocol,config=subcfg)
            if (directory/'protocol.json').exists() and json.loads((directory/'protocol.json').read_text()) != subprotocol:
                raise ValueError('Source-group protocol changed')
            write(directory/'protocol.json',subprotocol);nrows=0
            for seed in range(source['seed_start'],source['seed_start']+source['seeds']):
                destination=shards/f'{seed}.pt';receipt=destination.with_suffix('.audit.json')
                if destination.exists() or receipt.exists():
                    if not args.resume or not destination.exists() or not receipt.exists():raise ValueError('Inspect incomplete existing shard')
                    if digest(destination) != json.loads(receipt.read_text())['sha256']:raise ValueError('Completed future shard changed')
                    saved=torch.load(destination,map_location='cpu',weights_only=False)
                else:
                    original=Path(source['path'])/'walls-private'/f'{seed}.pt'
                    if digest(original) != hashes[str(original)]:raise ValueError('Original shard changed')
                    original_payload=torch.load(original,map_location='cpu',weights_only=False);rows=[]
                    for row in by_wall[seed]:
                        rows.append(collect_case(row,source_proto,original_payload,policy,cfg))
                        write(out/'status.json',{'status':'collecting_future_labels','states_complete':completed+len(rows),
                            'states_total':len(cases),'worlds_per_state':cfg['worlds'],'current_source':source_index})
                    saved={'seed':seed,'rows':rows,'source_path':str(original),'source_sha256':hashes[str(original)],
                        'collection_method':METHOD};temp=destination.with_suffix('.tmp');torch.save(saved,temp);temp.replace(destination)
                    write(receipt,{'sha256':digest(destination),'rows':len(rows)})
                for row in saved['rows']:counts[row['category']]+=1
                completed+=len(saved['rows']);nrows+=len(saved['rows'])
            write(directory/'collection-result.json',{'status':'collection_complete','method':METHOD,
                'walls':source['seeds'],'rows':nrows,'origin_wall_groups_not_fresh_matches':True})
        verify(protocol)
        for path,h in hashes.items():
            if digest(Path(path))!=h:raise ValueError('Original data changed during revaluation')
        result={'status':'collection_complete','method':METHOD,'states':completed,'worlds_per_state':cfg['worlds'],
            'native_hand_rollouts':completed*cfg['worlds']*2,'fresh_matches':0,'rows_by_original_category':dict(counts),
            'source_outputs':[str((out/f'source_{i}').resolve()) for i in range(len(sources))],
            'human_test_opened':False,'optimizer_updates':0}
        write(out/'collection-result.json',result);write(out/'status.json',result)
    finally:policy.close()


if __name__=='__main__':main()
