"""Prepare fresh whole-match-isolated full-policy expert supervision privately."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import zstandard
from riichienv import MjaiReplay
from audit_shard import split_for_match
from call_policy import digest, request_hash
from clean_policy_data import pass_is_identified, pass_outcomes, phase, policy_request
from head_rl import write
from replay_call_repair import build_replay_request, canonical_signature, UnsafeReplayCall, REPAIR_VERSION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    out, run = ROOT/cfg['data'], ROOT/cfg['out']
    if out.exists() or run.exists():
        raise ValueError('Refuse overwriting any previous data or experiment')
    audit = json.loads((ROOT/cfg['candidate_audit']/'summary.json').read_text())
    if not audit['passed_matched_candidate_sets'] or not audit['required_phase_coverage_met'] or audit['totals'].get('unmatched',0):
        raise ValueError('Fresh full-policy candidate audit must pass')
    if audit['repair_sha256'] != digest(ROOT/'replay_call_repair.py'):
        raise ValueError('Repair changed after audit')
    excluded = set()
    for name in cfg['exclude_provenance']:
        source = json.loads((ROOT/name).read_text())
        excluded.update(m for ms in source['matches'].values() for m in ms)
    out.mkdir(parents=True); run.mkdir(parents=True)
    targets = cfg['raw_phase_counts']
    counts = {s:Counter() for s in targets}
    matches = {s:set() for s in targets}
    actions = {s:Counter() for s in targets}
    exclusions, repairs, failures = Counter(), Counter(), []
    seen = set()
    streams = {s:(out/(s+'.jsonl')).open('w') for s in targets}
    raw_streams = {s:(out/(s+'-raw.jsonl')).open('w') for s in ('train','validation')}
    views_count = {s:Counter() for s in targets}
    try:
        with tarfile.open(ROOT/cfg['archive'],'r|*') as archive, tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'match.jsonl'
            for member in archive:
                if not member.isfile() or not member.name.endswith('.zst'):
                    continue
                identity = Path(member.name).name.removesuffix('.mjai.json.zst')
                if identity in excluded:
                    continue
                split = split_for_match(identity)
                if all(counts[split][k]>=v for k,v in targets[split].items()):
                    continue
                converted = []
                try:
                    if member.size>10_000_000:
                        raise ValueError('Oversized compressed match')
                    with archive.extractfile(member) as compressed, zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
                        raw = reader.read(10_000_001)
                    if len(raw)>10_000_000:
                        raise ValueError('Oversized match')
                    events = [json.loads(l) for l in raw.splitlines() if l.strip()]
                    start = next(e for e in events if e['type']=='start_kyoku')
                    if len(start['tehais'])!=4:
                        continue
                    outcomes = pass_outcomes(events)
                    path.write_bytes(raw)
                    replay = MjaiReplay.from_jsonl(str(path),rule='mjsoul')
                    pools = defaultdict(list)
                    for hand in replay.take_kyokus():
                        for pid, obs, expert in hand.steps(seat=None,skip_single_action=True):
                            kinds = {json.loads(x.to_mjai())['type'] for x in obs.legal_actions()}
                            if kinds & {'hora','ankan','kakan','daiminkan','ryukyoku'}:
                                exclusions['win_kan_abort_outside_learned_scope'] += 1
                                continue
                            if json.loads(expert.to_mjai())['type']=='none' and not pass_is_identified(obs,outcomes):
                                exclusions['censored_or_unresolved_pass'] += 1
                                continue
                            if kinds <= {'none','chi','pon'} and kinds & {'chi','pon'}:
                                category = 'reaction'
                            elif kinds <= {'dahai','reach'} and 'dahai' in kinds:
                                category = 'open_discard' if obs.to_dict()['melds'][pid] else 'closed_discard'
                            else:
                                exclusions['unsupported_phase'] += 1
                                continue
                            pools[category].append((obs,expert))
                    rng = random.Random(int(hashlib.sha256((str(cfg['seed'])+identity).encode()).hexdigest()[:16],16))
                    for category, pool in pools.items():
                        rng.shuffle(pool)
                        allowance = min(cfg['per_match_phase_cap'][category],targets[split][category]-counts[split][category])
                        # Select states before looking at the expert label; failures are explicit.
                        for obs, expert in pool[:max(0,allowance)]:
                            try:
                                request, signatures, audit_row = build_replay_request(obs)
                            except UnsafeReplayCall as exc:
                                exclusions[str(exc)] += 1
                                continue
                            if len(signatures)<2:
                                exclusions['repaired_single_option'] += 1
                                continue
                            gold = canonical_signature(expert)
                            if gold not in signatures:
                                raise ValueError('Expert label absent after candidate repair')
                            if phase(request)!=category:
                                raise ValueError('Phase changed after repair')
                            request['questions']['action']['label'] = f'a{signatures.index(gold)}'
                            request['_meta'] = {'id':f'{identity}/{len(converted)}','group_id':identity,
                                'source':'majsoul-throne-2026','license':'CC-BY-4.0','split':split,
                                'phase':category,'schema':'clean-full-policy-v1','call_repair_version':REPAIR_VERSION}
                            if category=='reaction':
                                d=obs.to_dict(); owners=[i for i,r in enumerate(d['discards']) if r and r[-1]==obs.last_discard]
                                if len(owners)!=1 or owners[0]==obs.player_id:
                                    raise ValueError('Unresolved reaction source')
                                request['_meta']['reaction_relative_seat']=(owners[0]-obs.player_id)%4
                            request = policy_request(request)
                            h=request_hash(request)
                            if h in seen:
                                exclusions['duplicate_state'] += 1
                                continue
                            request['_meta']['text_sha256']=h
                            converted.append((request,category,audit_row,h))
                except Exception as exc:
                    failures.append({'match':identity,'error':str(exc)[:300]})
                    continue
                for request, category, audit_row, h in converted:
                    if h in seen:
                        continue
                    seen.add(h);counts[split][category]+=1;matches[split].add(identity)
                    gold=request['questions']['action']['criteria'][request['questions']['action']['label']]['type']
                    actions[split][gold]+=1;repairs.update(audit_row)
                    if split in raw_streams:
                        raw_streams[split].write(json.dumps(request,ensure_ascii=False)+'\n')
                    chosen_views = (cfg['views'] if split=='validation' else
                        [cfg['views'][int(hashlib.sha256(request['_meta']['id'].encode()).hexdigest()[:16],16)%len(cfg['views'])]]
                        if split=='train' else ['native'])
                    for view in chosen_views:
                        result=policy_request(request,view)
                        result['_meta'].update(id=request['_meta']['id']+':'+view,parent_id=request['_meta']['id'],view=view)
                        result['_meta']['text_sha256']=request_hash(result)
                        streams[split].write(json.dumps(result,ensure_ascii=False)+'\n');views_count[split][view]+=1
                progress={'status':'preparing_fresh_full_policy','counts':{s:dict(c) for s,c in counts.items()},
                    'matches':{s:len(v) for s,v in matches.items()},'failed_matches':len(failures)}
                write(run/'status.json',progress)
                if sum(sum(c.values()) for c in counts.values())%1024<64:
                    print(json.dumps(progress),flush=True)
                if all(all(counts[s][k]>=v for k,v in target.items()) for s,target in targets.items()):
                    break
    finally:
        for f in list(streams.values())+list(raw_streams.values()):
            f.close()
    complete = all(dict(counts[s])==target for s,target in targets.items())
    if any(matches[s]&matches[t] for s in matches for t in matches if s!=t) or any(set.union(*matches.values())&excluded):
        raise ValueError('Whole-match overlap')
    report={'complete':complete,'config':cfg,'raw_phase_counts':{s:dict(c) for s,c in counts.items()},
        'raw_records':{s:sum(c.values()) for s,c in counts.items()},'records':{s:sum(c.values()) for s,c in views_count.items()},
        'views':{s:dict(c) for s,c in views_count.items()},'matches':{s:sorted(v) for s,v in matches.items()},
        'action_types':{s:dict(c) for s,c in actions.items()},'exclusions':dict(exclusions),'repairs':dict(repairs),
        'failures':failures,'excluded_prior_matches':len(excluded),'archive_sha256':digest(ROOT/cfg['archive']),
        'files':{s:digest(out/(s+'.jsonl')) for s in targets},'repair_sha256':digest(ROOT/'replay_call_repair.py'),
        'semantic_sha256':digest(ROOT/'semantic.py'),'policy_data_sha256':digest(ROOT/'clean_policy_data.py'),
        'source':'https://huggingface.co/datasets/nikketryhard/tenhou-houou-mjai',
        'source_revision':'6450bb584cef43ede3b514df48e301a832c6818b',
        'attribution':['NikkeTryHard/tenhou-to-mjai','Mahjong Soul'],'license':'CC-BY-4.0',
        'model_test_opened':False,'selection':'whole-match hash split; phase quotas then shuffled states, no expert action-label quota'}
    write(out/'provenance.json',report);write(run/'preparation.json',report)
    write(run/'status.json',{'status':'data_prepared' if complete else 'data_incomplete',
        'raw_records':report['raw_records'],'records':report['records']})
    print(json.dumps({k:report[k] for k in ('complete','raw_records','records','action_types','excluded_prior_matches')}),flush=True)
    if not complete:
        raise RuntimeError('Archive exhausted before phase quotas; do not train partial data')


if __name__=='__main__':
    main()
