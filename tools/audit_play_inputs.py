"""Probe a frozen calling policy across native/screen views; never create expert labels.

prepare writes private replay inputs; probe requires one GPU and writes private traces
plus an aggregate summary. Historical screenshots lack complete inference inputs, so
their replays are explicitly reconstructions, not exact reproduction or strength tests.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    from desktop.jev_sidecar.engine import Board, request, reaction_candidates
    from desktop.jev_sidecar.call_policy import prepare_request
    from jevmahjong.data.policy_data import policy_request
    from jevmahjong.policy.call_features import enrich_request
    from jevany.data import api_request
    source = Path(args.samples)
    records = [json.loads(s) for s in source.read_text().splitlines()]
    samples = {r['id']: r for r in records if r['event'] == 'sample'}
    predictions = {r['id']: r for r in records if r['event'] == 'prediction'}
    verified = json.loads(Path(args.contexts).read_text()) if args.contexts else {}
    output = []
    for ident, prediction in predictions.items():
        result = prediction['result']
        if not result.get('reaction') or not any(
                v['action']['type'] in ('chi', 'pon') for v in result['all_candidates']): continue
        labels = samples[ident]['labels']; screen = labels['screen']
        board = Board(labels['hand'], rivers=[labels.get('rivers', {}).get(f'river{i}') for i in range(4)],
            melds=[screen.get('melds') or [], [], [], []], reaction=screen['reaction'],
            honba=screen.get('counters', {}).get('honba', 0), sticks=screen.get('counters', {}).get('sticks', 0))
        values = reaction_candidates(board)
        observed = [v['action'] for v in result['all_candidates']]
        if [v['action'] for v in values] != observed:
            raise ValueError(f'Candidate reconstruction differs: {ident}')
        for view in ('screen_unknown', 'verified_context'):
            if view == 'verified_context':
                if ident not in verified: continue
                for key, value in verified[ident]['board'].items(): setattr(board, key, value)
            payload = request(board, [])
            payload['state']['screen_observation'].update(reaction=board.reaction,
                confirmed_metadata=['honba', 'sticks', *board.confirmed_metadata])
            payload['questions']['action']['criteria'] = {f'a{i}': v['action'] for i, v in enumerate(values)}
            _, effective = prepare_request(payload)
            output.append(dict(id=ident, group='live_reconstruction', view=view, payload=effective,
                exact_replay=False, expert_label=False,
                observed_preferences={f'a{i}': v['preference'] for i, v in enumerate(result['all_candidates'])}))
    if args.native_source:
        import torch
        seen = set(); count = 0; buckets = {True: 0, False: 0}
        for path in sorted(Path(args.native_source).glob('walls-private/*.pt')):
            for row in torch.load(path, map_location='cpu', weights_only=False)['rows']:
                if row['source_split'] != 'train' or row['request_sha256'] in seen: continue
                criteria = row['public_request']['questions']['action']['criteria']
                cached_choice = list(criteria.values())[int(row['reference'].argmax())]['type'] != 'none'
                quota = (args.controls + int(cached_choice)) // 2
                if buckets[cached_choice] >= quota: continue
                buckets[cached_choice] += 1
                seen.add(row['request_sha256']); count += 1
                native = copy.deepcopy(row['public_request'])
                for view in ('native', 'screen_confirmed', 'screen_unknown'):
                    transformed = policy_request(native)
                    if view != 'native':
                        for field in ('tsumogiri_flags', 'riichi_sutehais', 'last_tedashis', 'last_discard'):
                            transformed['state'][field] = None
                        transformed['state']['screen_observation'] = dict(temporal_information='unknown',
                            metadata_confirmed=view == 'screen_confirmed', confirmed_metadata=['honba', 'sticks'])
                        public_prefix = row['counterfactual_worlds'][0]['control']['mjai_log'][:row['prefix_events']]
                        event = next(e for e in reversed(public_prefix) if e['type'] == 'dahai')
                        from desktop.jev_sidecar.tiles import HONORS
                        action = next(v for v in native['questions']['action']['criteria'].values() if v['type'] in ('chi', 'pon'))
                        tile = HONORS.get(action['pai'], '0' + action['pai'][1] if action['pai'].endswith('r') else action['pai'])
                        transformed['state']['screen_observation']['reaction'] = dict(tile=tile,
                            seat=(event['actor']-native['state']['seat']) % 4,
                            offered=sorted({v['type'] for v in native['questions']['action']['criteria'].values()}-{'none'}))
                        if view == 'screen_unknown':
                            for field in ('scores','dealer','round_wind','hand_number','riichi'): transformed['state'][field] = None
                    payload = api_request(enrich_request(transformed))
                    output.append(dict(id=row['request_sha256'], group='native_training_control', view=view,
                        payload=payload, source_sha256=sha(path), expert_label=False,
                        cached_preferences=dict(zip(native['questions']['action']['criteria'], row['reference'].tolist()))))
                if count >= args.controls: break
            if count >= args.controls: break
    target = Path(args.out); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(''.join(json.dumps(r) + '\n' for r in output))
    print(json.dumps(dict(inputs=len(output), live_cases=sum(r['view']=='screen_unknown' and r['group']=='live_reconstruction' for r in output),
        native_controls=sum(r['view']=='native' for r in output), sha256=sha(target))))


def probe(args):
    from jevmahjong.policy.routing import load_model
    from jevany.api import validate_distribution
    from desktop.jev_sidecar.call_policy import validate_call_checkpoint
    import torch
    identity = validate_call_checkpoint(args.checkpoint)
    rows = [json.loads(s) for s in Path(args.inputs).read_text().splitlines()]
    model = load_model(args.checkpoint)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    results = []; summary = dict(model=identity, inputs_sha256=sha(args.inputs),
        inference_only=True, expert_labels_created=0, not_strength_evaluation=True, groups={})
    with (out / 'predictions-private.jsonl').open('w') as stream, torch.inference_mode():
        for row in rows:
            output = model.system_one(**row['payload'])
            keys = list(row['payload']['questions']['action']['criteria'])
            values, _ = validate_distribution(output['answers']['action']['probabilities'], keys)
            probs = dict(zip(keys, map(float, values))); winner = max(probs, key=probs.get)
            record = dict(id=row['id'], group=row['group'], view=row['view'], probabilities=probs,
                selected=row['payload']['questions']['action']['criteria'][winner]['type'],
                model_request=row['payload'], exact_replay=row.get('exact_replay', True))
            baseline = row.get('cached_preferences', row.get('observed_preferences'))
            record['preference_max_difference'] = max(abs(probs[k]-baseline[k]) for k in keys)
            record['selection_changed_from_baseline'] = winner != max(baseline, key=baseline.get)
            results.append(record); stream.write(json.dumps(record) + '\n'); stream.flush()
            print(json.dumps({k:record[k] for k in ('group','view','selected','preference_max_difference')}), flush=True)
    for group in sorted({r['group'] for r in results}):
        summary['groups'][group] = {}
        for view in sorted({r['view'] for r in results if r['group']==group}):
            subset = [r for r in results if r['group']==group and r['view']==view]
            summary['groups'][group][view] = dict(cases=len(subset), calls=sum(r['selected'] in ('chi','pon') for r in subset),
                changes_from_baseline=sum(r['selection_changed_from_baseline'] for r in subset),
                max_preference_difference=max(r['preference_max_difference'] for r in subset))
    native = [r for r in results if r['view']=='native']
    summary['native_cached_parity_passed'] = bool(native) and all(r['preference_max_difference'] < .005 for r in native)
    summary['predictions_sha256'] = sha(out / 'predictions-private.jsonl')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


def inventory(args):
    """Describe actual strategy support; sampled return signs are not expert labels."""
    import torch
    from jevmahjong.policy.call_features import option_features
    counters = defaultdict(Counter); states = defaultdict(set); sources = {}; cache = {}
    for directory in args.data:
        root = Path(directory); marker = json.loads((root/'collection-result.json').read_text())
        if marker['status'] != 'collection_complete': raise ValueError('Complete audited sources required')
        for path in sorted(root.glob('walls-private/*.pt')):
            checksum = sha(path)
            if checksum != json.loads(path.with_suffix('.audit.json').read_text())['sha256']:
                raise ValueError('Source shard changed')
            sources[str(path)] = checksum
            for row in torch.load(path, map_location='cpu', weights_only=False)['rows']:
                req = row['public_request']; keys = list(req['questions']['action']['criteria'])
                if row['scope'] != 'calling': continue
                ident = row['request_sha256']
                if ident not in cache: cache[ident] = option_features(req)
                fs = cache[ident]; criteria = req['questions']['action']['criteria']
                seat = req['state']['seat']; closed = not any(m['type'] != 3 for m in req['state']['melds'][seat])
                threat = bool(set(req['state']['riichi'] or []) - {seat})
                selected = keys[int(row['reference'].argmax())]
                branch, teacher = keys[row['action']], keys[row['teacher_action']]
                for key in keys:
                    kind = criteria[key]['type']
                    if kind not in ('chi','pon'): continue
                    f = fs[key]
                    yaku = 'secured_yakuhai' if (f['secured_dragon_han'] or f['secured_wind_han']) else (
                        'tanyao_material' if f['all_simples_material_after'] else 'other_material')
                    stratum = '/'.join([row['source_split'], 'closed' if closed else 'open', kind,
                        'threat' if threat else 'no_threat', yaku, 'accelerates' if f['shanten_gain'] > 0 else 'no_shanten_gain'])
                    state_key = (row['episode_key'][0], row['episode_key'][1], ident, key)
                    if state_key not in states[stratum]:
                        states[stratum].add(state_key); counters[stratum]['options'] += 1
                        counters[stratum]['reference_selected_this_call'] += selected == key
                    # Only call-vs-pass pairs identify whether this call helped.
                    if {branch, teacher} != {key, next((k for k in keys if criteria[k]['type']=='none'), None)}: continue
                    delta = row['training_advantage'] * (1 if branch == key else -1)
                    counters[stratum]['sampled_call_better' if delta > 1e-12 else 'sampled_pass_better' if delta < -1e-12 else 'sampled_tie'] += 1
                    credit = row['training_credit']; net = credit['branch']['net_points'] - credit['control']['net_points']
                    counters[stratum]['sum_sampled_call_point_advantage'] += net * (1 if branch == key else -1)
        print(json.dumps({'source': str(root), 'shards_read': len(sources)}), flush=True)
    result = dict(strata={key:dict(value) for key,value in sorted(counters.items())},
        sampled_future_return_signs_not_expert_labels=True, training_and_held_separate=True,
        source_shards=len(sources), source_sha256=sources)
    Path(args.out).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'strata':len(counters),'source_shards':len(sources)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--samples', required=True); p.add_argument('--contexts')
    p.add_argument('--native-source'); p.add_argument('--controls', type=int, default=12); p.add_argument('--out', required=True)
    p = sub.add_parser('probe'); p.add_argument('--inputs', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p = sub.add_parser('inventory'); p.add_argument('--data', action='append', required=True); p.add_argument('--out', required=True)
    args = parser.parse_args(); {'prepare':prepare, 'probe':probe, 'inventory':inventory}[args.command](args)
