"""Controlled frozen-model ablation of native versus screenshot call inputs.

No labels are inferred from action frequencies. Only player-visible requests
are saved; simulated hands/walls never enter the model's input or this corpus.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def collect(seeds, count):
    from riichienv import GameRule, RiichiEnv
    from arena import EfficiencyAgent
    from semantic import build_request
    from seed_control import engine_seed
    pools = {'chi': [], 'pon': []}
    baseline = EfficiencyAgent()
    for seed in range(7819000, 7819000 + seeds):
        env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=engine_seed(seed, 'spaced-v1'))
        observations = env.reset()
        step = 0
        while not env.done():
            selected = {}
            for player, obs in observations.items():
                actions = obs.legal_actions()
                kinds = {json.loads(a.to_mjai())['type'] for a in actions}
                if 'none' in kinds and kinds & {'chi', 'pon'} and 'hora' not in kinds:
                    payload, _, _ = build_request(obs)
                    serialized = json.dumps(payload, sort_keys=True)
                    digest = hashlib.sha256(serialized.encode()).hexdigest()
                    record = {'id': digest, 'seed': seed, 'step': step, 'request': payload,
                              'label': None, 'label_status': 'unlabelled_native_visible_state'}
                    for kind in kinds & {'chi', 'pon'}:
                        pools[kind].append(record)
                selected[player] = baseline.act(obs)
            observations = env.step(selected)
            step += 1
            if step > 100000:
                raise RuntimeError('Simulation did not terminate')
        print(json.dumps({'collected_seed': seed, 'pool': {k: len(v) for k, v in pools.items()}}), flush=True)
    chosen = {}
    for kind, rows in pools.items():
        for row in sorted(rows, key=lambda r: r['id'])[:count // 2]:
            chosen[row['id']] = row
    return list(chosen.values())


def variants(payload):
    out = {'native': copy.deepcopy(payload)}
    legacy = copy.deepcopy(payload)
    for action in legacy['questions']['action']['criteria'].values():
        if action['type'] in ('chi', 'pon'):
            action['consumed'] = sorted([*action['consumed'], action['pai']])
    # Diagnostic only: this is not a legal live MJAI consumed list.
    out['legacy_answer_leaking_consumed'] = legacy
    temporal = copy.deepcopy(payload)
    for name in ('tsumogiri_flags', 'riichi_sutehais', 'last_tedashis', 'last_discard'):
        temporal['state'][name] = None
    out['missing_temporal'] = temporal
    defaults = copy.deepcopy(temporal)
    defaults['state'].update(scores=[25000] * 4, dealer=0, round_wind=0, honba=0, sticks=0, riichi=[], hand_number=1)
    out['missing_temporal_default_metadata'] = defaults
    unknown = copy.deepcopy(temporal)
    for name in ('scores', 'dealer', 'round_wind', 'honba', 'sticks', 'hand_number', 'riichi'):
        unknown['state'][name] = None
    out['missing_temporal_unknown_metadata'] = unknown
    return out


def evaluate(records, directory, checkpoint, head_override=None):
    from desktop.jev_sidecar.engine import Adviser
    adviser = Adviser()
    print(adviser.load(checkpoint), flush=True)
    head_file = head_override or checkpoint / 'head.pt'
    identity = {'head_sha256': hashlib.sha256(head_file.read_bytes()).hexdigest(),
                'corpus_sha256': hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
                'variant_names': list(variants(records[0]['request']))}
    identity_path = directory / 'run-identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Changed weights/corpus/protocol: use a new output directory')
    identity_path.write_text(json.dumps(identity, indent=2))
    if head_override:
        import torch
        adviser.model.runtime.model.head.load_state_dict(torch.load(head_override, map_location='cpu', weights_only=True)['head'])
    rows = []
    output = directory / 'predictions.jsonl'
    completed = set()
    if output.exists():
        rows = [json.loads(line) for line in output.read_text(encoding='utf8').splitlines()]
        completed = {(r['id'], r['variant']) for r in rows}
    with output.open('a', encoding='utf8') as stream:
        for record in records:
            for kind, payload in variants(record['request']).items():
                if (record['id'], kind) in completed:
                    continue
                started = time.monotonic()
                response = adviser.model.system_one(**payload)
                criteria = payload['questions']['action']['criteria']
                probabilities = response['answers']['action']['probabilities']
                from jevany.api import validate_distribution
                validate_distribution(probabilities, list(criteria))
                selected = max(criteria, key=lambda key: probabilities[key])
                row = {'id': record['id'], 'variant': kind, 'seed': record['seed'],
                       'chosen': criteria[selected], 'probabilities': probabilities,
                       'none_probability': sum(probabilities[k] for k, v in criteria.items() if v['type'] == 'none'),
                       'seconds': time.monotonic() - started}
                rows.append(row)
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                print(json.dumps({k: row[k] for k in ('variant', 'chosen', 'none_probability', 'seconds')}), flush=True)
    summary = {}
    for variant in variants(records[0]['request']):
        group = [r for r in rows if r['variant'] == variant]
        summary[variant] = {'n': len(group), 'pass_choices': sum(r['chosen']['type'] == 'none' for r in group),
                            'mean_pass_preference': sum(r['none_probability'] for r in group) / len(group)}
    report = {'variants': summary, 'note': 'Unlabelled request ablation. Frequency does not measure strategic correctness or win rate.',
              'checkpoint': str(checkpoint), 'head_override': str(head_override) if head_override else None,
              'identity': identity, 'trained_weights': False}
    (directory / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--cases', type=int, default=24)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'models/calibration-02')
    parser.add_argument('--head-override', type=Path, help='Offline verified head comparison; never writes deployed weights')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    corpus = args.out / 'cases.json'
    if corpus.exists():
        records = json.loads(corpus.read_text(encoding='utf8'))
    else:
        records = collect(args.seeds, args.cases)
        corpus.write_text(json.dumps(records, indent=2) + '\n', encoding='utf8')
    if not records:
        raise RuntimeError('No call decisions collected')
    if args.evaluate:
        evaluate(records, args.out, args.checkpoint, args.head_override)


if __name__ == '__main__':
    main()
