"""Recompute the fixed East-match confirmation from raw records, without loading a model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_rows(rows, config):
    expected = {(seed, seat)
                for seed in range(config['confirmation_seed_start'],
                                  config['confirmation_seed_start'] + config['confirmation_seeds'])
                for seat in range(4)}
    keys = [(row['seed'], row['seat']) for row in rows]
    require(len(keys) == len(expected) and set(keys) == expected,
            'Confirmation must contain every fixed seed/seat exactly once')
    for row in rows:
        require(row['engine_seed'] == row['seed'] * (1 << 32)
                and row['seed_scheme'] == 'spaced-v1', 'Unexpected wall stream')
        require(row['game_mode'] == '4p-red-east'
                and row['reward_mode'] == 'net_points', 'Unexpected match or reward mode')
        require(row['reward_scale'] == config['reward_scale'], 'Unexpected reward scale')
        require(isinstance(row['score'], int) and row['rank'] in (1, 2, 3, 4),
                'Invalid terminal outcome')
        require(math.isclose(row['reward'], (row['score'] - 25000) / config['reward_scale'],
                             rel_tol=0, abs_tol=1e-10), 'Incorrect signed terminal reward')
        require(row['sampling_seed'] is None and row['trainable_decisions'] == 0
                and row['old_opponent'] is False, 'Confirmation is not frozen rule-opponent evaluation')
    return {(row['seed'], row['seat']): row for row in rows}


def summarize(rows):
    n = len(rows)
    return {'games': n, 'independent_seeds': len({r['seed'] for r in rows}),
            'firsts': sum(r['rank'] == 1 for r in rows),
            'first_rate': sum(r['rank'] == 1 for r in rows) / n,
            'top2s': sum(r['rank'] <= 2 for r in rows),
            'top2_rate': sum(r['rank'] <= 2 for r in rows) / n,
            'mean_rank': sum(r['rank'] for r in rows) / n,
            'fourth_rate': sum(r['rank'] == 4 for r in rows) / n,
            'mean_final_points': sum(r['score'] for r in rows) / n,
            'mean_net_points': sum(r['score'] - 25000 for r in rows) / n}


def compare(candidate, reference, config):
    from scipy.stats import t
    a, b = audit_rows(candidate, config), audit_rows(reference, config)
    seeds = sorted({seed for seed, _ in a})
    require(len(seeds) >= 2, 'At least two independent streams are required')
    # Four rotated seats share a stream; the confidence interval uses streams as units.
    delta = [sum(a[seed, seat]['score'] - b[seed, seat]['score'] for seat in range(4)) / 4
             for seed in seeds]
    gain = statistics.mean(delta)
    half = float(t.ppf(.975, len(delta) - 1)) * statistics.stdev(delta) / math.sqrt(len(delta))
    interval = [gain - half, gain + half]
    candidate_summary, reference_summary = summarize(candidate), summarize(reference)
    qualified = (candidate_summary['mean_net_points'] >= config['target_mean_net_points']
                 and gain >= config['target_net_points_gain'] and interval[0] > 0)
    return {'candidate': candidate_summary, 'reference': reference_summary,
            'primary_delta': gain, 'primary_delta_ci95': interval,
            'qualified': qualified,
            'changed_outcomes': sum((a[k]['score'], a[k]['rank']) != (b[k]['score'], b[k]['rank']) for k in a),
            'positive_point_deltas': sum(a[k]['score'] > b[k]['score'] for k in a),
            'negative_point_deltas': sum(a[k]['score'] < b[k]['score'] for k in a)}


def audit(run, expected_head_sha256):
    protocol = json.loads((run / 'protocol.json').read_text())
    selection = json.loads((run / 'confirmation-selection.json').read_text())
    config = protocol['config']
    require(selection['config'] == config, 'Selected protocol changed')
    require(hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            == protocol['sha256'], 'Protocol fingerprint changed')
    require(config['game_mode'] == '4p-red-east' and config['reward_mode'] == 'net_points'
            and config['seed_scheme'] == 'spaced-v1', 'Unsupported confirmation protocol')
    confirmation_seeds = set(range(config['confirmation_seed_start'],
                                   config['confirmation_seed_start'] + config['confirmation_seeds']))
    development_seeds = set(range(config['development_seed_start'],
                                  config['development_seed_start'] + config['development_seeds']))
    require(not confirmation_seeds & development_seeds, 'Development and confirmation streams overlap')
    training_files = sorted(run.glob('training-games-stage-*.json'))
    require(bool(training_files), 'Missing training provenance')
    training_seeds = {row['seed'] for path in training_files for row in json.loads(path.read_text())}
    require(not training_seeds & (confirmation_seeds | development_seeds),
            'Training and evaluation streams overlap')
    for name, expected in protocol['code_sha256'].items():
        require(sha256(ROOT / name) == expected, f'Numerical code changed: {name}')
    checkpoint = ROOT / selection['checkpoint']
    reference = ROOT / config['reference']
    require(sha256(checkpoint / 'head.pt') == expected_head_sha256, 'Selected candidate head changed')
    require(sha256(reference / 'head.pt') == protocol['reference_head_sha256'], 'Reference head changed')
    for directory in (checkpoint, reference):
        require(sha256(directory / 'adapter_model.safetensors') == protocol['reference_adapter_sha256'],
                'Frozen adapter changed')
    data = {name: [json.loads(line) for line in (run / f'confirmation-{name}' / 'games.jsonl').read_text().splitlines()]
            for name in ('reference', 'candidate')}
    computed = compare(data['candidate'], data['reference'], config)
    recorded = json.loads((run / 'confirmation-result.json').read_text())
    require(recorded['status'] == 'complete' and recorded['checkpoint'] == selection['checkpoint'],
            'Unexpected result or selected checkpoint')
    require(recorded['default_policy_changed'] is False, 'Default was promoted before audit')
    require(recorded['primary_metric'] == 'mean_net_points', 'Unexpected primary metric')
    for name in ('reference', 'candidate'):
        for metric, value in computed[name].items():
            require(math.isclose(recorded[name][metric], value, rel_tol=0, abs_tol=1e-8),
                    f'Recorded {name} {metric} differs from raw outcomes')
    require(len(recorded['primary_delta_ci95']) == 2, 'Malformed confidence interval')
    for actual, expected in zip(recorded['primary_delta_ci95'], computed['primary_delta_ci95']):
        require(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-8), 'Recorded confidence interval differs')
    require(recorded['qualified'] == computed['qualified'], 'Recorded qualification differs')
    require(math.isclose(recorded['paired']['primary_delta'], computed['primary_delta'], rel_tol=0, abs_tol=1e-8),
            'Recorded paired gain differs')
    return {'audit_status': 'passed', 'checkpoint': selection['checkpoint'],
            'candidate_head_sha256': expected_head_sha256,
            'config_sha256': protocol['sha256'],
            'raw_record_sha256': {name: sha256(run / f'confirmation-{name}' / 'games.jsonl') for name in data},
            'training_and_evaluation_streams_disjoint': True,
            'scope': 'Fixed four-player East simulator with three rule opponents; not Mahjong Soul ranked performance',
            **computed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--expected-head-sha256', required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = audit(args.run, args.expected_head_sha256)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.out:
        args.out.write_text(rendered)
    print(rendered, end='')


if __name__ == '__main__':
    main()
