"""Reconcile completed call-policy suites against raw games and frozen identities."""
import argparse
from itertools import product
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from call_policy import digest


def equal(actual, expected, name):
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-7):
        raise ValueError(f'Summary mismatch: {name}')


def audit_suite(rows, summary, cfg, suite):
    from scipy.stats import t
    policies = ('reference', 'candidate', 'previous') if cfg.get('previous_call_checkpoint') else ('reference', 'candidate')
    seeds = range(cfg['development_seed_start'], cfg['development_seed_start'] + cfg['development_seeds'])
    selected = [r for r in rows if r['suite'] == suite]
    lookup = {(r['seed'], r['seat'], r['policy']): r for r in selected}
    if len(lookup) != len(selected) or set(lookup) != set(product(seeds, range(4), policies)):
        raise ValueError('Missing, duplicate or unexpected paired games')
    if len(seeds) < 2: raise ValueError('Insufficient independent wall clusters')
    for r in selected:
        if (not isinstance(r['score'], int) or r['rank'] not in (1, 2, 3, 4)
            or r['hands'] <= 0 or r['hand_wins'] < 0 or r['dealins'] < 0):
            raise ValueError('Invalid game outcome')
    if set(summary['policies']) != set(policies): raise ValueError('Policy summary mismatch')
    for policy in policies:
        games = [r for r in selected if r['policy'] == policy]; n = len(games)
        hands = sum(r['hands'] for r in games)
        expected = {'games': n, 'independent_seeds': len(seeds),
            'firsts': sum(r['rank'] == 1 for r in games), 'top2s': sum(r['rank'] <= 2 for r in games),
            'first_rate': sum(r['rank'] == 1 for r in games) / n,
            'top2_rate': sum(r['rank'] <= 2 for r in games) / n,
            'fourth_rate': sum(r['rank'] == 4 for r in games) / n,
            'mean_rank': sum(r['rank'] for r in games) / n,
            'mean_final_points': sum(r['score'] for r in games) / n,
            'mean_net_points': sum(r['score'] - 25000 for r in games) / n,
            'hand_win_rate': sum(r['hand_wins'] for r in games) / hands,
            'dealin_rate': sum(r['dealins'] for r in games) / hands}
        for k, value in expected.items(): equal(summary['policies'][policy][k], value, f'{policy}.{k}')
    pairs = [('candidate', 'reference')]
    if 'previous' in policies: pairs += [('candidate', 'previous'), ('previous', 'reference')]
    comparisons = {}
    for left, right in pairs:
        # Each cluster contains the four paired seat differences, not eight independent games.
        deltas = [sum(lookup[seed, seat, left]['score'] - lookup[seed, seat, right]['score']
                      for seat in range(4)) / 4 for seed in seeds]
        mean = statistics.mean(deltas)
        half = float(t.ppf(.975, len(deltas) - 1)) * statistics.stdev(deltas) / math.sqrt(len(deltas))
        key = f'{left}_minus_{right}'
        actual = summary.get('comparisons', {}).get(key)
        if actual is not None:
            equal(actual['gain'], mean, key + '.gain')
            if len(actual['ci95']) != 2: raise ValueError('Invalid confidence interval')
            for a, e in zip(actual['ci95'], [mean - half, mean + half]): equal(a, e, key + '.ci95')
        elif 'previous' in policies: raise ValueError('Missing three-policy comparison')
        comparisons[key] = {'gain': mean, 'ci95': [mean - half, mean + half]}
    primary = comparisons['candidate_minus_reference']
    equal(summary['paired_net_points_gain'], primary['gain'], 'primary gain')
    if len(summary['paired_seed_cluster_ci95']) != 2: raise ValueError('Invalid primary confidence interval')
    for a, e in zip(summary['paired_seed_cluster_ci95'], primary['ci95']): equal(a, e, 'primary CI')
    equal(summary['independent_wall_streams'], len(seeds), 'wall clusters')
    if summary['is_final_confirmation'] != (cfg.get('evaluation_stage') == 'independent_confirmation'):
        raise ValueError('Development/confirmation label mismatch')
    return {'games': len(selected), 'independent_wall_clusters': len(seeds), 'comparisons': comparisons}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--completed-suites-only', action='store_true', help='Partial audit cannot authorize confirmation/deployment')
    args = p.parse_args(); cfg = json.loads(args.config.read_text()); run = ROOT / cfg['out']
    out = run / 'native-development'; protocol = json.loads((out / 'protocol.json').read_text())
    if protocol['config'] != cfg: raise ValueError('Evaluation config changed')
    for name, h in protocol['code_sha256'].items():
        if digest(ROOT / name) != h: raise ValueError(f'Active evaluation code changed: {name}')
    selection = json.loads((run / 'selected.json').read_text()); checkpoint = ROOT / selection['checkpoint']
    if selection != protocol['selected'] or digest(checkpoint) != protocol['candidate_sha256']:
        raise ValueError('Frozen candidate changed')
    if selection.get('checkpoint_sha256', digest(checkpoint)) != digest(checkpoint): raise ValueError('Candidate selection hash mismatch')
    if digest(ROOT / cfg['reference'] / 'head.pt') != cfg['reference_head_sha256']: raise ValueError('Reference head changed')
    if cfg.get('cache_source'):
        source = json.loads((ROOT / cfg['cache_source'] / 'protocol.json').read_text())
        for name, h in source['reference_files'].items():
            if digest(ROOT / cfg['reference'] / name) != h: raise ValueError(f'Frozen reference file changed: {name}')
        vendor = 'vendor/JevAny/jevany/model.py'
        if digest(ROOT / vendor) != source['code_sha256'][vendor]: raise ValueError('Frozen JevAny model implementation changed')
    if cfg.get('previous_call_checkpoint'):
        training = json.loads((run / 'protocol.json').read_text())
        if cfg['previous_call_checkpoint'] != cfg['warm_start'] or digest(ROOT / cfg['previous_call_checkpoint']) != training['warm_start_sha256']:
            raise ValueError('Previous comparison head identity not established')
    status = json.loads((run / 'strength-status.json').read_text())
    complete = status['status'] == 'development_complete'
    if not complete and not args.completed_suites_only: raise ValueError('Full evaluation is not complete')
    raw = (out / 'games-private.jsonl').read_bytes()
    if not raw.endswith(b'\n'):
        if complete: raise ValueError('Completed journal has an incomplete final line')
        raw = raw[:raw.rfind(b'\n') + 1]
    rows = [json.loads(line) for line in raw.splitlines()]
    if any(r['suite'] not in ('closed', 'mixed') for r in rows): raise ValueError('Unexpected opponent suite')
    reports = {}
    for suite in ('closed', 'mixed'):
        path = out / f'{suite}-summary.json'
        if not path.exists():
            if complete: raise ValueError('Complete evaluation missing suite summary')
            continue
        reports[suite] = audit_suite(rows, json.loads(path.read_text()), cfg, suite)
    expected_total = cfg['development_seeds'] * 4 * (3 if cfg.get('previous_call_checkpoint') else 2) * 2
    if complete and (len(rows) != expected_total or status['games'] != expected_total): raise ValueError('Final game count mismatch')
    result = {'status': 'passed_complete' if complete else 'passed_completed_suites_only', 'complete': complete,
        'suites': reports, 'shared_wall_clusters_across_suites': cfg['development_seeds'],
        'raw_records_public': False, 'candidate_sha256': digest(checkpoint), 'deployed': False,
        'note': 'Audit verifies measurement and identity, not statistical improvement or desktop generalization.'}
    path = run / ('development-audit.json' if complete else 'development-partial-audit.json')
    tmp = path.with_suffix('.json.tmp'); tmp.write_text(json.dumps(result, indent=2) + '\n'); tmp.replace(path)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
