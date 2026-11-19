"""Candidate-only tuning rollouts with audited, unchanged-v8 baseline reuse."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import socket
import sys


def standard_summary(rows):
    from jevmahjong.endgame.evaluate import interval, point_and_risk_summary
    groups = defaultdict(list)
    for row in rows:
        groups[row['seed']].append(row['candidate']['utility'] - row['baseline']['utility'])
    differences = [sum(v) / len(v) for v in groups.values()]
    rates = {}
    for initial, name, cutoff in ((3, 'third_to_top2', 2), (4, 'fourth_escape', 3), (0, 'hold_top2', 2)):
        subset = [r for r in rows if r['initial_rank'] <= 2] if initial == 0 else [r for r in rows if r['initial_rank'] == initial]
        if not subset:
            raise ValueError('Standard tuning pool lacks placement coverage')
        values = {p: sum(r[p]['rank'] <= cutoff for r in subset) / len(subset) for p in ('baseline', 'candidate')}
        rates[name] = dict(positions=len(subset), **values,
                          candidate_minus_baseline=values['candidate'] - values['baseline'])
    return {'status': 'reused_pool_endgame_development_complete', 'independent_source_groups': len(groups),
        'paired_positions': len(rows), 'terminal_utility_gain': sum(differences) / len(differences),
        'whole_source_group_paired_bootstrap_ci95': interval(differences), 'placement_rates': rates,
        'points_and_dealins': point_and_risk_summary(rows), 'consumed_development_pool': True,
        'new_candidate_rollouts': len(rows), 'baseline_rollouts_reused': len(rows), 'deployed': False}


def combined_eligible(general, targeted):
    rates, ranks = general['placement_rates'], targeted['by_initial_rank']
    return (general['terminal_utility_gain'] >= 0
        and rates['third_to_top2']['candidate_minus_baseline'] >= 0
        and rates['fourth_escape']['candidate_minus_baseline'] >= 0
        and rates['hold_top2']['candidate_minus_baseline'] >= -.05
        and targeted['paired_utility_gain'] > 1e-12
        and ranks['3']['candidate']['top2_count'] > ranks['3']['baseline']['top2_count']
        and ranks['4']['candidate']['top2_count'] >= ranks['4']['baseline']['top2_count'])


def evaluate(config, reference_run):
    from riichienv import GameRule, RiichiEnv
    from jevmahjong.core.seeds import engine_seed
    from jevmahjong.core.state import build_request
    from jevmahjong.training.futures import replay_prefix
    from jevmahjong.endgame.collect import native_match, digest, write
    from jevmahjong.endgame.context import placements, placement_objective, serving_request
    from jevmahjong.endgame.coverage import public_stratum
    from jevmahjong.endgame.evaluate import rollout
    from jevmahjong.endgame.futures import bundle_identity
    from jevmahjong.endgame.policy import ModelAgent
    from jevmahjong.policy.routing import load_model
    from tools.evaluate_endgame_pursuit import select_targets, summarize as targeted_summary
    root = Path(__file__).resolve().parents[1]
    out, old = Path(config['out']), Path(reference_run)
    fit = json.loads((out / 'fit-result-private.json').read_text())
    if not fit['eligible_for_development']:
        result = {'status': 'consumed_development_skipped_zero_step', 'deployed': False,
                  'eligible_for_independent_confirmation': False}
        write(out / 'combined-development-result-private.json', result)
        return result
    old_config = json.loads((old / 'config-private.json').read_text())
    if any(config[k] != old_config[k] for k in ('checkpoint', 'terminal_step_cap', 'development_groups', 'development_seed_start')):
        raise ValueError('Baseline source range, checkpoint or rollout cap changed')
    if config['development_groups'] != 16 or config['development_seed_start'] != 2026117900:
        raise ValueError('This evaluator may only reuse the registered consumed tuning pool')
    ready = json.loads((out / 'preflight-status-private.json').read_text())
    if ready.get('native_prototype_parity_positions') != 308 or not ready.get('same_source_ids_and_targets'):
        raise ValueError('Original-path/native input-scope checks missing')
    old_launch = json.loads((old / 'fit-launch-protocol-private.json').read_text())
    # These reviewed changes concern input scope and offline targets. The
    # placement_aware=False branch retains the original format/action guards;
    # native integration parity is separately required above. Freeze everything
    # else involved in source generation, features, inference and continuation.
    reviewed_changes = {f'jevmahjong/endgame/{name}.py' for name in ('context', 'policy', 'prepare', 'reference', 'mix', 'fit')}
    for path, expected in old_launch['code_sha256'].items():
        if path not in reviewed_changes and hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
            raise ValueError('Baseline/source implementation changed: ' + path)
    baseline_identity = bundle_identity(config['checkpoint'])
    old_general = json.loads((old / 'development-result-private.json').read_text())
    old_targeted = json.loads((old / 'targeted-development-result-private.json').read_text())
    old_target_protocol = json.loads((old / 'targeted-development-protocol-private.json').read_text())
    if old_general['baseline_identity'] != baseline_identity or old_targeted['baseline_identity'] != baseline_identity:
        raise ValueError('Cached baseline is not the exact released checkpoint')
    target_dir = old / 'targeted-development'
    sources = [json.loads(p.read_text()) for p in sorted((target_dir / 'sources').glob('*.json'))]
    if len(sources) != 256 or {s['seed'] for s in sources} != set(range(2026119900, 2026120156)):
        raise ValueError('Targeted tuning source pool changed')
    for source in sources:
        if source['profile'] != 'dama':
            raise ValueError('Targeted source opponent profile changed')
        for row in source['selected']:
            if public_stratum(row['public_request']) != row['coverage_stratum']:
                raise ValueError('Targeted public-condition selection changed')
    selected = select_targets(sources, 32)
    if selected != json.loads((target_dir / 'selected-positions-private.json').read_text()) or len(selected) != 21:
        raise ValueError('The fixed consumed targeted manifest changed')
    sources_by_seed = {s['seed']: s for s in sources}
    entries = {'standard': [], 'targeted': []}
    for index in range(16):
        source = native_match(2026117900 + index)
        write(out / 'development-sources' / f"{source['seed']}.json", source)
        for position in source['selected']:
            entries['standard'].append((source, position, old / 'development'))
    for position in selected:
        entries['targeted'].append((sources_by_seed[position['seed']], position, target_dir / 'rollouts'))
    if len(entries['standard']) != old_general['paired_positions']:
        raise ValueError('Standard source windows changed')
    cache_hashes = {}
    for cohort, items in entries.items():
        for source, position, directory in items:
            seed, seat = position['seed'], position['seat']
            path = directory / f"{seed}-{position['step']}-{seat}.json"
            row = json.loads(path.read_text())
            if (row['seed'], row['step'], row['seat']) != (seed, position['step'], seat):
                raise ValueError('Cached source/window/seat differs')
            baseline = row['baseline']
            if (not baseline['terminal_ledger_passed'] or placements(baseline['final_scores'])[seat] != baseline['rank']
                    or placement_objective(baseline['rank']) != baseline['utility']
                    or baseline['net_points_from_window'] != baseline['final_scores'][seat] - baseline['start_scores'][seat]):
                raise ValueError('Cached terminal outcome failed score/rank/utility checks')
            if cohort == 'standard' and row['baseline_identity'] != baseline_identity:
                raise ValueError('Standard cached baseline identity differs')
            if cohort == 'targeted' and (row['protocol_sha256'] != digest(old_target_protocol)
                    or row['source_request_sha256'] != position['request_sha256']
                    or row['manifest_sha256'] != digest(selected)):
                raise ValueError('Targeted cached protocol, public input or manifest changed')
            cache_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    candidate = (out / 'training-run' / fit['selection']['selected_checkpoint']).resolve()
    if not candidate.is_relative_to((out / 'training-run/checkpoints').resolve()):
        raise ValueError('Selected checkpoint outside registered training run')
    candidate_identity = bundle_identity(candidate)
    paths = sorted(set(root.glob('jevmahjong/**/*.py')) | {Path(__file__).resolve(),
        root / 'tools/evaluate_endgame_pursuit.py', root / 'desktop/jev_sidecar/unified_policy.py',
        root / 'desktop/jev_sidecar/call_policy.py', root / 'desktop/jev_sidecar/tiles.py'})
    protocol = {'schema': 'consumed-development-baseline-reuse-v1', 'input_version': config['input_version'],
        'candidate_identity': candidate_identity, 'baseline_identity': baseline_identity,
        'terminal_step_cap': config['terminal_step_cap'], 'reviewed_baseline_unaffected_changes': sorted(reviewed_changes),
        'scope': 'consumed standard16/targeted256 tuning pools; not independent confirmation',
        'cached_baseline_files_sha256': cache_hashes,
        'code_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    protocol_file = out / 'reused-development-protocol-private.json'
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
        raise ValueError('Existing tuning protocol changed; no silent resume')
    write(protocol_file, protocol)
    learner = ModelAgent(load_model(candidate), placement_aware=True, input_version=config['input_version'])
    results = {}
    for cohort, items in entries.items():
        records = []
        for source, position, directory in items:
            seed, seat = position['seed'], position['seat']
            name = f"{seed}-{position['step']}-{seat}.json"
            cached = json.loads((directory / name).read_text())
            path = out / ('development' if cohort == 'standard' else 'targeted-development/rollouts') / name
            if path.exists():
                row = json.loads(path.read_text())
                if row['protocol_sha256'] != digest(protocol):
                    raise ValueError('Existing candidate rollout protocol differs')
            else:
                env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=engine_seed(seed, 'spaced-v1'))
                replay_prefix(env, source['mjai_log'], position['step'])
                if placements(list(env.scores()))[seat] != cached['initial_rank'] or list(env.scores()) != cached['baseline']['start_scores']:
                    raise ValueError('Replayed window scores/rank differ from cached baseline')
                obs = env.get_observations()[seat]
                request, _, _ = build_request(obs); request['state']['hand_number'] = int(obs.kyoku_index) + 1
                if digest(serving_request(request)) != position['request_sha256']:
                    raise ValueError('Replayed public source request differs')
                row = {k: cached[k] for k in ('seed', 'seat', 'step', 'initial_rank', 'baseline')}
                row.update(candidate=rollout(env, seat, seed, learner, config['terminal_step_cap']),
                    baseline_identity=baseline_identity, candidate_identity=candidate_identity,
                    protocol_sha256=digest(protocol), baseline_reused=True)
                write(path, row)
            records.append(row)
            write(out / 'reused-development-status-private.json', {'status': 'consumed_tuning_candidate_only',
                'cohort': cohort, 'positions_complete': len(records), 'positions_total': len(items)})
        results[cohort] = standard_summary(records) if cohort == 'standard' else targeted_summary(records)
        results[cohort].update(candidate_identity=candidate_identity, baseline_identity=baseline_identity,
            protocol_sha256=digest(protocol), consumed_development_pool=True, deployed=False)
        if cohort == 'targeted':
            results[cohort]['scope'] = 'consumed targeted tuning pool; not fresh independent evidence'
    for path, sha in protocol['code_sha256'].items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != sha:
            raise ValueError('Source changed during candidate-only tuning')
    for path, sha in cache_hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError('Read-only baseline cache changed')
    eligible = combined_eligible(results['standard'], results['targeted'])
    results['standard']['eligible_for_independent_confirmation'] = eligible
    write(out / 'development-result-private.json', results['standard'])
    write(out / 'targeted-development-result-private.json', results['targeted'])
    combined = {'status': 'combined_consumed_development_complete', 'standard': results['standard'],
        'targeted': results['targeted'], 'candidate_identity': candidate_identity,
        'eligible_for_independent_confirmation': eligible, 'independent_test': False, 'deployed': False}
    write(out / 'combined-development-result-private.json', combined)
    write(out / 'pipeline-status-private.json', {'status': 'seventh_fit_and_standard_development_complete',
        'combined_development_complete': True, 'eligible_for_independent_confirmation': eligible, 'deployed': False})
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--reference-run', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS') or socket.gethostname().startswith(('lo-', 'login')):
        raise RuntimeError('Evaluation requires a scheduled GPU job, never login')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    config = json.loads(args.config.read_text()); sys.path.insert(0, config['upstream_source'])
    print(json.dumps(evaluate(config, args.reference_run), indent=2), flush=True)


if __name__ == '__main__':
    main()
