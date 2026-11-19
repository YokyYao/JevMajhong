"""Fresh, public-condition endgame pursuit cohort alongside general regression."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import sys


SCHEDULE = {
    'development': {'seed_start': 2026119900, 'groups': 256, 'cap': 32, 'minimum': 12, 'minimum_per_rank': 4},
    'confirmation': {'seed_start': 2026121900, 'groups': 512, 'cap': 64, 'minimum': 24, 'minimum_per_rank': 8},
}


def select_targets(sources, cap):
    """One public position per source; cell round robin, hash order within cell."""
    cells = {}
    for source in sources:
        candidates = [r for r in source['selected'] if r['coverage_stratum']['target']]
        if not candidates:
            continue
        row = min(candidates, key=lambda r: r['selection_key'])
        cells.setdefault(row['coverage_stratum']['cell'], []).append(row)
    for values in cells.values():
        values.sort(key=lambda r: r['selection_key'])
    selected = []
    while len(selected) < cap:
        added = False
        for cell in sorted(cells):
            if cells[cell] and len(selected) < cap:
                selected.append(cells[cell].pop(0))
                added = True
        if not added:
            break
    if len({r['seed'] for r in selected}) != len(selected):
        raise ValueError('Targeted evaluation must use distinct original sources')
    return selected


def summarize(records):
    from jevmahjong.endgame.evaluate import interval
    if not records or len({r['seed'] for r in records}) != len(records):
        raise ValueError('Expected nonempty one-window-per-source evaluation')
    differences = [r['candidate']['utility'] - r['baseline']['utility'] for r in records]
    points = [r['candidate']['net_points_from_window'] - r['baseline']['net_points_from_window'] for r in records]
    by_rank = {}
    for rank in (3, 4):
        rows = [r for r in records if r['initial_rank'] == rank]
        if not rows:
            raise ValueError('Targeted cohort lacks a required initial rank')
        by_rank[str(rank)] = {'positions': len(rows)}
        for name in ('baseline', 'candidate'):
            by_rank[str(rank)][name] = {
                'top2_count': sum(r[name]['rank'] <= 2 for r in rows),
                'top2_rate': sum(r[name]['rank'] <= 2 for r in rows) / len(rows),
                'escape_fourth_count': sum(r[name]['rank'] < 4 for r in rows),
                'escape_fourth_rate': sum(r[name]['rank'] < 4 for r in rows) / len(rows),
                'mean_utility': sum(r[name]['utility'] for r in rows) / len(rows)}
    return {'positions': len(records), 'source_groups': len(records), 'by_initial_rank': by_rank,
        'paired_utility_gain': sum(differences) / len(differences),
        'paired_source_bootstrap_utility_ci95': interval(differences),
        'paired_net_point_gain': sum(points) / len(points),
        'paired_source_bootstrap_net_point_ci95': interval(points),
        'per_policy_declined_win_events': {name: sum(r[name]['declined_wins'] for r in records)
                                         for name in ('baseline', 'candidate')},
        'scope': 'fresh rule-generated public-condition windows, one per source; not full-start or human win rate'}


def evaluate(config, phase):
    import hashlib
    from riichienv import GameRule, RiichiEnv
    from jevmahjong.core.seeds import engine_seed
    from jevmahjong.core.state import build_request
    from jevmahjong.training.futures import replay_prefix
    from jevmahjong.endgame.collect import digest, write
    from jevmahjong.endgame.coverage import SCHEMA, coverage_match, public_stratum
    from jevmahjong.endgame.context import placements, serving_request
    from jevmahjong.endgame.evaluate import rollout
    from jevmahjong.endgame.futures import bundle_identity
    from jevmahjong.endgame.policy import ModelAgent
    from jevmahjong.policy.routing import load_model
    out = Path(config['out'])
    fit = json.loads((out / 'fit-result-private.json').read_text())
    name = 'targeted-' + phase
    if not fit['eligible_for_development']:
        result = {'status': 'targeted_evaluation_skipped_zero_step', 'deployed': False}
        write(out / (name + '-result-private.json'), result)
        return result
    pipeline = json.loads((out / 'pipeline-status-private.json').read_text())
    if not pipeline['status'].endswith('_fit_and_standard_development_complete'):
        raise ValueError('Finish the registered fit and general development before targeted evaluation')
    candidate = (out / 'training-run' / fit['selection']['selected_checkpoint']).resolve()
    if not candidate.is_relative_to((out / 'training-run/checkpoints').resolve()):
        raise ValueError('Selected checkpoint outside registered fit output')
    baseline_identity, candidate_identity = bundle_identity(config['checkpoint']), bundle_identity(candidate)
    if phase == 'confirmation':
        qualification = json.loads((out / 'combined-development-result-private.json').read_text())
        if not qualification['eligible_for_independent_confirmation'] or qualification['candidate_identity'] != candidate_identity:
            raise ValueError('Confirmation needs the fixed combined-qualified candidate')
    schedule = SCHEDULE[phase]
    fitted = json.loads((out / 'selected-positions-private.json').read_text())
    seeds = set(range(schedule['seed_start'], schedule['seed_start'] + schedule['groups']))
    if seeds & {r['seed'] for r in fitted}:
        raise ValueError('Targeted evaluation overlaps fitted sources')
    root = Path(__file__).resolve().parents[1]
    paths = sorted(set(root.glob('jevmahjong/**/*.py')) | {Path(__file__).resolve(),
        root / 'desktop/jev_sidecar/unified_policy.py', root / 'desktop/jev_sidecar/call_policy.py',
        root / 'desktop/jev_sidecar/tiles.py'})
    protocol = {'schema': 'fresh-pursuit-evaluation-v1', 'phase': phase, 'schedule': schedule,
        'candidate_identity': candidate_identity, 'baseline_identity': baseline_identity,
        'input_version': config['input_version'], 'terminal_step_cap': config['terminal_step_cap'],
        'source_profile': 'dama', 'selection': 'public_target_one_per_source_then_cell_hash_round_robin',
        'code_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    protocol_path = out / (name + '-protocol-private.json')
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Existing targeted protocol differs; inspect before resuming')
    write(protocol_path, protocol)
    directory = out / name
    sources = []
    for index in range(schedule['groups']):
        seed = schedule['seed_start'] + index
        path = directory / 'sources' / f'{seed}.json'
        if path.exists():
            source = json.loads(path.read_text())
            if source['schema'] != SCHEMA or source['profile'] != 'dama' or source['seed'] != seed:
                raise ValueError('Cached targeted source differs')
        else:
            source = coverage_match(seed, 'dama')
            write(path, source)
        for row in source['selected']:
            if public_stratum(row['public_request']) != row['coverage_stratum']:
                raise ValueError('Public selection features changed')
        sources.append(source)
        write(out / (name + '-status-private.json'), {'status': 'collecting_fresh_public_target_sources',
            'groups_complete': index + 1, 'groups_total': schedule['groups'], 'model_rollouts': 0})
    selected = select_targets(sources, schedule['cap'])
    ranks = Counter(r['coverage_stratum']['placement'] for r in selected)
    if len(selected) < schedule['minimum'] or any(ranks[r] < schedule['minimum_per_rank'] for r in (3, 4)):
        result = {'status': 'targeted_public_coverage_insufficient', 'positions': len(selected),
                  'by_initial_rank': dict(ranks), 'model_rollouts': 0, 'deployed': False}
        write(out / (name + '-result-private.json'), result)
        return result
    manifest_path = directory / 'selected-positions-private.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != selected:
        raise ValueError('Existing targeted position manifest differs')
    write(manifest_path, selected)
    manifest_sha = digest(selected)
    source_by_seed = {source['seed']: source for source in sources}
    baseline = ModelAgent(load_model(config['checkpoint']), placement_aware=False)
    learner = ModelAgent(load_model(candidate), placement_aware=True, input_version=config['input_version'])
    records = []
    for position in selected:
        seed, seat = position['seed'], position['seat']
        path = directory / 'rollouts' / f"{seed}-{position['step']}-{seat}.json"
        if path.exists():
            row = json.loads(path.read_text())
            if (row['protocol_sha256'] != digest(protocol) or row['manifest_sha256'] != manifest_sha
                    or row['source_request_sha256'] != position['request_sha256']):
                raise ValueError('Existing targeted rollout has different provenance')
        else:
            env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=engine_seed(seed, 'spaced-v1'))
            replay_prefix(env, source_by_seed[seed]['mjai_log'], position['step'])
            obs = env.get_observations()[seat]
            request, _, _ = build_request(obs)
            request['state']['hand_number'] = int(obs.kyoku_index) + 1
            if digest(serving_request(request)) != position['request_sha256']:
                raise ValueError('Replayed public target request differs from source')
            initial_rank = placements(list(env.scores()))[seat]
            if initial_rank != position['coverage_stratum']['placement']:
                raise ValueError('Replayed initial placement differs')
            row = {'seed': seed, 'seat': seat, 'step': position['step'], 'initial_rank': initial_rank,
                'source_request_sha256': position['request_sha256'], 'protocol_sha256': digest(protocol),
                'manifest_sha256': manifest_sha,
                'baseline': rollout(env.clone(), seat, seed, baseline, config['terminal_step_cap']),
                'candidate': rollout(env.clone(), seat, seed, learner, config['terminal_step_cap'])}
            write(path, row)
        records.append(row)
        write(out / (name + '-status-private.json'), {'status': 'paired_targeted_terminal_evaluation',
            'positions_complete': len(records), 'positions_total': len(selected)})
    for path, expected in protocol['code_sha256'].items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
            raise ValueError('Code changed during targeted evaluation')
    if digest(json.loads(manifest_path.read_text())) != manifest_sha:
        raise ValueError('Targeted manifest changed during evaluation')
    result = dict(summarize(records), status='fresh_targeted_evaluation_complete',
        phase=phase, original_source_groups=schedule['groups'], schedule=schedule,
        candidate_identity=candidate_identity, baseline_identity=baseline_identity,
        protocol_sha256=digest(protocol), deployed=False)
    write(out / (name + '-result-private.json'), result)
    write(out / (name + '-status-private.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--phase', choices=tuple(SCHEDULE), default='development')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS') or socket.gethostname().startswith(('lo-', 'login')):
        raise RuntimeError('Evaluation requires a scheduled GPU job, never login')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    config = json.loads(args.config.read_text())
    sys.path.insert(0, config['upstream_source'])
    print(json.dumps(evaluate(config, args.phase), indent=2), flush=True)


if __name__ == '__main__':
    main()
