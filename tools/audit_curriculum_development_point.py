"""Audit a completed interim game's exact matched subset; no new model games."""
import argparse
import json
import os
from pathlib import Path
import socket
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_basic_repair import grade
from tools.curriculum_transfer import read, sha


def audit(registered, step, out, control=None):
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS')
        or socket.gethostname().startswith(('lo-', 'login'))):
        raise RuntimeError('Bounded allocated GPU job required')
    reg = read(registered); c = reg['curriculum']; fit = ROOT/c['out']
    if not all(sha(p) == h for p, h in reg['inputs'].items()):
        raise ValueError('Active registered input changed')
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.training.metric_journal import aggregate_complete_matches
    from jevmahjong.core.scoring import settlement_ledger
    from jevmahjong.evaluation.metrics import hand_dealin_count
    from jevmahjong.evaluation.unified_contract import bundle
    if package_digest(Path(reg['source'])/'jevmahjong') != reg['source_sha256']:
        raise ValueError('Pinned implementation changed')
    current_path = fit/'evaluations'/f'step-{step:06d}'
    if read(current_path/'status.json') != {'status': 'complete', 'games': 16}:
        raise ValueError('This entry audits only completed periodic sixteen-match points')
    protocol = read(current_path/'protocol.json'); current_identity = bundle(fit/'training-run/checkpoints'/f'step-{step:06d}')
    if (protocol['checkpoint'] != current_identity or protocol['seeds'] != list(range(c['curve_seed_start'], c['curve_seed_start']+c['curve_seed_groups']))
        or protocol['profiles'] != c['opponent_profiles'] or protocol['seats'] != 4):
        raise ValueError('Actual interim checkpoint/grid differs from registration')
    expected = {(seed, profile, seat) for seed in protocol['seeds'] for profile in protocol['profiles'] for seat in range(4)}
    inputs = {str(Path(__file__).resolve()): sha(__file__), str(registered): sha(registered)}
    def load(path):
        source_protocol = read(path/'protocol.json'); summary = read(path/'summary.json')
        rows = [json.loads(line) for line in (path/'matches-private.jsonl').open()]
        all_keys = {(seed, profile, seat) for seed in source_protocol['seeds'] for profile in protocol['profiles'] for seat in range(4)}
        bykey = {(r['seed'], r['profile'], r['seat']): r for r in rows}
        if (source_protocol['profiles'] != protocol['profiles'] or source_protocol['seats'] != 4
            or len(bykey) != len(rows) or set(bykey) != all_keys or not expected <= all_keys
            or source_protocol['checkpoint']['identity_sha256'] != summary['checkpoint_sha256']):
            raise ValueError('Incomplete/duplicate/mismatched reference grid or identity')
        selected = [bykey[k] for k in sorted(expected)]
        for r in selected:
            seat = r['seat']; ledger = settlement_ledger(r['mjai_log'], r['final_scores'])
            if (r['completed'] is not True or r['ledger_passed'] is not True or ledger['hands'] != r['hands']
                or r['net_points'] != r['final_scores'][seat]-25000
                or sorted(range(4), key=lambda p: (-r['final_scores'][p], p)).index(seat)+1 != r['rank']
                or hand_dealin_count(r['mjai_log'], seat) != r['dealin_hands']
                or sum(e['type'] == 'hora' and e.get('actor') == seat for e in r['mjai_log']) != r['wins']):
                raise ValueError('Actual complete match or financial ledger check failed')
        for name in ('protocol.json', 'summary.json', 'matches-private.jsonl'):
            inputs[str((path/name).resolve())] = sha(path/name)
        return selected, aggregate_complete_matches(selected), summary['checkpoint_sha256']
    current, current_summary, identity = load(current_path)
    for k, v in current_summary.items():
        if read(current_path/'summary.json')[k] != v:
            raise ValueError('Cached current summary differs from recomputed sixteen matches')
    if identity != current_identity['identity_sha256']:
        raise ValueError('Saved student identity differs')
    references = {'initial_fresh_step0': fit/'evaluations/step-000000', 'frozen_v8': ROOT/c['reference_evaluation_cache']}
    if control is not None: references['completed_control'] = control
    comparisons = {}
    functions = {'net_points': lambda r: r['net_points'], 'top2_rate': lambda r: float(r['rank'] <= 2),
        'hand_win_rate': lambda r: r['wins']/r['hands'], 'dealin_hand_rate': lambda r: r['dealin_hands']/r['hands'],
        'first_rate': lambda r: float(r['rank'] == 1), 'fourth_rate': lambda r: float(r['rank'] == 4)}
    for name, path in references.items():
        rows, summary, reference_identity = load(path)
        if name == 'frozen_v8' and reference_identity != c['reference_identity']:
            raise ValueError('Wrong frozen v8 reference')
        if name == 'completed_control':
            control_bundle = bundle(path.parent.parent/'training-run/checkpoints'/path.name)
            if control_bundle['identity_sha256'] != reference_identity:
                raise ValueError('Completed control checkpoint identity differs from its game cache')
        bykey = {(r['seed'], r['profile'], r['seat']): r for r in rows}
        gain = {field: statistics.mean(fn(r)-fn(bykey[r['seed'], r['profile'], r['seat']]) for r in current)
            for field, fn in functions.items()}
        comparisons[name] = dict(identity=reference_identity, matched_summary=summary, mean_paired_gains=gain,
            seed_groups=len(protocol['seeds']), ci95=None, insufficient_clusters_for_strength_claim=True)
    validation_snapshot = (fit/'metrics/validation_metrics.jsonl').read_text()
    validations = [json.loads(line) for line in validation_snapshot.splitlines()]
    point = next(p for p in validations if p['local_step'] == step)
    # Validation is append-only while the fit continues. Capture the exact point
    # and a snapshot of its observed file, never bind that live file into the fit.
    result = dict(status='completed_interim_matched_development_point_audited', step=step,
        summary=current_summary, comparisons=comparisons, source_quality=grade(point), validation_point=point,
        interim_only=True, not_a_candidate_selection=True, full608_fit_unchanged=True,
        no_new_model_games_or_labels=True, final_reserved_grid_unused=True)
    out.mkdir(parents=True, exist_ok=False)
    snapshot_path = out/'observed-validation-private.jsonl'
    snapshot_path.write_text(validation_snapshot)
    inputs[str(snapshot_path.resolve())] = sha(snapshot_path)
    for name, value in (('summary-private.json', result), ('binding-private.json', {'files': inputs,
        'checkpoint': current_identity, 'job_id': os.environ['SLURM_JOB_ID'], 'live_validation_file_is_snapshot_only': True})):
        with (out/name).open('x') as stream: json.dump(value, stream, indent=2); stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registered', type=Path, required=True); p.add_argument('--step', type=int, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--control', type=Path)
    a = p.parse_args(); audit(a.registered.resolve(), a.step, a.out.resolve(), a.control.resolve() if a.control else None)
