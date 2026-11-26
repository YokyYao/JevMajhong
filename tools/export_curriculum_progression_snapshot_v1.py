"""Snapshot completed 3A development observations without changing a live fit.

Run on companion CPU resources inside an existing scheduler GPU allocation.
Raw journals and native matches stay private; only aggregate figures/reports
are intended for publication. This does not grade or admit a later course.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def require_resources():
    job = os.environ.get('SLURM_JOB_ID')
    node = socket.gethostname()
    if not job or node.startswith(('lo-', 'login')):
        raise RuntimeError('An active scheduler GPU allocation is required')
    actual = subprocess.check_output(['scontrol', 'show', 'job', '-o', job], text=True)
    fields = dict(part.split('=', 1) for part in actual.split() if '=' in part)
    if fields.get('JobState') != 'RUNNING' or fields.get('NodeList') != node or 'gres/gpu=' not in fields.get('AllocTRES', ''):
        raise RuntimeError('Actual allocation/node/GPU grant differs')


def complete_rows(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines(keepends=True)
            if line.endswith(b'\n') and line.strip()]


def export(registered, out, through):
    require_resources()
    reg = read(registered)
    from tools.curriculum_progression_runtime_v1 import require_registration, policy_contract
    require_registration(reg)
    c = reg['curriculum']
    if through not in (600, 1200, 1800):
        raise ValueError('Intermediate snapshots only; final evidence needs its independent audit')
    sys.path.insert(0, reg['source'])
    sys.path.insert(0, reg['training_config']['upstream_source'])
    from jevmahjong.core.scoring import settlement_ledger
    from jevmahjong.evaluation.metrics import hand_dealin_count
    from jevmahjong.training.metric_journal import aggregate_complete_matches
    from jevmahjong.evaluation.unified_contract import bundle
    from tools.plot_curriculum_fixed_grid import render
    fit = ROOT / c['out']
    captured = {kind: [r for r in complete_rows(fit / 'metrics' / f'{kind}_metrics.jsonl')
                       if r['local_step'] <= through] for kind in ('train', 'validation', 'evaluation')}
    if [r['local_step'] for r in captured['train']] != list(range(1, through + 1)):
        raise ValueError('Every actual optimizer update through snapshot required')
    points = list(range(0, through + 1, 600))
    if any([r['local_step'] for r in captured[k]] != points for k in ('validation', 'evaluation')):
        raise ValueError('Full validation and native results required for each snapshot point')
    if any(r['global_step'] != 996 + r['local_step'] or r['stage'] != c['stage']
           for rows in captured.values() for r in rows):
        raise ValueError('Actual parameter-ancestry step or stage changed')
    if any(r.get('reward') is not None for r in captured['train']):
        raise ValueError('SFT optimizer updates do not have game reward')
    cells = {(seed, profile, seat) for seed in range(c['curve_seed_start'], c['curve_seed_start'] + 2)
             for profile in c['opponent_profiles'] for seat in range(4)}

    def check_games(games):
        if len(games) != 16 or {(g['seed'], g['profile'], g['seat']) for g in games} != cells:
            raise ValueError('Exactly the same consumed16 native cells required')
        for g in games:
            seat = g['seat']; log = g['mjai_log']
            ledger = settlement_ledger(log, g['final_scores'])
            rank = sorted(range(4), key=lambda p: (-g['final_scores'][p], p)).index(seat) + 1
            if (not g['completed'] or not g['ledger_passed'] or ledger['hands'] != g['hands']
                    or g['net_points'] != g['final_scores'][seat] - 25000 or rank != g['rank']
                    or hand_dealin_count(log, seat) != g['dealin_hands']
                    or sum(e['type'] == 'hora' and e.get('actor') == seat for e in log) != g['wins']):
                raise ValueError('Native financial/rank/hand counters differ')
        return aggregate_complete_matches(games)

    comparison = {}
    for row in captured['evaluation']:
        step = row['local_step']; folder = fit / 'evaluations' / f'step-{step:06d}'
        games = complete_rows(folder / 'matches-private.jsonl')
        actual = check_games(games)
        identity = bundle(fit / 'training-run' / 'checkpoints' / f'step-{step:06d}')
        spec, policy_id = policy_contract(identity, reg)
        protocol = read(folder / 'protocol.json')
        if (any(row[k] != v for k, v in actual.items()) or protocol['checkpoint'] != identity
                or protocol['policy'] != spec or protocol['policy_identity'] != policy_id
                or row['checkpoint_sha256'] != identity['identity_sha256'] or row['policy_identity'] != policy_id):
            raise ValueError('Native journal/model/policy identity differs')
        comparison[f'step{step}'] = actual
    reference = [g for g in complete_rows(ROOT / c['reference_evaluation_cache'] / 'matches-private.jsonl')
                 if (g['seed'], g['profile'], g['seat']) in cells]
    if bundle(c['reference_checkpoint'])['identity_sha256'] != c['reference_identity']:
        raise ValueError('Frozen v8 parameter identity differs')
    comparison['frozen_v8'] = check_games(reference)
    out.mkdir(parents=True, exist_ok=False)
    digests = {}
    for kind, rows in captured.items():
        raw = ''.join(json.dumps(row) + '\n' for row in rows).encode()
        (out / f'{kind}_metrics.jsonl').write_bytes(raw)
        digests[kind] = hashlib.sha256(raw).hexdigest()
    render(out)
    proof = dict(status='actual_intermediate_3A_snapshot_not_final_confirmation', snapshot_through=through,
                 actual_optimizer_updates=through, actual_native_points=points, comparison=comparison,
                 checked_native_ledgers=16 * (len(points) + 1), snapshot_sha256=digests,
                 current_fit_and_weights_unchanged=True, new_model_forward_or_optimizer_updates=0,
                 reserved_grid_opened=False, only_two_consumed_logical_seed_groups=True,
                 reference_original_guarded_student_complete_legal=True, raw_snapshots_private=True)
    (out / 'summary-private.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps({k: proof[k] for k in ('status', 'snapshot_through', 'checked_native_ledgers', 'comparison')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registered', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--through-step', type=int, required=True)
    args = parser.parse_args()
    export(args.registered.resolve(), args.out.resolve(), args.through_step)
