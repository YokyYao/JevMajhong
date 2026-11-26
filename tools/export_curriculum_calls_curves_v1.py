"""Minimal real calls-course curves; private journals never enter public CSV.

No model load, new game, synthetic observation, smoothing or grid mixing.
Running journal bytes are snapshotted privately and may lag the live fit.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write, MASS
from tools.curriculum_calls_classroute_v1 import METHOD, STAGE, profile, require_recipe

TRAIN_FIELDS = ('local_step', 'loss', 'ce', 'lr', 'head_lr_derived', 'records_seen')
GAME_FIELDS = ('matches', 'seed_groups', 'hand_win_rate', 'dealin_hand_rate', 'first_rate',
               'top2_rate', 'fourth_rate', 'net_points', 'reward')
OPERATIONS = ('chi', 'pon', 'ankan', 'daiminkan', 'kakan')
POINTS = (0, 194, 388)


def aligned_record(row):
    # Keep this projection pure: importing the older plotting wrapper before
    # pinning runtime would import the workspace's jevmahjong package.
    result = copy.deepcopy(row)
    if row['matches'] == 64:
        subset = row.get('curve_grid_subset')
        if not subset or subset.get('matches') != 16 or subset.get('seed_groups') != 2:
            raise ValueError('Verified matching16 subset required for a full64 point')
        result.update(subset)
    elif row['matches'] != 16 or row['seed_groups'] != 2:
        raise ValueError('Only the same16-game process grid may be connected')
    return result


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Finite real scalar required')
    return value


def complete_jsonl(raw):
    # Ignore only an unfinished final append; malformed complete lines fail.
    return [json.loads(line) for line in raw.splitlines(keepends=True) if line.endswith(b'\n')]


def prefix(rows, through):
    steps = [r['local_step'] for r in rows]
    if not steps or any(type(s) is not int for s in steps) or steps != list(POINTS[:len(steps)]) or steps[-1] > through:
        raise ValueError('Actual uninterrupted0/194/388 metric prefix required')
    return steps


def check_train(train, step, weights):
    if type(step) is not int or not 1 <= step <= 388 or [r['local_step'] for r in train] != list(range(1, step+1)):
        raise ValueError('Every actual update through a registered budget snapshot required')
    if set(weights) != set(MASS) or any(finite(w) <= 0 for w in weights.values()):
        raise ValueError('Registered positive13-source weights required')
    for row in train:
        n = row['local_step']
        if (row['stage'] != STAGE or row['loss_method'] != METHOD or row['global_step'] != 608+n
                or row.get('reward') is not None or row['records_seen'] != (n//194)*49022+(n%194)*253):
            raise ValueError('Course, objective, supervised reward or exposure changed')
        for name in TRAIN_FIELDS: finite(row[name])
        counts = row['source_counts']
        if (set(counts) != set(MASS) or any(type(v) is not int or v <= 0 for v in counts.values())
                or sum(counts.values()) != (193 if n in (194, 388) else 253)):
            raise ValueError('Actual update must retain all13 strata and its registered tail')
        for field, expected in (('source_objective', row['loss']), ('source_ce', row['ce'])):
            actual = sum(counts[s]*weights[s]*finite(row[field][s]) for s in counts)/sum(counts.values())
            if abs(actual-expected) > 1e-5:
                raise ValueError('Actual objective/original CE differs from source observations')
        if not math.isclose(row['head_lr_derived'], 5*row['lr'], abs_tol=1e-12):
            raise ValueError('Actual head/backbone learning-rate ratio changed')


def recalls(counts):
    if any(type(counts.get(k, 0)) is not int or counts.get(k, 0) < 0 for k in ('tp', 'fn', 'tn', 'fp')):
        raise ValueError('Actual nonnegative integer confusion counts required')
    yes = counts.get('tp', 0)+counts.get('fn', 0); no = counts.get('tn', 0)+counts.get('fp', 0)
    if not yes or not no: raise ValueError('Both actual classes required; never invent a recall')
    return counts.get('tp', 0)/yes, counts.get('tn', 0)/no


def check_validation(rows, through):
    prefix(rows, through)
    for row in rows:
        if row['stage'] != STAGE or row['loss_method'] != METHOD or row['global_step'] != 608+row['local_step']:
            raise ValueError('Wrong validation course/objective')
        for key in ('loss', 'ce', 'phase_weighted_ce', 'natural_target_ce'): finite(row[key])
        for name in ('basic_replay', 'raw_basic_replay'):
            b = row[name]
            for source, yes, no in (('legacy', 37, 70), ('teacher', 287, 238)):
                c = b['riichi_by_source'][source]; recalls(c)
                if c.get('tp', 0)+c.get('fn', 0) != yes or c.get('tn', 0)+c.get('fp', 0) != no:
                    raise ValueError('Basic RI source denominator changed')
            recalls(b['riichi_type_counts'])
            if not 0 <= finite(b['stratum_accuracy']['plain']) <= 1:
                raise ValueError('Invalid actual plain agreement')


def check_games(rows, through, validation_last):
    prefix(rows, through)
    if rows[-1]['local_step'] > validation_last: raise ValueError('Game point precedes its saved validation')
    matched = []; full = []
    for row in rows:
        if (row['stage'] != STAGE or row['global_step'] != 608+row['local_step']
                or row['reward_definition'] != 'terminal_raw_net_points_div8000_v1'):
            raise ValueError('Wrong native course or reward definition')
        if row['local_step'] == 388:
            if row['matches'] != 64 or row['seed_groups'] != 8: raise ValueError('Final must retain full64')
            full.append(row)
        elif row['matches'] != 16 or row['seed_groups'] != 2:
            raise ValueError('Process games must use the same16-match grid')
        point = aligned_record(row); matched.append(point)
        for value in (row, point):
            for key in GAME_FIELDS: finite(value[key])
            if not math.isclose(value['reward'], value['net_points']/8000, abs_tol=1e-12):
                raise ValueError('Actual net/reward mismatch')
            for key in ('hand_win_rate', 'dealin_hand_rate', 'first_rate', 'top2_rate', 'fourth_rate'):
                if not 0 <= value[key] <= 1: raise ValueError('Invalid native rate')
    return matched, full


def require_final_packet(evidence, step, validation, matched, full):
    if (step != 388 or [r['local_step'] for r in validation] != list(POINTS)
            or [r['local_step'] for r in matched] != list(POINTS) or len(full) != 1
            or evidence.get('status') != 'stage2_calls_classroute_final_evidence_audit_passed'
            or evidence.get('optimizer_steps') != 388 or evidence.get('real_records_seen') != 98044
            or evidence.get('saved_final_reloaded_menus') != 12479
            or not 0 <= evidence.get('saved_final_max_probability_error', float('inf')) < 5e-5
            or evidence.get('checked_complete_games') != 224
            or evidence.get('all376_optimizer_states_match') is not True
            or evidence.get('actual_parent16_full_events_exact') is not True
            or evidence.get('source_and_data_hashes_unchanged') is not True
            or evidence.get('exact_train_validation_replay') is not True):
        raise ValueError('Actual full388 independent evidence required for a final export')


def project_validation(rows):
    projected = []
    for row in rows:
        p = dict(local_step=row['local_step'], loss=row['loss'], ce=row['ce'],
                 phase_weighted_ce=row['phase_weighted_ce'], natural_target_ce=row['natural_target_ce'])
        for name, prefix_name in (('basic_replay', 'selected'), ('raw_basic_replay', 'raw')):
            b = row[name]; p[prefix_name+'_plain_agreement'] = b['stratum_accuracy']['plain']
            for source, c in [('pooled', b['riichi_type_counts']), *b['riichi_by_source'].items()]:
                yes, no = recalls(c)
                p[prefix_name+'_'+source+'_RI_accept'] = yes; p[prefix_name+'_'+source+'_RI_decline'] = no
        projected.append(p)
    return projected


def csv_file(path, fields, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def export(registered, out, through=None, final=False):
    gpu_only(); reg = read(registered); c = reg['curriculum']
    if reg['status'] != 'stage2_calls_classroute_registered_not_started': raise ValueError('Wrong frozen registration')
    require_recipe(c, read(c['data_ready']))
    for p, h in reg['inputs'].items():
        if sha(p) != h: raise ValueError('Registered input changed: '+p)
    out.mkdir(parents=True, exist_ok=False); private = out/'snapshots-private'; public = out/'public-ready'
    private.mkdir(); public.mkdir(); captured = {}; captured_hashes = {}
    fit = ROOT/c['out']
    for kind in ('train', 'validation', 'evaluation'):
        path = fit/'metrics'/f'{kind}_metrics.jsonl'; raw = path.read_bytes()
        (private/path.name).write_bytes(raw); captured[kind] = complete_jsonl(raw)
        captured_hashes[path.name] = hashlib.sha256(raw).hexdigest()
    if not captured['train']: raise ValueError('No real optimizer updates to export')
    step = through if through is not None else captured['train'][-1]['local_step']
    train, val, games = ([r for r in captured[k] if r['local_step'] <= step] for k in ('train', 'validation', 'evaluation'))
    check_train(train, step, reg['course_loss_weights']); check_validation(val, step)
    matched, full = check_games(games, step, val[-1]['local_step'])
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    from jevmahjong.training.base_lora import target_cross_entropy
    from jevmahjong.training.metric_journal import aggregate_complete_matches
    from jevmahjong.core.scoring import settlement_ledger
    from jevmahjong.evaluation.metrics import hand_dealin_count
    from jevmahjong.evaluation.unified_contract import bundle
    from tools.evaluate_curriculum_calls_classroute_v1 import policy_contract
    import jevmahjong
    if Path(jevmahjong.__file__).resolve().parent != (Path(reg['source'])/'jevmahjong').resolve():
        raise ValueError('Curve validation must use the actual pinned runtime, not the workspace package')
    data = read(c['data_ready']); records = [json.loads(line) for line in (ROOT/data['files']['validation']['path']).open()]
    for row in val:
        point = row['local_step']; saved = read(fit/'training-run/training_eval'/f'step-{point:06d}'/'development/rows.json')
        actual = profile(records, saved, MASS, {'weighted_target_ce': row['phase_weighted_ce']}, target_cross_entropy)
        if abs(actual['weighted_class_objective']-row['loss']) > 1e-12 or abs(actual['weighted_target_ce']-row['ce']) > 1e-12:
            raise ValueError('Cached validation objective/common CE differs from journal')
        for name in ('basic_replay', 'raw_basic_replay', 'operation_type_counts', 'raw_operation_type_counts',
                     'basic_replay_source_quality', 'raw_basic_source_quality'):
            if actual[name] != row[name]: raise ValueError('Actual cached validation differs: '+name)
    checked_games = 0
    for row in games:
        point = row['local_step']; folder = fit/'evaluations'/f'step-{point:06d}'
        protocol = read(folder/'protocol.json'); raw_games = [json.loads(line) for line in (folder/'matches-private.jsonl').open()]
        groups = 8 if point == 388 else 2
        seeds = list(range(c['curve_seed_start'], c['curve_seed_start']+groups))
        grid = {(s, p, seat) for s in seeds for p in c['opponent_profiles'] for seat in range(4)}
        if (protocol['seeds'] != seeds or protocol['profiles'] != c['opponent_profiles'] or protocol['seats'] != 4
                or len(raw_games) != len(grid) or {(g['seed'], g['profile'], g['seat']) for g in raw_games} != grid):
            raise ValueError('Saved actual native game grid differs')
        for game in raw_games:
            seat = game['seat']; ledger = settlement_ledger(game['mjai_log'], game['final_scores'])
            if (not game['completed'] or not game['ledger_passed'] or ledger['hands'] != game['hands']
                    or game['net_points'] != game['final_scores'][seat]-25000
                    or hand_dealin_count(game['mjai_log'], seat) != game['dealin_hands']
                    or sum(e['type'] == 'hora' and e.get('actor') == seat for e in game['mjai_log']) != game['wins']
                    or sorted(range(4), key=lambda p: (-game['final_scores'][p], p)).index(seat)+1 != game['rank']):
                raise ValueError('Actual native financial/win/rank record differs')
        actual = aggregate_complete_matches(raw_games)
        if any(row[k] != v for k, v in actual.items()): raise ValueError('Native journal differs from actual games')
        identity = bundle(fit/'training-run/checkpoints'/f'step-{point:06d}')
        spec, policy_id = policy_contract(identity, reg)
        if (protocol['checkpoint'] != identity or row['checkpoint_sha256'] != identity['identity_sha256']
                or protocol['policy'] != spec or protocol['policy_identity'] != row['policy_identity'] or row['policy_identity'] != policy_id):
            raise ValueError('Native result belongs to another checkpoint or selector')
        if point == 388 and row['curve_grid_subset'] != aggregate_complete_matches([g for g in raw_games if g['seed'] < seeds[0]+2]):
            raise ValueError('Final matching16 subset differs')
        checked_games += len(raw_games)
    evidence = None
    if final:
        evidence_path = fit.parent/'final-evidence-private.json'; evidence = read(evidence_path)
        require_final_packet(evidence, step, val, matched, full)
        captured_hashes['final-evidence-private.json'] = sha(evidence_path)
    csv_file(public/'train-curve.csv', TRAIN_FIELDS, train)
    projected = project_validation(val); csv_file(public/'validation-curve.csv', tuple(projected[0]), projected)
    operations = []
    for row in val:
        for selection, field in (('selected', 'operation_type_counts'), ('raw', 'raw_operation_type_counts')):
            for operation in OPERATIONS:
                counts = row[field][operation]; yes, no = recalls(counts)
                operations.append(dict(local_step=row['local_step'], selection=selection, operation=operation,
                    **{k: counts.get(k, 0) for k in ('tp', 'fn', 'tn', 'fp')}, accept_recall=yes, decline_recall=no))
    csv_file(public/'operation-validation.csv', tuple(operations[0]), operations)
    csv_file(public/'evaluation-matched16.csv', ('local_step', *GAME_FIELDS), matched)
    csv_file(public/'evaluation-final64.csv', ('local_step', *GAME_FIELDS), full)
    plot(public, train, projected, matched, step, final)
    for p, h in reg['inputs'].items():
        if sha(p) != h: raise ValueError('Frozen input changed during export')
    proof = dict(method='real_calls13_minimal_curves_v1', registered_sha256=sha(registered),
        exporter_source_sha256=sha(__file__), captured_journal_sha256=captured_hashes,
        actual_train_through=step, actual_validation_steps=[r['local_step'] for r in val],
        actual_matched16_steps=[r['local_step'] for r in matched], actual_final64_steps=[r['local_step'] for r in full],
        checked_cached_validation_menus=len(records)*len(val), checked_actual_native_ledgers=checked_games,
        registered_final_step=388, loss_method=METHOD, original_action_ce_separate=True, supervised_reward=None,
        game_reward_definition='terminal_raw_net_points_div8000_v1', independent_full_final_audit_checked=final,
        stage2_advancement=evidence['stage2_advancement'] if evidence else None,
        snapshots_private=True, publication_ready_not_automatically_published=True,
        no_raw_cases_predictions_gradients_sources_per_update_or_weights_in_public=True,
        no_new_model_inference_games_updates_smoothing_or_synthetic_observations=True,
        goal_achieved=False, output_files={p.name: sha(p) for p in public.iterdir()})
    write(public/'provenance.json', proof); print(json.dumps(proof), flush=True)


def plot(public, train, val, games, through, final):
    import matplotlib
    matplotlib.use('Agg'); matplotlib.rcParams['svg.fonttype'] = 'none'
    matplotlib.rcParams['svg.hashsalt'] = 'jev-calls13-curves-v1'
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13, 12)); axes = list(axes.flat)
    for key, label, style in (('loss', 'Actual objective', '-'), ('ce', 'Original action CE', '--')):
        axes[0].plot([r['local_step'] for r in train], [r[key] for r in train], style, linewidth=.8, label=label)
        axes[1].plot([r['local_step'] for r in val], [r[key] for r in val], style, marker='o', label=label)
    axes[0].set_title('Training: observed updates only'); axes[0].set_ylabel('Weighted loss / CE')
    axes[1].set_title('Held-out loss: actual validation points'); axes[1].set_ylabel('Weighted loss / CE')
    for key, label in (('selected_pooled_RI_accept', 'Selected RI accept'), ('selected_pooled_RI_decline', 'Selected RI decline')):
        axes[2].plot([r['local_step'] for r in val], [r[key] for r in val], marker='o', label=label)
    axes[2].set_title('RI source-target recall, not win probability'); axes[2].set_ylim(0, 1)
    x = [r['local_step'] for r in games]
    for key, label in (('hand_win_rate', 'Hand win'), ('dealin_hand_rate', 'Deal-in')):
        axes[3].plot(x, [r[key] for r in games], marker='o', label=label)
    axes[3].set_title('Native same16 matches: hand outcomes'); axes[3].set_ylim(0, 1)
    for key, label in (('top2_rate', 'Top two'), ('first_rate', 'First'), ('fourth_rate', 'Fourth')):
        axes[4].plot(x, [r[key] for r in games], marker='o', label=label)
    axes[4].set_title('Native same16 matches: terminal rank'); axes[4].set_ylim(0, 1)
    axes[5].plot(x, [r['net_points'] for r in games], marker='o', label='Observed terminal net points')
    axes[5].axhline(0, color='gray', linestyle=':', linewidth=.7)
    axes[5].set_title('Native same16 matches: net points and reward'); axes[5].set_ylabel('Net points/match')
    axes[5].secondary_yaxis('right', functions=(lambda x: x/8000, lambda x: x*8000)).set_ylabel('Reward = net / 8,000')
    for ax in axes:
        ax.set_xlabel('Actual calls-course optimizer updates'); ax.grid(alpha=.2); ax.legend(fontsize=8)
        ax.set_xlim(0, max(through, 1))
    status = 'independently audited final' if final else 'unfinished real snapshot'
    fig.suptitle('Calls13 / fixed RI-class route: '+status, fontsize=14)
    note = f'Train through{through}; validation through{val[-1]["local_step"]}; games through{games[-1]["local_step"]}. Same16 grid; final64 separate. No smoothing.'
    if len(games) == 1: note += ' Native step0 only: post-training game results pending.'
    fig.text(.5, .008, note, ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .025, 1, .965))
    for ext in ('svg', 'png', 'pdf'):
        options = {'metadata': {'Date': None}} if ext == 'svg' else {}
        fig.savefig(public/f'calls13-curves.{ext}', dpi=170, **options)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--registered', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--through-step', type=int)
    p.add_argument('--require-final', action='store_true'); a = p.parse_args()
    export(a.registered.resolve(), a.out.resolve(), a.through_step, a.require_final)
