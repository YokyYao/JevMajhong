"""Public-condition fourth-place windows reused from existing source games."""
from collections import Counter
import copy
import json
from pathlib import Path

from .collect import digest, write
from .context import public_context
from .coverage import public_stratum
from .prepare import checksum
from .ledger import hand_score_history

SCHEMA = 'reused-fourth-win32-v1'
LIMITS = {'train': 64, 'validation': 16}


def fourth_cell(row):
    payload = row['public_request']
    stratum = public_stratum(payload)
    if stratum != row['coverage_stratum'] or stratum is None or stratum['placement'] != 4:
        raise ValueError('Only verified public fourth-place win windows are eligible')
    context = public_context(payload['state'])
    if context['placement'] != 4 or not context['late_match_window']:
        raise ValueError('Current public placement differs')
    gap = context['gap_to_third']
    band = 'le3000' if gap <= 3000 else '3001_8000' if gap <= 8000 else 'gt8000'
    quote = payload['state']['endgame_context']['current_win_settlement']
    escapes = quote['minimum_baseline_scenario']['rank_after'] < 4
    return {'style': stratum['style'], 'gap_to_third': gap, 'gap_to_third_band': band,
            'minimum_single_winner_baseline_escapes4': escapes,
            'cell': ':'.join((row['source_profile'], stratum['style'], band, 'escapes4' if escapes else 'still4'))}


def select_reused_rows(rows, limits):
    if limits != LIMITS:
        raise ValueError('The registered manifest is 64 training and 16 validation groups')
    groups = {s: {r['seed'] for r in rows if r['split'] == s} for s in limits}
    if groups['train'] & groups['validation'] or any(r['split'] not in limits for r in rows):
        raise ValueError('Original source groups cross train/validation')
    result = []
    for split in ('train', 'validation'):
        cells = {}
        for row in rows:
            if row['split'] == split:
                cells.setdefault(row['reuse_fourth']['cell'], []).append(row)
        for cell in cells:
            cells[cell].sort(key=lambda r: r['reuse_selection_key'])
        used = set()
        while len(used) < limits[split]:
            added = False
            for cell in sorted(cells):
                while cells[cell] and cells[cell][0]['seed'] in used:
                    cells[cell].pop(0)
                if cells[cell] and len(used) < limits[split]:
                    row = cells[cell].pop(0)
                    result.append(row); used.add(row['seed']); added = True
            if not added:
                break
    return result


def make_reused_manifest(config):
    if (config.get('schema') != SCHEMA or config.get('future_worlds') != 32
            or config.get('fit_registered') is not False or config.get('manifest_limits') != LIMITS):
        raise ValueError('Only the registered public 32-world data stage is enabled')
    out, coverage = Path(config['out']), Path(config['coverage_out'])
    prior_path = Path(config['exclude_manifest'])
    prior = json.loads(prior_path.read_text())
    if digest(prior) != config['exclude_manifest_positions_sha256']:
        raise ValueError('Previously labeled source manifest changed')
    if len(prior) != 44:
        raise ValueError('Original excluded source manifest must contain 44 positions')
    used = {r['seed'] for r in prior}
    rows = []; input_hashes = {}
    for profile in ('standard', 'dama'):
        path = coverage / profile / 'selected-positions-private.json'
        input_hashes[profile] = checksum(path)
        if input_hashes[profile] != config['coverage_source_sha256'][profile]:
            raise ValueError('Existing coverage rows changed')
        for original in json.loads(path.read_text()):
            seed = original['seed']
            expected_split = 'validation' if (seed - 2026116900) % 4 == 3 else 'train'
            if (not 2026116900 <= seed <= 2026117411 or original['split'] != expected_split
                    or original['group_id'] != 'endgame-native:' + str(seed)):
                raise ValueError('Original native source split/identity differs')
            if seed in used or original['coverage_stratum']['placement'] != 4:
                continue
            row = copy.deepcopy(original); row['source_profile'] = profile
            row['reuse_fourth'] = fourth_cell(row)
            row['reuse_selection_key'] = digest([SCHEMA, seed, profile, row['request_sha256']])
            rows.append(row)
    selected = select_reused_rows(rows, config['manifest_limits'])
    counts = Counter(r['split'] for r in selected)
    if dict(counts) != LIMITS or len({r['seed'] for r in selected}) != 80:
        raise ValueError('Insufficient distinct isolated sources for the registered 64/16 manifest')
    strata = {}
    for split in LIMITS:
        subset = [r for r in selected if r['split'] == split]
        n_closed = sum(r['reuse_fourth']['style'] == 'closed_nonriichi' for r in subset)
        n_escape = sum(r['reuse_fourth']['minimum_single_winner_baseline_escapes4'] for r in subset)
        minima = (12, 12) if split == 'train' else (4, 4)
        if n_closed < minima[0] or min(n_escape, len(subset) - n_escape) < minima[1]:
            raise ValueError('Public hand-style or baseline-escape coverage is insufficient')
        strata[split] = {'positions': len(subset), 'closed_nonriichi': n_closed,
            'baseline_escapes4': n_escape, 'baseline_still4': len(subset) - n_escape,
            'cells': dict(Counter(r['reuse_fourth']['cell'] for r in subset))}
    manifest = out / 'selected-positions-private.json'
    if manifest.exists() and json.loads(manifest.read_text()) != selected:
        raise ValueError('Existing reused manifest differs; inspect before changing it')
    (out / 'sources').mkdir(parents=True, exist_ok=True)
    source_hashes = {}
    for row in selected:
        seed = row['seed']
        source = json.loads((coverage / row['source_profile'] / 'sources' / f'{seed}.json').read_text())
        if source['seed'] != seed or source['profile'] != row['source_profile']:
            raise ValueError('Copied source profile differs')
        history = hand_score_history(source['mjai_log'])
        if history['final_scores'] != source['final_scores']:
            raise ValueError('Source score ledger does not reconcile')
        # Derived history is only a consistency check, never a sampling input.
        source['hand_score_history'] = history
        destination = out / 'sources' / f'{seed}.json'
        if destination.exists() and json.loads(destination.read_text()) != source:
            raise ValueError('Existing copied source differs')
        write(destination, source)
        source_hashes[str(seed)] = checksum(destination)
    write(manifest, selected)
    result = {'status': 'reused_fourth_manifest_fixed_before_terminal_labels',
        'positions': len(selected), 'splits': dict(counts), 'strata': strata,
        'positions_sha256': digest(selected), 'source_sha256': source_hashes,
        'coverage_source_sha256': input_hashes, 'future_worlds': 32,
        'selected_by_future_outcomes': False, 'source_group_overlap': 0,
        'one_position_per_original_seed_across_profiles': True, 'optimizer_updates': 0}
    write(out / 'manifest-status-private.json', result)
    return result
