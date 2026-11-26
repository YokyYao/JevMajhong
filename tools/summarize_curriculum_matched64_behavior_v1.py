"""Describe completed matched64 native events and financial components.

Uses the actual audited 3B endpoint, its parent and frozen v8. This is cached
event analysis, not new games, inference, action-opportunity rates or causality.
"""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_transfer import read, sha
from tools.summarize_curriculum_match_behavior_v1 import COMPONENTS, CALLS

REGISTRATION = ROOT/'runs/curriculum-yaku-early-v2/fit-plan/registered.json'


def describe(matches, expected, ledger_fn):
    counts, components = Counter(), Counter()
    for match in matches:
        if not match['completed'] or not match['ledger_passed']:
            raise ValueError('Only complete native audited matches may be described')
        seat = match['seat']; log = match['mjai_log']
        ledger = ledger_fn(log, match['final_scores']); personal = ledger['components'][seat]
        if (ledger['hands'] != match['hands'] or set(personal)-set(COMPONENTS)
                or sum(personal.values()) != match['net_points']):
            raise ValueError('Actual per-player financial components do not close')
        components.update(personal); counts['hands'] += ledger['hands']
        opened, wins = False, 0
        for event in log:
            kind = event['type']
            if kind == 'start_kyoku': opened = False
            if event.get('actor') != seat: continue
            if kind in CALLS:
                counts[kind] += 1
                if kind != 'ankan': opened = True
            elif kind == 'reach_accepted': counts['riichi'] += 1
            elif kind == 'hora':
                wins += 1; counts['open_win_events' if opened else 'closed_win_events'] += 1
        if wins != match['wins']:
            raise ValueError('Actual native win events differ from the completed match')
        counts['wins'] += wins
    n = len(matches)
    result = dict(matches=n, played_hands=counts['hands'], own_win_events=counts['wins'],
        open_win_events=counts['open_win_events'], closed_win_events=counts['closed_win_events'],
        mean_own_win_settlement=components['win_income']/counts['wins'] if counts['wins'] else None)
    for key in COMPONENTS: result[key+'_per_match'] = components[key]/n
    result['net_points_per_match'] = sum(components.values())/n
    if not math.isclose(result['net_points_per_match'], expected['net_points'], rel_tol=0, abs_tol=1e-10):
        raise ValueError('The original full64 net points must be reproduced exactly')
    for key in ('riichi', *CALLS):
        result[key+'_events'] = counts[key]
        result[key+'_per100_played_hands'] = 100*counts[key]/counts['hands']
    return result


def summarize(out):
    allocation = resource(); out = Path(out).resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):
        raise ValueError('Use a new private output directory')
    from tools.curriculum_placement_early_parent_v1 import parent_artifacts
    parent_artifacts(REGISTRATION)
    reg = read(REGISTRATION); config = reg['curriculum']; fit = ROOT/config['out']
    audit_path = fit.parent/'final-evidence-private.json'; audit = read(audit_path)
    if (audit['status'] != 'yaku2500_final_evidence_audit_passed'
            or audit['optimizer_steps'] != 2500 or audit['actual_records_seen'] != 182494):
        raise ValueError('The actual completed endpoint audit is required')
    seeds = range(config['curve_seed_start'], config['curve_seed_start']+8)
    if any(2026300000 <= seed < 2026300032 for seed in seeds):
        raise ValueError('Never consume reserved final records as development')
    cells = {(seed, profile, seat) for seed in seeds
        for profile in config['opponent_profiles'] for seat in range(4)}
    if len(cells) != 64: raise ValueError('Exactly the existing two-profile/four-seat grid required')
    sys.path.insert(0, reg['source'])
    from jevmahjong.core.scoring import settlement_ledger
    files = {str(path):sha(path) for path in (Path(__file__), REGISTRATION, audit_path,
        ROOT/'tools/summarize_curriculum_match_behavior_v1.py',
        Path(sys.modules[settlement_ledger.__module__].__file__).resolve())}
    specs = [('reference-parent', '3A parent', config['parent_identity']),
        ('reference-v8', 'Frozen v8', config['reference_identity']),
        ('step-002500', '3B 2500', audit['final_identity'])]
    results = []
    for folder, display, identity in specs:
        root = fit/'evaluations'/folder
        expected = read(root/'summary.json')
        if (expected['matches'] != 64 or expected['seed_groups'] != 8
                or expected['checkpoint_sha256'] != identity):
            raise ValueError('Actual matched64 aggregate/model identity differs')
        path = root/'matches-private.jsonl'; before = sha(path)
        with path.open() as stream:
            matches = [json.loads(line) for line in stream if line.strip()]
        if (len(matches) != 64 or {(m['seed'], m['profile'], m['seat']) for m in matches} != cells):
            raise ValueError('Every policy must use exactly the registered same64 cells')
        result = dict(policy=display, **describe(matches, expected, settlement_ledger))
        if sha(path) != before: raise ValueError('Completed native records changed during analysis')
        results.append(result)
        for artifact in (path, root/'summary.json', root/'protocol.json'):
            files[str(artifact)] = sha(artifact)
    out.mkdir(parents=True)
    with (out/'matched64-behavior.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    lines = ['# Matched64 native behavior and financial components', '',
        'Each column uses the same64 completed East matches: eight consumed logical seeds, two profiles, four seats. '
        'This analysis reads existing results; it generates no games, predictions or training updates.', '',
        '## Point components per match', '',
        'Own-win income includes honba and awarded riichi sticks. Mean settlement per own-win event is not '
        'intrinsic han/fu value. The six signed components close exactly to the original net points.', '',
        '| Metric | '+' | '.join(r['policy'] for r in results)+' |',
        '|---|'+'|'.join('---:' for _ in results)+'|']
    for key in (*[name+'_per_match' for name in COMPONENTS], 'net_points_per_match', 'mean_own_win_settlement'):
        values = ['n/a' if r[key] is None else f'{r[key]:.3f}' for r in results]
        lines.append('| '+key+' | '+' | '.join(values)+' |')
    lines += ['', '## Realized action events', '',
        'These are counts and frequencies per100 played hands, not acceptance rates conditional on a legal '
        'action opportunity. The pooled win counts differ from headline per-match average hand-win rates. '
        'Wins may include the actual available-win shortcut.', '',
        '| Metric | '+' | '.join(r['policy'] for r in results)+' |',
        '|---|'+'|'.join('---:' for _ in results)+'|']
    for key in ('played_hands', 'own_win_events', 'open_win_events', 'closed_win_events',
                *[op+'_events' for op in ('riichi', *CALLS)],
                *[op+'_per100_played_hands' for op in ('riichi', *CALLS)]):
        lines.append('| '+key+' | '+' | '.join(f'{r[key]:.3f}' for r in results)+' |')
    lines += ['', 'Descriptive consumed development evidence only. These tables do not identify the cause '
        'of a policy difference, prove which calls were profitable, or establish rare-yaku mastery or independent '
        'parity with v8. The registered placement recipe and unopened final confirmation are unchanged.', '']
    (out/'matched64-behavior.md').write_text('\n'.join(lines))
    write(out/'summary-private.json', dict(status='actual_matched64_native_financial_and_action_events_exported',
        rows=results, matches_per_policy=64, consumed_seed_groups=8,
        all_native_ledgers_closed=True, original_net_points_exact=True,
        new_games=0, model_loaded=False, student_updates=0, allocation=allocation))
    for path in out.iterdir(): files[str(path)] = sha(path)
    write(out/'completion-binding-private.json', dict(files=files))
    print(json.dumps(dict(status='actual_matched64_behavior_exported', rows=results)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); summarize(args.out)
