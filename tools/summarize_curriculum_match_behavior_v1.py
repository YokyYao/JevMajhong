"""Describe actual native match events and point components, without inference.

Consume an already completed same16 report inside a finite GPU allocation.
Financial decomposition and event frequencies are descriptive, not causal
evidence, action-opportunity rates, yaku mastery or independent strength.
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

COMPONENTS = ('win_income', 'riichi_deposits', 'ron_payments', 'tsumo_payments',
              'draw_settlement', 'final_unclaimed_pool')
CALLS = ('chi', 'pon', 'daiminkan', 'ankan', 'kakan')


def summarize(comparison, out):
    allocation = resource()
    comparison, out = Path(comparison).resolve(), Path(out).resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):
        raise ValueError('Use a new private runs/ output directory')
    binding_path = comparison/'completion-binding-private.json'
    bound = read(binding_path)['files']
    files = {str(Path(__file__)): sha(__file__), str(binding_path): sha(binding_path)}
    def checked(path):
        path = Path(path).resolve()
        digest = sha(path)
        if bound.get(str(path)) != digest:
            raise ValueError('A completed original report input changed: '+str(path))
        files[str(path)] = digest
        return read(path)
    done = checked(comparison/'summary-private.json')
    points = checked(comparison/'projected-curve16-points-private.json')
    if (done['status'] != 'actual_same16_development_points_and_full_validation_summarized'
            or not done['same_grid_and_execution_policy'] or done['matches_per_point'] != 16
            or done['consumed_seed_groups'] != 2 or done['new_games'] != 0):
        raise ValueError('Actual complete same16 development evidence required')
    registrations = [Path(path) for path in bound if path.endswith('/fit-plan/registered.json')]
    if len(registrations) != 1:
        raise ValueError('One actual original registration required')
    reg = checked(registrations[0]); config = reg['curriculum']
    fit = ROOT/config['out']
    seeds = range(config['curve_seed_start'], config['curve_seed_start']+2)
    if any(2026300000 <= seed < 2026300032 for seed in seeds):
        raise ValueError('Never read reserved confirmation as development')
    cells = {(seed, profile, seat) for seed in seeds
             for profile in config['opponent_profiles'] for seat in range(4)}
    if len(cells) != 16:
        raise ValueError('Exactly the registered two-profile, four-seat development grid required')
    sys.path.insert(0, reg['source'])
    from jevmahjong.core.scoring import settlement_ledger
    scoring_source = Path(sys.modules[settlement_ledger.__module__].__file__).resolve()
    files[str(scoring_source)] = sha(scoring_source)
    rows = []
    for step in done['actual_steps']:
        folder = fit/'evaluations'/f'step-{step:06d}'
        path = folder/'matches-private.jsonl'
        digest = sha(path)
        if bound.get(str(path)) != digest:
            raise ValueError('Only the actual source-bound completed match bytes may be read')
        files[str(path)] = digest
        all_matches = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if checked(folder/'status.json') != dict(status='complete', games=len(all_matches)):
            raise ValueError('Wait for the complete original16/64 result')
        matches = [m for m in all_matches if (m['seed'], m['profile'], m['seat']) in cells]
        if len(matches) != 16 or {(m['seed'], m['profile'], m['seat']) for m in matches} != cells:
            raise ValueError('Every point must use exactly the same16 registered cells')
        counts, components = Counter(), Counter()
        for match in matches:
            if not match['completed'] or not match['ledger_passed']:
                raise ValueError('No partial or unaudited matches')
            log, seat = match['mjai_log'], match['seat']
            ledger = settlement_ledger(log, match['final_scores'])
            personal = ledger['components'][seat]
            if (ledger['hands'] != match['hands'] or set(personal)-set(COMPONENTS)
                    or sum(personal.values()) != match['net_points']):
                raise ValueError('Every player point component must close to the actual net score')
            components.update(personal)
            counts['hands'] += ledger['hands']
            own_wins, opened = 0, False
            for event in log:
                kind = event['type']
                if kind == 'start_kyoku': opened = False
                if event.get('actor') != seat: continue
                if kind in CALLS:
                    counts[kind] += 1
                    if kind != 'ankan': opened = True
                elif kind == 'reach_accepted': counts['riichi'] += 1
                elif kind == 'hora':
                    own_wins += 1
                    counts['open_win_events' if opened else 'closed_win_events'] += 1
            if own_wins != match['wins']:
                raise ValueError('Actual native own-win events disagree with original counts')
            counts['wins'] += own_wins
        row = dict(step=step, matches=16, played_hands=counts['hands'],
            own_win_events=counts['wins'], open_win_events=counts['open_win_events'],
            closed_win_events=counts['closed_win_events'],
            mean_own_win_settlement=(components['win_income']/counts['wins'] if counts['wins'] else None))
        for component in COMPONENTS:
            row[component+'_per_match'] = components[component]/16
        row['net_points_per_match'] = sum(components.values())/16
        if not math.isclose(row['net_points_per_match'], points[str(step)]['net_points'], rel_tol=0, abs_tol=1e-10):
            raise ValueError('The point decomposition must reproduce the published same16 net score')
        for operation in ('riichi', *CALLS):
            row[operation+'_events'] = counts[operation]
            row[operation+'_per100_played_hands'] = 100*counts[operation]/counts['hands']
        rows.append(row)
    out.mkdir(parents=True, exist_ok=False)
    with (out/'match-behavior.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['# Observed native match behavior and point components', '',
        'All columns reuse the same16 complete development East matches (two consumed seeds, two profiles, four seats). '
        'No model inference, new games, optimization or checkpoint selection was performed.', '',
        '## Exact financial decomposition', '',
        'Every component below is points per match. Own-win settlements include honba and awarded riichi sticks; '
        'they are not intrinsic hand value or han/fu. Deposits and opponent payments retain their negative signs. '
        'The six components sum exactly to the previously reported net points.', '',
        '| Metric | '+' | '.join('Step '+str(row['step']) for row in rows)+' |',
        '| --- | '+' | '.join('---:' for row in rows)+' |']
    for key, label in [*((c+'_per_match', c) for c in COMPONENTS),
            ('net_points_per_match', 'Net points'), ('mean_own_win_settlement', 'Mean own-win settlement per win event')]:
        values = ['n/a' if row[key] is None else f'{row[key]:.2f}' for row in rows]
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    lines += ['', '## Realized events', '',
        'These are event counts and pooled frequencies per100 played hands, not probabilities of accepting a legal '
        'operation. Hand counts vary with continuations. Native win-event counts here use pooled totals, while '
        'the headline hand-win rate averages per-match rates. Both open and closed win events may include forced-win shortcuts.', '',
        '| Metric | '+' | '.join('Step '+str(row['step']) for row in rows)+' |',
        '| --- | '+' | '.join('---:' for row in rows)+' |']
    for key in ('played_hands', 'own_win_events', 'open_win_events', 'closed_win_events',
                *(op+'_events' for op in ('riichi', *CALLS)),
                *(op+'_per100_played_hands' for op in ('riichi', *CALLS))):
        lines.append('| '+key+' | '+' | '.join(f'{row[key]:.2f}' for row in rows)+' |')
    lines += ['', 'These correlated development observations describe outcomes and actions, not their causes. '
        'They do not prove stable improvement, profitable calls, rare-yaku mastery or superiority to frozen v8. '
        'The original2500-update course and unopened final confirmation remain unchanged. Raw records stay private.', '']
    (out/'match-behavior.md').write_text('\n'.join(lines))
    write(out/'summary-private.json', dict(status='actual_same16_native_event_and_point_components_exported',
        actual_steps=done['actual_steps'], native_ledgers_closed=True, all_components_match_original_net=True,
        matches_per_point=16, consumed_seed_groups=2, new_games=0, model_loaded=False, student_updates=0,
        causal_or_independent_strength_claim=False, allocation=allocation))
    for path in out.iterdir(): files[str(path)] = sha(path)
    write(out/'completion-binding-private.json', dict(files=files))
    print(json.dumps(dict(status='actual_match_behavior_exported', rows=rows)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); summarize(args.comparison, args.out)
