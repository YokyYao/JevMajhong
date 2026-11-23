"""Prospective development assessment, distinct from independent v8 proof.

Compare fresh child-policy zero/final games, never parent-cache-as-zero.
Early inherited label retention excludes all changed terminal instructions.
"""
from collections import Counter, defaultdict
import math
import statistics

from tools.curriculum_endgame import in_endgame_scope
from tools.curriculum_complete_legal_class_policy_v1 import select

OPERATIONS = ('chi', 'pon', 'ankan', 'daiminkan', 'kakan')
PROFILES = ('efficiency', 'calling')
CRITERIA = dict(method='placement_final_budget_fresh64_top2_and_early_retention_v1',
    status='prospective_freeze_in_launch_before_first_student_update',
    final_candidate_only=True, intermediate_early_stopping=False,
    minimum_final64_top2_gain_over_child_policy_zero=0.,
    max_early_plain_agreement_drop_per_source=.03,
    max_early_RI_balanced_accuracy_drop_per_source=.03,
    max_mean_original_five_call_balanced_accuracy_drop=.03,
    require_complete_early_retention_coverage=True,
    conditional_fourth_escape_veto=False, net_points_is_development_veto=False,
    independent_v8_top2_and_net_points_proof_still_required=True,
    automatic_release=False)


def require_criteria(actual):
    if actual != CRITERIA:
        raise ValueError('Exact prospective criteria must be frozen before the first update')


def balanced(counts):
    yes = counts['tp']+counts['fn']; no = counts['tn']+counts['fp']
    return .5*(counts['tp']/yes+counts['tn']/no) if yes and no else None


def early_profile(records, predictions):
    saved = {p['id']: p for p in predictions}; ids = {r['_meta']['id'] for r in records}
    if len(saved) != len(predictions) or len(ids) != len(records) or ids != set(saved):
        raise ValueError('Complete distinct current validation inputs/predictions required')
    plain = defaultdict(list); riichi = defaultdict(Counter); calls = defaultdict(Counter)
    early = 0; excluded = 0
    for row in records:
        if in_endgame_scope(row['state']): excluded += 1; continue
        q = row['questions']['action']; source = q['src']
        if not source.startswith('curriculum2:'): continue
        p = saved[row['_meta']['id']]; keys = p['keys']; probability = p['p']
        if (q.get('target') is not None or len(keys) != len(set(keys)) or set(keys) != set(q['criteria'])
                or len(keys) != len(probability) or any(not math.isfinite(v) or v < 0 for v in probability)
                or abs(sum(probability)-1) > 1e-5 or q['label'] not in keys):
            raise ValueError('Aligned finite original hard-labelled early calls/basic menus required')
        kinds = [q['criteria'][key]['type'] for key in keys]
        chosen = kinds.index('hora') if 'hora' in kinds else select(kinds, probability)[0]
        truth = q['criteria'][q['label']]['type']; actual = kinds[chosen]; early += 1
        outcome = lambda operation: ('tp' if truth == operation and actual == operation else
            'fn' if truth == operation else 'fp' if actual == operation else 'tn')
        for operation in set(kinds) & set(OPERATIONS): calls[operation][outcome(operation)] += 1
        if source.startswith('curriculum2:basic:'):
            origin, stratum = source.removeprefix('curriculum2:basic:').split(':')
            if origin not in ('legacy', 'teacher') or stratum not in ('plain', 'riichi_yes', 'riichi_no'):
                raise ValueError('Explicit original basic origin/stratum required')
            if stratum == 'plain': plain[origin].append(keys[chosen] == q['label'])
            else:
                if set(kinds) != {'reach', 'dahai'}: raise ValueError('Original pure RI/discard menu required')
                riichi[origin][outcome('reach')] += 1
    counts = lambda c: {k: c[k] for k in ('tp', 'fn', 'tn', 'fp')}
    return dict(early_original_replay_records=early, late_records_excluded=excluded,
        plain_records={s: len(plain[s]) for s in ('legacy', 'teacher')},
        plain_agreement={s: statistics.mean(plain[s]) if plain[s] else None for s in ('legacy', 'teacher')},
        RI_counts={s: counts(riichi[s]) for s in ('legacy', 'teacher')},
        RI_balanced_accuracy={s: balanced(riichi[s]) for s in ('legacy', 'teacher')},
        original_call_counts={s: counts(calls[s]) for s in OPERATIONS},
        original_call_balanced_accuracy={s: balanced(calls[s]) for s in OPERATIONS},
        inherited_label_agreement_not_optimality_or_match_strength=True,
        early_forced_wins_not_counted_as_learned_win_decisions=True)


def development_matrix(rows, seed_start):
    expected = {(seed, profile, seat) for seed in range(seed_start, seed_start+8)
        for profile in PROFILES for seat in range(4)}
    saved = {}
    for row in rows:
        key = row['seed'], row['profile'], row['seat']; scores = row['final_scores']; seat = row['seat']
        if (key not in expected or key in saved or row['completed'] is not True or row['ledger_passed'] is not True
                or type(row['seed']) is not int or type(seat) is not int
                or not isinstance(scores, list) or len(scores) != 4 or any(type(v) is not int for v in scores)
                or type(row['rank']) is not int or row['rank'] not in range(1, 5)
                or row['rank'] != sorted(range(4), key=lambda i: (-scores[i], i)).index(seat)+1
                or row['net_points'] != scores[seat]-25000
                or type(row['hands']) is not int or row['hands'] <= 0
                or any(type(row[k]) is not int or not 0 <= row[k] <= row['hands'] for k in ('wins', 'dealin_hands'))):
            raise ValueError('Distinct complete actual64 development cells and financial counters required')
        saved[key] = row
    if set(saved) != expected: raise ValueError('Full fresh64 development matrix required')
    return saved


def paired_development(candidate, reference, seed_start):
    left = development_matrix(candidate, seed_start); right = development_matrix(reference, seed_start)
    functions = dict(top2_rate=lambda r: float(r['rank'] <= 2), net_points=lambda r: r['net_points'],
        first_rate=lambda r: float(r['rank'] == 1), fourth_rate=lambda r: float(r['rank'] == 4),
        hand_win_rate=lambda r: r['wins']/r['hands'], dealin_hand_rate=lambda r: r['dealin_hands']/r['hands'])
    result = {}
    for name, function in functions.items():
        delta = [statistics.mean(function(left[seed, profile, seat])-function(right[seed, profile, seat])
            for profile in PROFILES for seat in range(4)) for seed in range(seed_start, seed_start+8)]
        mean = statistics.mean(delta); radius = 2.3646242510102993*statistics.stdev(delta)/math.sqrt(8)
        result[name] = dict(gain=mean, ci95=[mean-radius, mean+radius], seed_groups=8,
            candidate=statistics.mean(function(r) for r in left.values()),
            reference=statistics.mean(function(r) for r in right.values()))
    return dict(metrics=result, consumed_development_only=True, independent_strength_or_parity_proof=False,
        intervals_are_descriptive_paired_eight_seed_t_df7=True)


def grade(first, last, comparison, criteria):
    require_criteria(criteria)
    for name in ('plain_records',):
        if first[name] != last[name]: raise ValueError('Fixed early validation denominators required')
    for name in ('RI_counts', 'original_call_counts'):
        if set(first[name]) != set(last[name]): raise ValueError('Fixed early operation scopes required')
        for source, a in first[name].items():
            b = last[name][source]
            if (a['tp']+a['fn'], a['tn']+a['fp']) != (b['tp']+b['fn'], b['tn']+b['fp']):
                raise ValueError('Actual positive/negative retention cases changed')
    gains = {}; complete = True
    for key, names in (('plain_agreement', ('legacy', 'teacher')),
                       ('RI_balanced_accuracy', ('legacy', 'teacher')),
                       ('original_call_balanced_accuracy', OPERATIONS)):
        gains[key] = {}
        for name in names:
            a,b = first[key][name], last[key][name]
            if a is None or b is None:
                complete = False; gains[key][name] = None
            elif not all(math.isfinite(v) and 0 <= v <= 1 for v in (a,b)):
                raise ValueError('Finite actual early validation rates required')
            else: gains[key][name] = b-a
    call_values = list(gains['original_call_balanced_accuracy'].values())
    mean_call = statistics.mean(call_values) if all(v is not None for v in call_values) else None
    top2 = comparison['metrics']['top2_rate']['gain']
    if (comparison['consumed_development_only'] is not True
            or comparison['independent_strength_or_parity_proof'] or not math.isfinite(top2)):
        raise ValueError('Actual consumed development comparison required')
    passed = (complete and top2 >= criteria['minimum_final64_top2_gain_over_child_policy_zero']
        and min(gains['plain_agreement'].values())+criteria['max_early_plain_agreement_drop_per_source'] >= -1e-12
        and min(gains['RI_balanced_accuracy'].values())+criteria['max_early_RI_balanced_accuracy_drop_per_source'] >= -1e-12
        and mean_call+criteria['max_mean_original_five_call_balanced_accuracy_drop'] >= -1e-12)
    return dict(passed=bool(passed), early_retention_coverage_complete=complete, early_retention_gains=gains,
        mean_original_call_balanced_gain=mean_call, final64_top2_gain=top2,
        terminal_or_conditional_fourth_escape_veto=False, net_points_development_veto=False,
        independent_v8_strength_not_established=True, task_completion_or_release_authorized=False)
