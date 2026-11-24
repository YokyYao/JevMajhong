"""3B validation using retained-parent losses and unchanged serving metrics."""
from collections import Counter, defaultdict
import math

from tools.curriculum_yaku_parent_loss_v1 import MASS, METHOD, COMPONENTS, NEW_SOURCES, annotate, parts
from tools.curriculum_yaku_fit_v1 import random_proxy
from tools.curriculum_progression_parent_fit_v1 import profile as replay_profile
from tools.curriculum_progression_parent_loss_v1 import MASS as REPLAY_MASS
from tools.curriculum_complete_legal_class_policy_v1 import select, METHOD as ROUTE


def profile(records, predictions, base, old_ce):
    import torch
    from jevany.data import materialize
    rows = {r['_meta']['id']: r for r in records}; saved = {p['id']: p for p in predictions}
    if (len(rows) != 24516 or len(saved) != len(predictions) or len(rows) != len(records)
            or set(rows) != set(saved)):
        raise ValueError('Complete24516 distinct validation menus required')
    replay = [r for r in records if r['questions']['action']['src'] in REPLAY_MASS]
    if len(replay) != 23688: raise ValueError('Unchanged complete progression replay required')
    inherited = replay_profile(replay, [saved[r['_meta']['id']] for r in replay], {}, old_ce)
    means = {s: dict(v, new_route_objective=0.) for s, v in inherited['source_loss_components'].items()}
    source_counts = dict(inherited['stratum_counts'])
    new_values = defaultdict(list); new_hits = defaultdict(list); new_ops = defaultdict(Counter)
    phases = defaultdict(list)
    for row in records:
        q = row['questions']['action']; src = q['src']
        if src in REPLAY_MASS: continue
        p = saved[row['_meta']['id']]; keys = p['keys']
        if (src not in NEW_SOURCES or q.get('target') is not None or len(keys) != len(set(keys))
                or set(keys) != set(q['criteria']) or len(keys) != len(p['p'])
                or any(not math.isfinite(v) or v < 0 for v in p['p']) or abs(sum(p['p'])-1) > 1e-5):
            raise ValueError('Actual new-source hard labels and valid probability menus required')
        material = annotate(materialize(row), row)['questions'][0]
        if material['keys'] != keys: raise ValueError('New validation candidate order changed')
        z = torch.tensor([math.log(max(v, 1e-30)) for v in p['p']], dtype=torch.float64)
        values = parts(z, material, 'cpu')
        new_values[src].append({k: float(v) for k, v in values.items()})
        kinds = [q['criteria'][k]['type'] for k in keys]; chosen, _ = select(kinds, p['p'])
        hit = keys[chosen] == q['label']; new_hits[src].append(hit)
        phases[row['_meta']['phase']].append(hit)
        truth, actual = q['criteria'][q['label']]['type'], kinds[chosen]
        for op in set(kinds) & {'reach', 'chi', 'pon', 'ankan', 'daiminkan', 'kakan', 'ryukyoku'}:
            outcome = 'tp' if truth == op and actual == op else 'fn' if truth == op else 'fp' if actual == op else 'tn'
            new_ops[op][outcome] += 1
    if set(new_values) != set(NEW_SOURCES): raise ValueError('All six route/control sources required')
    for src, values in new_values.items():
        means[src] = {k: sum(v[k] for v in values)/len(values) for k in COMPONENTS}
        source_counts[src] = len(values)
    if set(means) != set(MASS): raise ValueError('Complete25-source validation required')
    weighted = lambda name: sum(MASS[s]*means[s][name] for s in MASS)
    result = dict(base, weighted_class_objective=weighted('objective'), weighted_target_ce=weighted('standard_target_ce'),
        natural_target_ce=sum(source_counts[s]*means[s]['standard_target_ce'] for s in MASS)/len(records),
        signed_objective_adjustment=weighted('signed_objective_minus_standard_ce'),
        new_route_objective=weighted('new_route_objective'), source_loss_components=means,
        source_objective={s: v['objective'] for s, v in means.items()},
        stratum_ce={s: v['standard_target_ce'] for s, v in means.items()}, stratum_counts=source_counts,
        new_source_teacher_agreement={s: sum(v)/len(v) for s, v in new_hits.items()},
        stratum_hard_accuracy=dict(inherited['stratum_hard_accuracy'], **{s: sum(v)/len(v) for s, v in new_hits.items()}),
        hard_target_records=inherited['hard_target_records']+sum(len(v) for v in new_hits.values()),
        MC_selected_mean_target_mass=inherited['MC_selected_mean_target_mass'],
        MC_selected_target_mass_is_not_policy_value=True,
        new_source_teacher_agreement_is_not_yaku_or_game_mastery=True,
        new_phase_teacher_agreement={s: dict(records=len(v), agreement=sum(v)/len(v)) for s, v in phases.items()},
        new_operation_type_counts={s: {k: v[k] for k in ('tp', 'fn', 'tn', 'fp')} for s, v in new_ops.items()},
        replay_profile=inherited, loss_method=METHOD, selection_method=ROUTE,
        cached_hard_agreement_is_not_match_strength=True)
    for key in ('basic_replay', 'raw_basic_replay', 'basic_replay_source_quality',
                'raw_basic_source_quality', 'riichi_type_counts', 'riichi_by_source',
                'original_calls_operation_type_counts', 'original_calls_operation_type_counts_by_source',
                'original_calls_loss_diagnostic_uses_old_objective'):
        result[key] = inherited[key]
    if abs(result['weighted_class_objective']-result['weighted_target_ce']-result['signed_objective_adjustment']) > 1e-10:
        raise ValueError('Actual signed validation objective accounting differs')
    return result
