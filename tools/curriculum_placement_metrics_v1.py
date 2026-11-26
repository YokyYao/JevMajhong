"""Placement loss journals and unique-validation diagnostics, not win rates.

Masked CE cannot measure untested alternatives. Report actual full-menu
selection on unlabelled candidates separately instead of treating low CE as
evidence of top-two improvement.
"""
from collections import defaultdict
import math

from tools.curriculum_placement_mix_v1 import NEW_SOURCES, REPLAY_MASS, masses
from tools.curriculum_placement_exposures_v1 import REPEAT_FIELD
from tools.curriculum_placement_objective_v1 import COMPONENTS, METHOD, annotate, parts
from tools.curriculum_complete_legal_class_policy_v1 import select as ordinary_select
from tools.curriculum_placement_full_win_policy_v1 import select as win_select


def summarize(pending, weights):
    if (not pending or set(weights) != set(masses())
            or any(not math.isfinite(v) or not 0 < v <= 16 for v in weights.values())):
        raise ValueError('Actual examples and complete57-source bounded positive weights required')
    grouped = defaultdict(list)
    for row in pending:
        if row['source'] not in weights or any(not math.isfinite(row[k]) for k in COMPONENTS):
            raise ValueError('Actual finite loss components and registered source required')
        grouped[row['source']].append(row)
    means = {s: {k: sum(r[k] for r in rows)/len(rows) for k in COMPONENTS} for s, rows in grouped.items()}
    counts = {s: len(rows) for s, rows in grouped.items()}
    average = lambda k: sum(counts[s]*weights[s]*m[k] for s, m in means.items())/len(pending)
    return dict(source_loss_components=means, source_counts=counts,
        source_objective={s: m['objective'] for s, m in means.items()},
        source_ce={s: m['standard_target_ce'] for s, m in means.items()},
        loss=average('objective'), ce=average('standard_target_ce'),
        extra_RI_class_loss=average('extra_RI_group_ce'),
        sparse_sources_per_update_expected=True)


def selected_key(question, keys, probabilities):
    kinds = [question['criteria'][key]['type'] for key in keys]
    if 'hora' in kinds:
        return win_select(dict(questions=dict(action=dict(criteria={k: question['criteria'][k] for k in keys}))),
                          dict(zip(keys, probabilities)))
    selected, _ = ordinary_select(kinds, probabilities)
    return keys[selected]


def new_profile(records, predictions, materialize):
    """Score unique held-out new rows with real serving selectors.

    `materialize` is the pinned upstream function at execution time. Loss is
    reconstructed from cached probabilities with an explicit1e-30 log floor.
    No model is loaded and no target is altered.
    """
    import torch
    rows = {r['_meta']['id']: r for r in records}
    saved = {p['id']: p for p in predictions}
    if (not rows or len(rows) != len(records) or len(saved) != len(predictions)
            or set(rows) != set(saved)
            or any(r['_meta']['split'] != 'validation' or REPEAT_FIELD in r['_meta'] for r in records)):
        raise ValueError('Complete distinct original validation rows and predictions required')
    grouped = defaultdict(list); families = defaultdict(set)
    for rid, row in rows.items():
        q = row['questions']['action']; source = q['src']; p = saved[rid]; keys = p['keys']; probabilities = p['p']
        if (set(row['questions']) != {'action'} or source not in NEW_SOURCES
                or len(keys) != len(set(keys)) or set(keys) != set(q['criteria'])
                or len(probabilities) != len(keys)
                or any(not math.isfinite(v) or v < 0 for v in probabilities)
                or abs(sum(probabilities)-1) > 1e-5):
            raise ValueError('Aligned finite original source/action probabilities required')
        encoded = annotate(materialize(row), row)['questions'][0]
        if encoded['keys'] != keys:
            raise ValueError('Cached validation option order differs from actual materializer')
        logits = torch.tensor([math.log(max(v, 1e-30)) for v in probabilities], dtype=torch.float64)
        values = {k: float(v) for k, v in parts(logits, encoded, 'cpu').items()}
        selected = selected_key(q, keys, probabilities)
        grouped[source].append(dict(**values, selected_target_mass=q['target'][selected],
            selected_unlabelled=q['target'][selected] == 0, selected_label=selected == q['label']))
        families[source].add(row['_meta']['group_id'])
    if set(grouped) != set(NEW_SOURCES):
        raise ValueError('All32 original held-out placement source pools required')
    fields = (*COMPONENTS, 'selected_target_mass', 'selected_unlabelled', 'selected_label')
    means = {s: {k: sum(r[k] for r in values)/len(values) for k in fields} for s, values in grouped.items()}
    return dict(source_loss_components={s: {k: v[k] for k in COMPONENTS} for s, v in means.items()},
        source_counts={s: len(v) for s, v in grouped.items()},
        source_distinct_families={s: len(v) for s, v in families.items()},
        selected_target_mass_by_source={s: v['selected_target_mass'] for s, v in means.items()},
        selected_unlabelled_rate_by_source={s: v['selected_unlabelled'] for s, v in means.items()},
        soft_label_agreement_by_source={s: v['selected_label'] for s, v in means.items()},
        target_mass_is_not_predicted_win_or_top2_probability=True,
        masked_loss_does_not_identify_untested_action_value=True,
        validation_log_probability_floor=1e-30, records=len(records))


def profile(records, predictions, base, old_ce):
    from jevany.data import materialize
    from tools.curriculum_yaku_fit_v1 import profile as replay_profile
    ids = {r['_meta']['id'] for r in records}; saved = {p['id']: p for p in predictions}
    if (len(ids) != len(records) or len(saved) != len(predictions) or ids != set(saved)
            or any(r['_meta']['split'] != 'validation' or REPEAT_FIELD in r['_meta'] for r in records)
            or set(r['questions']['action']['src'] for r in records) != set(masses())):
        raise ValueError('Full distinct57-source validation surface required')
    replay = [r for r in records if r['questions']['action']['src'] in REPLAY_MASS]
    new = [r for r in records if r['questions']['action']['src'] in NEW_SOURCES]
    old = replay_profile(replay, [saved[r['_meta']['id']] for r in replay], {}, old_ce)
    fresh = new_profile(new, [saved[r['_meta']['id']] for r in new], materialize)
    components = {**old['source_loss_components'], **fresh['source_loss_components']}
    counts = {**old['stratum_counts'], **fresh['source_counts']}
    mass = {s: float(v) for s, v in masses().items()}
    weighted = lambda k: sum(mass[s]*components[s][k] for s in mass)
    return dict(base, weighted_class_objective=weighted('objective'),
        weighted_target_ce=weighted('standard_target_ce'),
        extra_RI_class_loss=weighted('extra_RI_group_ce'),
        natural_target_ce=sum(counts[s]*components[s]['standard_target_ce'] for s in counts)/len(records),
        source_loss_components=components, stratum_counts=counts,
        stratum_ce={s: v['standard_target_ce'] for s, v in components.items()},
        replay_profile=old, new_placement_profile=fresh, loss_method=METHOD,
        inherited_teacher_agreement_is_not_current_placement_optimality=True,
        late_instruction_or_win_selector_change_prevents_parent_game_cache_reuse=True,
        validation_loss_is_not_policy_strength=True)
