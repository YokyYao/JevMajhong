"""Explicit retained-parent placement validation; full57-source data required."""
from collections import defaultdict
import math

from tools import curriculum_placement_metrics_v1 as previous
from tools import curriculum_yaku_parent_fit_v1 as replay
from tools import curriculum_placement_partition_v1 as partition
from tools.curriculum_placement_exposures_v1 import REPEAT_FIELD
from tools.curriculum_placement_parent_loss_v2 import (
    METHOD, COMPONENTS, annotate, parts, summarize, masses, NEW_SOURCES,
)

selected_key = previous.selected_key


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


def combine(base, inherited, fresh, record_count):
    old = inherited['source_loss_components']; new = fresh['source_loss_components']
    if set(old) != set(previous.REPLAY_MASS) or set(new) != set(NEW_SOURCES):
        raise ValueError('All25 retained replay and32 original terminal source profiles required')
    components = {s: dict(v, terminal_soft_objective=0.) for s, v in old.items()}
    components.update(new)
    if any(set(v) != set(COMPONENTS) for v in components.values()):
        raise ValueError('Every source requires the complete ten-component profile')
    counts = {**inherited['stratum_counts'], **fresh['source_counts']}
    if set(counts) != set(masses()) or sum(counts.values()) != record_count:
        raise ValueError('Complete original unique57-source validation counts required')
    mass = {s: float(v) for s, v in masses().items()}
    weighted = lambda key: sum(mass[s]*components[s][key] for s in mass)
    return dict(base, weighted_class_objective=weighted('objective'),
        weighted_target_ce=weighted('standard_target_ce'),
        signed_objective_adjustment=weighted('signed_objective_minus_standard_ce'),
        new_route_objective=weighted('new_route_objective'),
        terminal_soft_objective=weighted('terminal_soft_objective'),
        natural_target_ce=sum(counts[s]*components[s]['standard_target_ce'] for s in counts)/record_count,
        source_loss_components=components, stratum_counts=counts,
        stratum_ce={s: v['standard_target_ce'] for s, v in components.items()},
        replay_profile=inherited, new_placement_profile=fresh, loss_method=METHOD,
        inherited_teacher_agreement_is_not_current_placement_optimality=True,
        late_instruction_or_win_selector_change_prevents_parent_game_cache_reuse=True,
        validation_loss_is_not_policy_strength=True)


def profile(records, predictions, base, old_ce):
    from jevany.data import materialize
    saved = {p['id']: p for p in predictions}; ids = {r['_meta']['id'] for r in records}
    if (len(ids) != len(records) or len(saved) != len(predictions) or ids != set(saved)
            or set(r['questions']['action']['src'] for r in records) != set(masses())
            or any(r['_meta']['split'] != 'validation' or REPEAT_FIELD in r['_meta'] for r in records)):
        raise ValueError('Full distinct57-source validation with original source membership required')
    old_rows = [r for r in records if r['questions']['action']['src'] in previous.REPLAY_MASS]
    new_rows = [r for r in records if r['questions']['action']['src'] in NEW_SOURCES]
    inherited = replay.profile(old_rows, [saved[r['_meta']['id']] for r in old_rows], {}, old_ce)
    fresh = new_profile(new_rows, [saved[r['_meta']['id']] for r in new_rows], materialize)
    return combine(base, inherited, fresh, len(records))


def partition_profile(records, predictions, base, old_ce, manifest):
    views = partition.partition_views(records, manifest, 'validation')
    result = profile(views, predictions, base, old_ce)
    result.update(validation_current_membership_verified=True,
        inherited_source_split_tag_not_current_partition=True,
        inherited_validation_source_split_counts=manifest['partitions']['validation']['legacy_split_counts'],
        current_membership_method=partition.METHOD)
    return result

