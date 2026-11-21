"""Describe expert and masked-MC operation coverage without inventing negatives."""
from collections import Counter, defaultdict
import statistics

OPERATIONS = {'chi', 'pon', 'ankan', 'daiminkan', 'kakan', 'reach'}


def coverage(records, confidence=None):
    counts = defaultdict(Counter)
    groups = defaultdict(lambda: defaultdict(set))
    masses = defaultdict(list)
    confidence_mass = defaultdict(list)
    for row in records:
        q = row['questions']['action']
        meta = row['_meta']
        offered = {v['type'] for v in q['criteria'].values()} & OPERATIONS
        for op in offered:
            counts[op]['offered'] += 1
            if meta['label_type'] == 'expert_replay':
                selected = q['criteria'][q['label']]['type'] == op
                kind = 'observed_selected' if selected else 'observed_other_action'
                counts[op][kind] += 1
                groups[op][kind].add(meta['group_id'])
                continue
            if meta['label_type'] != 'paired_future_masked':
                raise ValueError('Unknown label source')
            target = q['target']
            support = {k for k, v in target.items() if v > 0}
            own = {k for k in support if q['criteria'][k]['type'] == op}
            other = support-own
            if not own:
                counts[op]['operation_unlabelled_not_negative'] += 1
                continue
            if not other:
                counts[op]['other_operations_unlabelled_not_positive_contrast'] += 1
                continue
            counts[op]['labelled_operation_vs_other_contrast'] += 1
            groups[op]['labelled_contrast'].add(meta['group_id'])
            mass = sum(target[k] for k in own)/sum(target.values())
            masses[op].append(mass)
            counts[op]['operation_mass_above_half' if mass > .5+1e-12 else 'operation_mass_below_half' if mass < .5-1e-12 else 'operation_mass_tied'] += 1
            chosen = max(target, key=target.get)
            counts[op]['target_argmax_operation' if chosen in own else 'target_argmax_other'] += 1
            if confidence is not None:
                c = confidence[meta['id']]
                if c['phase'] != meta['phase'] or c['group_id'] != meta['group_id'] or c['target_reproduction_max_error'] >= 1e-10:
                    raise ValueError('Recovered confidence differs from the current record')
                confidence_mass[op].append(c['target_argmax_conditional_best_mass'])
                direction = 'operation' if chosen in own else 'other'
                counts[op][direction+'_argmax_noncontrol_positive_nominal_lower_bound'] += c['noncontrol_positive_lower_bound']
    result = {}
    for op, values in counts.items():
        result[op] = dict(values, source_groups={k: len(v) for k, v in groups[op].items()})
        if masses[op]:
            result[op]['median_operation_target_mass_given_labelled_contrast'] = statistics.median(masses[op])
        if confidence_mass[op]:
            result[op]['median_conditional_best_mass_given_labelled_contrast'] = statistics.median(confidence_mass[op])
    return result
