"""Action/direction weights with explicit missing coverage and capped reuse."""
from collections import Counter
import math

OPERATIONS = ('chi', 'pon', 'ankan', 'daiminkan', 'kakan')


def risk_context(row):
    from riichienv import MeldType
    state = row['public_request'].get('state', {})
    seat = state.get('seat')
    if not set(state.get('riichi') or []) - {seat}: return 'no_threat'
    if any(m['type'] != int(MeldType.Ankan) for m in state['melds'][seat]): return 'open_threat'
    return 'other_threat'


def stratum(row):
    category = row['category'].removesuffix('_open_threat').removesuffix('_threat')
    kinds = [v['type'] for v in row['public_request']['questions']['action']['criteria'].values()]
    alternative, baseline = kinds[row['action']], kinds[row['teacher_action']]
    if category not in OPERATIONS:
        category = alternative if alternative in OPERATIONS else baseline if baseline in OPERATIONS else 'ordinary'
    delta = row.get('training_advantage', row['fast_advantage'])
    if abs(delta) <= 1e-12:
        direction = 'neutral'
    elif category == 'ordinary':
        direction = 'positive' if delta > 0 else 'negative'
    elif alternative == category and baseline != category:
        direction = 'positive' if delta > 0 else 'negative'
    elif baseline == category and alternative != category:
        direction = 'negative' if delta > 0 else 'positive'
    else:
        direction = 'within_action'
    return category, direction


def weights(rows, mix, max_weight=4., context_mix=None):
    if max_weight <= 0 or not math.isfinite(max_weight) or abs(sum(mix.values()) - 1.) > 1e-8:
        raise ValueError('Positive cap and normalized action mix required')
    support = Counter(stratum(r) for r in rows)
    directions = {k: [d for c, d in support if c == k] for k in mix}
    contexts = Counter((stratum(r), risk_context(r)) for r in rows) if context_mix else None
    if context_mix is not None and (set(context_mix) != {'open_threat', 'other_threat', 'no_threat'}
            or any(not math.isfinite(v) or v < 0 for v in context_mix.values())
            or abs(sum(context_mix.values()) - 1.) > 1e-8):
        raise ValueError('Normalized nonnegative public risk context mix required')
    raw = []
    for row in rows:
        category, direction = stratum(row)
        ds = directions.get(category, [])
        value = mix.get(category, 0.) / len(ds) / support[category, direction] if ds else 0.
        if contexts and ds:
            context = risk_context(row)
            mass = sum(context_mix[c] for c in context_mix if contexts[(category, direction), c])
            value = (mix.get(category, 0.) / len(ds) * context_mix[context] / mass
                / contexts[(category, direction), context]) if mass else 0.
        raw.append(value)
    total = sum(raw)
    if total <= 0:
        raise ValueError('No supported requested action stratum')
    # Capping reduces a rare example's influence rather than copying it to fill quotas.
    values = [min(max_weight, x * len(rows) / total) for x in raw]
    report = {'support': {c + '/' + d: n for (c, d), n in sorted(support.items())},
        'missing_action_directions': [c + '/' + d for c in OPERATIONS for d in ('positive', 'negative') if not support[c, d]],
        'target_action_mix': mix, 'max_example_weight': max_weight,
        'effective_action_mass': {c: sum(w for r, w in zip(rows, values) if stratum(r)[0] == c) / sum(values) for c in mix},
        'copies_added': 0}
    if context_mix is not None:
        report.update(target_context_mix=context_mix,
            context_support={c+'/'+d+'/'+context:n for ((c,d),context),n in sorted(contexts.items())},
            effective_context_mass={c:sum(w for r,w in zip(rows,values) if risk_context(r)==c)/sum(values)
                for c in context_mix})
    return values, report
