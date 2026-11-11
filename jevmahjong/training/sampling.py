"""Action/direction weights with explicit missing coverage and capped reuse."""
from collections import Counter
import math

OPERATIONS = ('chi', 'pon', 'ankan', 'daiminkan', 'kakan')


def stratum(row):
    category = row['category']
    kinds = [v['type'] for v in row['public_request']['questions']['action']['criteria'].values()]
    alternative, baseline = kinds[row['action']], kinds[row['teacher_action']]
    if category not in OPERATIONS:
        category = alternative if alternative in OPERATIONS else baseline if baseline in OPERATIONS else 'ordinary'
    delta = row['fast_advantage']
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


def weights(rows, mix, max_weight=4.):
    if max_weight <= 0 or not math.isfinite(max_weight) or abs(sum(mix.values()) - 1.) > 1e-8:
        raise ValueError('Positive cap and normalized action mix required')
    support = Counter(stratum(r) for r in rows)
    directions = {k: [d for c, d in support if c == k] for k in mix}
    raw = []
    for row in rows:
        category, direction = stratum(row)
        ds = directions.get(category, [])
        raw.append(mix.get(category, 0.) / len(ds) / support[category, direction] if ds else 0.)
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
    return values, report
