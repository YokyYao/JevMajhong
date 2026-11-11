"""Summarize already-opened TRAIN/validation fit, without changing a policy."""
from collections import Counter, defaultdict
import math


def reaction_records(records, split):
    if split not in ('train', 'validation'):
        raise ValueError('Sealed test is not a diagnostic source')
    ids = [r['_meta']['id'] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate diagnostic request')
    if any(r['_meta']['split'] != split or r['_meta']['view'] != 'native' for r in records):
        raise ValueError('Diagnostic must use original TRAIN or already-opened validation')
    return [r for r in records if r['_meta']['phase'] == 'reaction']


def summarize_fit(records, rows):
    from targeted_reaction_sampling import public_stratum
    lookup = {r['_meta']['id']: r for r in records}
    if len(lookup) != len(records) or len(rows) != len(records):
        raise ValueError('Diagnostic coverage differs')
    seen = set()
    groups = defaultdict(lambda: {'n': 0, 'exact': 0, 'nll_sum': 0.,
                                  'gold': Counter(), 'predicted': Counter(), 'false_calls': 0})
    for row in rows:
        identity = row['id']
        if identity in seen or identity not in lookup:
            raise ValueError('Duplicate or foreign prediction')
        seen.add(identity)
        record = lookup[identity]
        q = record['questions']['action']
        keys = list(q['criteria'])
        label = keys.index(q['label'])
        if row['keys'] != keys or row['label'] != label:
            raise ValueError('Option/label alignment differs')
        probabilities = row['p']
        if (len(probabilities) != len(keys) or
                any(not math.isfinite(p) or p < 0 for p in probabilities) or
                not math.isclose(sum(probabilities), 1., abs_tol=1e-5)):
            raise ValueError('Invalid diagnostic probabilities')
        pred = max(range(len(keys)), key=probabilities.__getitem__)
        gold_option = q['criteria'][q['label']]
        gold, predicted = gold_option['type'], q['criteria'][keys[pred]]['type']
        if gold not in ('chi', 'pon', 'none') or predicted not in ('chi', 'pon', 'none'):
            raise ValueError('Foreign reaction action')
        names = ['all', 'gold/' + gold, 'public/' + public_stratum(record)]
        if gold in ('chi', 'pon'):
            closed = gold_option.get('call_features', {}).get('loses_closed_status')
            if type(closed) is not bool:
                raise ValueError('Known gold-call closed status required')
            names.append(('closed_' if closed else 'open_') + gold)
        for name in names:
            g = groups[name]
            g['n'] += 1
            g['exact'] += pred == label
            g['nll_sum'] -= math.log(max(probabilities[label], 1e-30))
            g['gold'][gold] += 1
            g['predicted'][predicted] += 1
            g['false_calls'] += gold == 'none' and predicted != 'none'
    return {name: {**g, 'gold': dict(g['gold']), 'predicted': dict(g['predicted']),
                   'nll': g['nll_sum'] / g['n'], 'exact_rate': g['exact'] / g['n']}
            for name, g in groups.items()}
