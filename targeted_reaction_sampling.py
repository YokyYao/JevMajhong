"""Choose unseen requests by public opportunity and threat, never expert labels."""
from collections import Counter
import random

STRATA = ('closed_chi_no_threat', 'closed_chi_threat', 'other_reaction', 'ordinary')


def public_stratum(record):
    if record['_meta']['phase'] != 'reaction':
        return 'ordinary'
    eligible = [option for option in record['questions']['action']['criteria'].values()
                if option['type'] == 'chi'
                and option.get('call_features', {}).get('loses_closed_status') is True]
    if not eligible:
        return 'other_reaction'
    threats = {option['call_features']['riichi_threat_count'] for option in eligible}
    if len(threats) != 1 or next(iter(threats)) not in (0, 1, 2, 3):
        raise ValueError('Known consistent public threat count required')
    return 'closed_chi_threat' if next(iter(threats)) else 'closed_chi_no_threat'


def choose_requests(records, excluded_ids, quotas, seed):
    if set(quotas) != set(STRATA) or any(type(v) is not int or v < 1 for v in quotas.values()):
        raise ValueError('Explicit positive quota for every public stratum required')
    buckets = {key: [] for key in STRATA}
    ids = set()
    excluded = set()
    for record in records:
        meta = record['_meta']
        if meta['id'] in ids or meta['split'] != 'train' or meta['view'] != 'native':
            raise ValueError('Unique original native TRAIN requests required')
        ids.add(meta['id'])
        if meta['id'] in excluded_ids:
            excluded.add(meta['id'])
            continue
        buckets[public_stratum(record)].append(record)
    if excluded != set(excluded_ids):
        raise ValueError('All registered prior exposure must exist in the source pool')
    rng = random.Random(seed)
    selected = []
    for key in STRATA:
        if len(buckets[key]) < quotas[key]:
            raise ValueError('Insufficient new public stratum: ' + key)
        rng.shuffle(buckets[key])
        selected.extend(buckets[key][:quotas[key]])
    rng.shuffle(selected)
    return selected, {'remaining_pool': sum(map(len, buckets.values())),
                      'available': {key: len(value) for key, value in buckets.items()},
                      'selected': dict(Counter(public_stratum(r) for r in selected)),
                      'excluded_requests': len(excluded)}
