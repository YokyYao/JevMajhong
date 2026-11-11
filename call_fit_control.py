"""TRAIN-only sampling for a bounded learning-capability control experiment."""
from collections import Counter
import random

QUOTAS = {'closed_chi': 128, 'closed_pon': 128, 'open_chi': 32,
          'open_pon': 32, 'none': 128, 'ordinary': 64}


def stratum(record):
    if record['_meta']['phase'] != 'reaction':
        return 'ordinary'
    q = record['questions']['action']
    option = q['criteria'][q['label']]
    kind = option['type']
    if kind == 'none':
        return 'none'
    if kind not in ('chi', 'pon'):
        raise ValueError('Unsupported reaction label')
    closed = option.get('call_features', {}).get('loses_closed_status')
    if type(closed) is not bool:
        raise ValueError('Known closed status required')
    return ('closed_' if closed else 'open_') + kind


def choose_control(records, arm, seed):
    ids = [r['_meta']['id'] for r in records]
    if len(ids) != len(set(ids)) or any(r['_meta']['split'] != 'train' or
            r['_meta']['view'] != 'native' for r in records):
        raise ValueError('Unique original TRAIN records required')
    buckets = {k: [] for k in QUOTAS}
    for record in records:
        buckets[stratum(record)].append(record)
    rng = random.Random(seed)
    selected = []
    if arm == 'balanced':
        for key, count in QUOTAS.items():
            if len(buckets[key]) < count:
                raise ValueError('Insufficient ' + key)
            selected.extend(rng.sample(buckets[key], count))
    elif arm == 'natural':
        reactions = [r for r in records if r['_meta']['phase'] == 'reaction']
        if len(reactions) < 448 or len(buckets['ordinary']) < 64:
            raise ValueError('Insufficient natural control pool')
        selected = rng.sample(reactions, 448) + rng.sample(buckets['ordinary'], 64)
    else:
        raise ValueError('Unknown control arm')
    rng.shuffle(selected)
    return selected, dict(Counter(stratum(r) for r in selected))
