"""Explicit TRAIN-only copies and deterministic placement epoch ordering.

The original corpus remains unique. Copies carry provenance, never become
independent families, and do not alter a model-visible field or supervision.
"""
from collections import Counter
import copy
import hashlib
import json
import math

from tools.curriculum_placement_mix_v1 import METHOD, NEW_SOURCES, REPLAY_MASS, layout
from tools.curriculum_yaku_objective_v1 import epoch_order as replay_order

REPEAT_FIELD = 'placement_virtual_exposure_v1'


def expand_train(rows, plan):
    counts = Counter(); families = {}; ids = set()
    for row in rows:
        meta = row['_meta']; source = row['questions']['action']['src']; rid = meta['id']
        if (set(row['questions']) != {'action'} or meta['split'] != 'train'
                or not isinstance(rid, str) or not rid or rid in ids or REPEAT_FIELD in meta):
            raise ValueError('Distinct pristine original TRAIN records required')
        ids.add(rid); counts[source] += 1; families.setdefault(source, set()).add(meta['group_id'])
    actual = layout(dict(counts), {s: len(g) for s, g in families.items()})
    if plan != actual or plan['method'] != METHOD:
        raise ValueError('Virtual expansion must match the actual count-only proposal')
    result = list(rows)
    for row in rows:
        meta = row['_meta']; source = row['questions']['action']['src']
        repeats = plan['virtual_repeats_per_original_by_source'][source]
        for index in range(1, repeats):
            if source not in NEW_SOURCES:
                raise ValueError('Old replay may not be silently repeated')
            replica = copy.deepcopy(row)
            rid = 'placement-repeat-v1:'+hashlib.sha256(meta['id'].encode()).hexdigest()+':'+str(index)
            if rid in ids:
                raise ValueError('Virtual exposure ID collided with an existing original')
            ids.add(rid)
            replica['_meta']['id'] = rid
            replica['_meta'][REPEAT_FIELD] = dict(original_id=meta['id'], copy_index=index,
                total_exposures_per_pass=repeats, adds_independent_data=False)
            result.append(replica)
    if (len(result) != plan['virtual_exposures_per_pass'] or
            dict(Counter(r['questions']['action']['src'] for r in result)) != plan['virtual_exposures_by_source']):
        raise ValueError('Actual expanded TRAIN exposure counts differ')
    return result


def epoch_order(categories, rng, accumulation):
    """Preserve old25-source ordering, evenly insert sparse new exposures.

    Every expanded index is visited once. An original new row can have several
    explicitly identified indices. No every-update/every511-total-row coverage
    promise is made for the sparse new pools.
    """
    if type(accumulation) is not int or accumulation < 2:
        raise ValueError('Registered positive accumulation required')
    old_indices = [i for i, s in enumerate(categories) if s in REPLAY_MASS]
    new_pools = {s: [] for s in NEW_SOURCES}
    for index, source in enumerate(categories):
        if source not in REPLAY_MASS:
            if source not in new_pools:
                raise ValueError('Unknown expanded TRAIN source')
            new_pools[source].append(index)
    if any(not values for values in new_pools.values()):
        raise ValueError('Every proposed new source must have actual exposures')
    selected, old_receipt = replay_order([categories[i] for i in old_indices], rng)
    old_order = [old_indices[i] for i in selected]
    scheduled = []
    for source in sorted(new_pools):
        pool = new_pools[source]; rng.shuffle(pool)
        for index, value in enumerate(pool):
            scheduled.append(((index+.5)/len(pool), rng.random(), value))
    scheduled.sort()
    new_order = [value for _, _, value in scheduled]
    total = len(categories); new_total = len(new_order)
    positions = {(2*i+1)*total//(2*new_total): value for i, value in enumerate(new_order)}
    if len(positions) != new_total:
        raise ValueError('Distinct sparse insertion positions required')
    original = iter(old_order)
    order = [positions[i] if i in positions else next(original) for i in range(total)]
    updates = [order[i:i+accumulation] for i in range(0, total, accumulation)]
    if (sorted(order) != list(range(total)) or len(updates[-1]) < .75*accumulation):
        raise ValueError('Complete permutation without a tiny final update required')
    counts = [dict(Counter(categories[i] for i in update)) for update in updates]
    return order, dict(expanded_records=total, optimizer_updates=len(updates),
        final_update_records=len(updates[-1]), expanded_index_each_once=True,
        old_replay_original_each_once=True, old_replay_order_preserved=True,
        old_replay_25_source_blocks=old_receipt['mixing_blocks'],
        old_replay_block_size_counts_only_replay_rows=True,
        new_exposures_evenly_inserted=True,
        new_sources_each_optimizer_update_claimed=False,
        new_sources_each_511_total_records_claimed=False,
        source_exposure_counts_by_update=counts,
        order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest())


def choose_accumulation(total, preferred=73):
    """Prospective count-only choice; a later registration fixes the budget."""
    if type(total) is not int or total <= 0:
        raise ValueError('Actual positive exposure count required')
    for size in sorted(range(64, 97), key=lambda n: (abs(n-preferred), n)):
        updates = math.ceil(total/size)
        if total-(updates-1)*size >= .75*size:
            return size
    raise ValueError('No prospective accumulation avoids a tiny final update')

