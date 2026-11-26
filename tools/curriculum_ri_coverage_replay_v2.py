"""Bounded full-coverage RI replay when a stratum exceeds the update count."""
from collections import Counter
import hashlib
import math
import random
from tools.curriculum_progression_parent_loss_v1 import MASS

RI = tuple(sorted(s for s in MASS if s.endswith((':riichi_yes', ':riichi_no'))))
METHOD = 'all_originals_at_least_one_per_RI_source_each_update_inverse_repeat_v2'


def build(categories, baseline, seed, batch=73, max_repeat=8, max_weight=16.):
    n = len(categories); counts = Counter(categories)
    if (set(counts) != set(MASS) or len(RI) != 4 or sorted(baseline) != list(range(n))
            or type(seed) is not int or type(batch) is not int or batch <= len(RI)
            or type(max_repeat) is not int or max_repeat < 1):
        raise ValueError('Full nineteen-source permutation and bounded full-batch recipe required')
    updates = math.ceil(n/batch)
    while n+sum(max(0, updates-counts[s]) for s in RI) > updates*batch:
        updates += 1
    if any(math.ceil(updates/counts[s]) > max_repeat for s in RI):
        raise ValueError('Required complete coverage exceeds the original repeat cap')
    rng = random.Random(seed); groups = [[] for _ in range(updates)]
    for source in RI:
        pool = [i for i in baseline if categories[i] == source]
        local = random.Random(int.from_bytes(hashlib.sha256(f'{seed}:{source}'.encode()).digest()[:8], 'big'))
        stream = []
        while len(stream) < max(updates, len(pool)):
            cycle = list(pool); local.shuffle(cycle)
            stream.extend(cycle[:max(updates, len(pool))-len(stream)])
        for position in range(updates): groups[position].append(stream[position])
        extra = len(stream)-updates
        for j,index in enumerate(stream[updates:]):
            position = ((2*j+1)*updates)//(2*extra)
            groups[position].append(index)
    if any(len(group) > batch for group in groups):
        raise ValueError('Evenly distributed extra RI rows exceed a batch; register another schedule')
    nonri = [i for i in baseline if categories[i] not in RI]
    padding = updates*batch-sum(map(len, groups))-len(nonri)
    if padding < 0: raise ValueError('Insufficient physical slots for every non-RI original')
    duplicated = []
    while len(duplicated) < padding:
        cycle = list(nonri); rng.shuffle(cycle)
        duplicated.extend(cycle[:padding-len(duplicated)])
    expanded = []; proof = []; inserted = 0
    for position,index in enumerate(nonri):
        expanded.append(index); proof.append(index)
        target = (position+1)*padding//len(nonri)
        while inserted < target:
            expanded.append(duplicated[inserted]); inserted += 1
    order = []; cursor = 0
    for group in groups:
        width = batch-len(group); group.extend(expanded[cursor:cursor+width]); cursor += width
        if len(group) != batch: raise ValueError('Every physical update must be complete')
        rng.shuffle(group); order.extend(group)
    repeats = Counter(order); exposures = len(order)
    if (cursor != len(expanded) or exposures != updates*batch or set(repeats) != set(range(n))
            or max(repeats.values()) > max_repeat or proof != nonri):
        raise ValueError('Every original must survive; repeat cap and non-RI sequence must hold')
    weights = {str(i):exposures*MASS[categories[i]]/(counts[categories[i]]*repeats[i]) for i in range(n)}
    error = max(abs(weights[str(i)]*repeats[i]/exposures-MASS[categories[i]]/counts[categories[i]]) for i in range(n))
    if (error > 1e-12 or any(not math.isfinite(v) or not 0 < v <= max_weight for v in weights.values())):
        raise ValueError('Exact per-original source mass and bounded weights required')
    by_update = [Counter(categories[i] for i in order[start:start+batch]) for start in range(0,exposures,batch)]
    if any(min(c[s] for s in RI) < 1 for c in by_update):
        raise ValueError('Each actual update must retain all four RI sources')
    return dict(indices=order,weights_by_original_index=weights,nonRI_original_subsequence=proof,
        summary=dict(method=METHOD,unique_TRAIN_rows=n,physical_exposures=exposures,
            optimizer_updates=updates,batch=batch,additional_exposures=exposures-n,
            nonRI_inserted_repeats=padding,max_repeat=max(repeats.values()),
            max_example_weight=max(weights.values()),maximum_original_row_mass_error=error,
            unique_source_counts=dict(counts),physical_source_counts=dict(Counter(categories[i] for i in order)),
            RI_per_update_range={s:[min(c[s] for c in by_update),max(c[s] for c in by_update)] for s in RI},
            nonRI_original_order_preserved=True,per_original_empirical_objective_preserved=True,
            optimization_trajectory_equivalence_claimed=False))
