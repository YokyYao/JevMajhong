"""Prospective within-block RI pairing; never modifies an active epoch order.

Preserves every original example, mixing-block membership and update size.
Pairing can increase empty-source updates: this is a dynamics hypothesis,
not an increase in source coverage or an established training improvement.
"""
from collections import Counter

PREFIXES = ('curriculum2:basic:legacy', 'curriculum2:basic:teacher')


def update_coverage(order, categories, accumulation, prefixes=PREFIXES):
    counts = {source: Counter() for source in prefixes}
    for start in range(0, len(order), accumulation):
        actual = Counter(categories[index] for index in order[start:start + accumulation])
        for source in prefixes:
            yes = actual[source + ':riichi_yes']
            no = actual[source + ':riichi_no']
            result = counts[source]
            result['updates'] += 1
            result['positive_records'] += yes
            result['negative_records'] += no
            result['updates_with_both_classes'] += (yes > 0 and no > 0)
            result['updates_without_RI'] += (yes + no == 0)
            result['updates_without_positive'] += (yes == 0)
            result['updates_without_negative'] += (no == 0)
    return {source: dict(value) for source, value in counts.items()}


def pair_within_blocks(order, categories, rng, block_size=511, accumulation=73, prefixes=PREFIXES):
    if (not categories or any(not isinstance(c, str) for c in categories)
            or any(type(index) is not int for index in order)
            or sorted(order) != list(range(len(categories)))
            or type(block_size) is not int or type(accumulation) is not int
            or accumulation < 2 or block_size < accumulation or block_size % accumulation
            or not prefixes or len(set(prefixes)) != len(prefixes)
            or any(not isinstance(prefix, str) for prefix in prefixes)):
        raise ValueError('Complete distinct original indices and aligned valid block/update sizes required')
    required = {prefix + ':riichi_' + kind for prefix in prefixes for kind in ('yes', 'no')}
    if not required <= set(categories):
        raise ValueError('Both actual hard RI categories required for every declared source')
    result = []
    optimal_pairs = Counter()
    for start in range(0, len(order), block_size):
        block = list(order[start:start + block_size])
        sizes = [min(accumulation, len(block) - offset) for offset in range(0, len(block), accumulation)]
        bins = [[] for _ in sizes]
        selected = set()
        for source in prefixes:
            yes = [index for index in block if categories[index] == source + ':riichi_yes']
            no = [index for index in block if categories[index] == source + ':riichi_no']
            destinations = list(range(len(bins)))
            rng.shuffle(destinations)
            # A tail update must be able to hold every reserved pair. Reject
            # impossible input rather than silently drop a source or a record.
            available = [index for index in destinations if sizes[index] - len(bins[index]) >= 2]
            target = min(len(yes), len(no), len(bins))
            if len(available) < target:
                raise ValueError('Update capacity cannot hold all prospective source pairs')
            optimal_pairs[source] += target
            for destination, positive, negative in zip(available[:target], yes[:target], no[:target]):
                bins[destination].extend((positive, negative))
                selected.update((positive, negative))
        remaining = [index for index in block if index not in selected]
        cursor = 0
        for destination, size in zip(bins, sizes):
            take = size - len(destination)
            destination.extend(remaining[cursor:cursor + take])
            cursor += take
            rng.shuffle(destination)
        if cursor != len(remaining):
            raise ValueError('Prospective pairing failed to preserve original block membership')
        flattened = [index for destination in bins for index in destination]
        if sorted(flattened) != sorted(block):
            raise ValueError('Prospective pairing lost or repeated original records')
        result.extend(flattened)
    before = update_coverage(order, categories, accumulation, prefixes)
    after = update_coverage(result, categories, accumulation, prefixes)
    if any(after[source]['updates_with_both_classes'] != optimal_pairs[source] for source in prefixes):
        raise ValueError('Maximum possible within-block pair count was not attained')
    if sorted(result) != list(range(len(categories))):
        raise ValueError('Prospective epoch is not the original complete permutation')
    return result, dict(method='prospective_RI_pairs_within_original_mixing_blocks_v1',
        records=len(result), copies_added=0, each_original_once=True,
        mixing_block_size=block_size, accumulation=accumulation,
        original_block_membership_preserved=True, update_sizes_preserved=True,
        maximum_within_block_pairs=dict(optimal_pairs), before=before, after=after,
        source_weights_and_total_exposure_unchanged=True,
        empty_source_updates_can_increase=True,
        model_improvement_not_established=True, training_started=False)
