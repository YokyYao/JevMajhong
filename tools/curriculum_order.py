"""Without-replacement epoch ordering with riichi in every accumulation group."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace


def balanced_order(categories, rng, batch_size):
    """Preserve every record once while mixing both riichi labels in each update."""
    if batch_size < 3 or not categories:
        raise ValueError('At least three records per update and nonempty data required')
    pools = {key: [] for key in ('plain', 'riichi_yes', 'riichi_no')}
    for index, key in enumerate(categories):
        if key not in pools:
            raise ValueError('Unknown loss-only category')
        pools[key].append(index)
    n = math.ceil(len(categories) / batch_size)
    if min(len(pools[k]) for k in ('riichi_yes', 'riichi_no')) < n:
        raise ValueError('Insufficient distinct riichi examples to cover every update without copies')
    sizes = [min(batch_size, len(categories)-i*batch_size) for i in range(n)]
    if min(sizes) < 2:
        raise ValueError('Final partial accumulation group cannot contain both labels')
    bins = [[] for _ in range(n)]
    for key in ('riichi_yes', 'riichi_no'):
        rng.shuffle(pools[key])
        offset = rng.randrange(n)
        for i, index in enumerate(pools[key]):
            target = (offset+i) % n
            if len(bins[target]) >= sizes[target]:
                raise ValueError('Rare-label allocation exceeds group capacity')
            bins[target].append(index)
    rng.shuffle(pools['plain'])
    cursor = 0
    for target, size in zip(bins, sizes):
        take = size-len(target)
        target.extend(pools['plain'][cursor:cursor+take])
        cursor += take
        rng.shuffle(target)
    if cursor != len(pools['plain']):
        raise ValueError('Not all ordinary records were allocated')
    # Keep the short final group last; all other groups have exactly batch_size records.
    full = bins[:-1] if sizes[-1] < batch_size else bins
    rng.shuffle(full)
    batches = full+[bins[-1]] if sizes[-1] < batch_size else full
    order = [index for batch in batches for index in batch]
    if sorted(order) != list(range(len(categories))):
        raise ValueError('Epoch order is not a complete unique permutation')
    if any(not {'riichi_yes', 'riichi_no'} <= {categories[i] for i in batch} for batch in batches):
        raise ValueError('An optimizer group lacks a riichi label class')
    return order


class BalancedEpochRandom(random.Random):
    def __init__(self, seed, categories, batch_size, receipt_dir=None):
        super().__init__(seed)
        self.categories = list(categories)
        self.batch_size = batch_size
        self.receipt_dir = Path(receipt_dir) if receipt_dir is not None else None
        self.epochs = 0

    def shuffle(self, order):
        if sorted(order) != list(range(len(self.categories))):
            raise ValueError('Balanced scheduler must receive the full original index set')
        # A separate RNG avoids recursively calling this shuffle during bin construction.
        local = random.Random()
        local.setstate(self.getstate())
        order[:] = balanced_order(self.categories, local, self.batch_size)
        self.setstate(local.getstate())
        self.epochs += 1
        if self.receipt_dir is not None:
            self.receipt_dir.mkdir(parents=True, exist_ok=True)
            counts = [dict(Counter(self.categories[i] for i in order[start:start+self.batch_size]))
                      for start in range(0, len(order), self.batch_size)]
            proof = dict(epoch=self.epochs, records=len(order), updates=len(counts),
                         order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),
                         each_record_exactly_once=True, copies_added=0,
                         all_updates_have_both_riichi_labels=True, source_counts_by_update=counts)
            with (self.receipt_dir/f'epoch-{self.epochs:02d}.json').open('x') as f:
                json.dump(proof, f, indent=2)
                f.write('\n')


def trainer_random_proxy(original, seed, categories, batch_size, receipt_dir):
    """Only replace the pinned trainer's epoch RNG; augmentation RNGs stay ordinary."""
    created = []

    def factory(value=None):
        if type(value) is int and value == seed:
            if created:
                raise ValueError('Unexpected second epoch RNG; inspect the pinned trainer')
            rng = BalancedEpochRandom(value, categories, batch_size, receipt_dir)
            created.append(rng)
            return rng
        return original.Random(value)

    return SimpleNamespace(seed=original.seed, Random=factory), created
