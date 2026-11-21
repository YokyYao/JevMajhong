"""Complete multistratum epochs for later courses, without tiny tail updates."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace


def course_order(categories,rng,batch_size,min_tail_fraction=.75):
    if not categories or batch_size<2:raise ValueError('Nonempty data and valid accumulation required')
    pools={}
    for i,key in enumerate(categories):pools.setdefault(key,[]).append(i)
    updates=math.ceil(len(categories)/batch_size)
    sizes=[min(batch_size,len(categories)-i*batch_size) for i in range(updates)]
    if sizes[-1]/batch_size<min_tail_fraction:
        raise ValueError('Tiny final update; select a count-only accumulation before registration')
    if min(sizes)<len(pools) or min(map(len,pools.values()))<updates:
        raise ValueError('Distinct stratum coverage cannot fit without copies')
    bins=[[] for _ in sizes];remaining=[]
    for key in sorted(pools):
        values=pools[key];rng.shuffle(values)
        destinations=list(range(updates));rng.shuffle(destinations)
        for target,index in zip(destinations,values[:updates]):bins[target].append(index)
        remaining.extend(values[updates:])
    rng.shuffle(remaining);cursor=0
    for target,size in zip(bins,sizes):
        take=size-len(target);target.extend(remaining[cursor:cursor+take]);cursor+=take;rng.shuffle(target)
    if cursor!=len(remaining):raise ValueError('Unallocated original records')
    full=bins[:-1] if sizes[-1]<batch_size else bins;rng.shuffle(full)
    batches=full+[bins[-1]] if sizes[-1]<batch_size else full
    result=[i for batch in batches for i in batch]
    if sorted(result)!=list(range(len(categories))):raise ValueError('Epoch permutation differs')
    if any({categories[i] for i in batch}!=set(pools) for batch in batches):raise ValueError('A source stratum is missing')
    return result


class CourseEpochRandom(random.Random):
    def __init__(self,seed,categories,batch_size,receipt_dir):
        super().__init__(seed);self.categories=list(categories);self.batch_size=batch_size
        self.receipt_dir=Path(receipt_dir);self.epochs=0
    def shuffle(self,order):
        if sorted(order)!=list(range(len(self.categories))):raise ValueError('Original full index set required')
        rng=random.Random();rng.setstate(self.getstate())
        order[:]=course_order(self.categories,rng,self.batch_size);self.setstate(rng.getstate());self.epochs+=1
        counts=[dict(Counter(self.categories[i] for i in order[start:start+self.batch_size]))
            for start in range(0,len(order),self.batch_size)]
        self.receipt_dir.mkdir(parents=True,exist_ok=True)
        proof=dict(epoch=self.epochs,records=len(order),updates=len(counts),each_record_exactly_once=True,copies_added=0,
            all_updates_have_all_declared_strata=True,min_tail_fraction=.75,
            source_counts_by_update=counts,order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest())
        with (self.receipt_dir/f'epoch-{self.epochs:02d}.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n')


def trainer_course_random_proxy(original,seed,categories,batch_size,receipt_dir):
    created=[]
    def factory(value=None):
        if type(value) is int and value==seed:
            if created:raise ValueError('Unexpected duplicate epoch RNG')
            rng=CourseEpochRandom(value,categories,batch_size,receipt_dir);created.append(rng);return rng
        return original.Random(value)
    return SimpleNamespace(seed=original.seed,Random=factory),created
