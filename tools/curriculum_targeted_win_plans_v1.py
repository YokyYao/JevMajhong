"""Fixed fresh canonical families for targeted late win/special decisions."""
from collections import Counter
import hashlib,json,random
from tools.generate_curriculum_yaku_scenarios_v1 import canonical_shape

KINDS=('win_reach','win_ankan','win_kakan')
METHOD='fresh_canonical_full_wall_delayed_native_win_special_sources_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def target(kind,rng):
    for _ in range(2048):
        pair=rng.randrange(34)
        if kind=='win_reach':
            triplet=rng.randrange(34);values=[triplet]*3
            for _ in range(3):
                start=rng.randrange(3)*9+rng.randrange(7);values+=list(range(start,start+3))
        else:
            quad=rng.randrange(3)*9+rng.randrange(7);values=[quad]*4+[quad+1,quad+2]
            start=rng.randrange(3)*9+rng.randrange(7);values+=list(range(start,start+3))
            if kind=='win_ankan':
                start=rng.randrange(3)*9+rng.randrange(7);values+=list(range(start,start+3))
            elif kind=='win_kakan':values += [rng.randrange(31,34)]*3
            else:raise ValueError('Unknown special-operation source')
        values += [pair,pair];counts=Counter(values)
        if (len(values)==14 and max(counts.values())<=(3 if kind=='win_reach' else 4)
            and (kind=='win_reach' or [t for t,n in counts.items() if n==4]==[quad])):
            return values
    raise ValueError('Finite target shape generator exhausted')


def plans(config):
    if config['families_per_kind']!={'train':8,'validation':4} or config['seed_start']!=2026611000:
        raise ValueError('Fixed8/4 fresh canonical families per kind required')
    result=[];seen=set()
    for index,kind in enumerate(KINDS):
        counts=Counter()
        for attempt in range(512):
            seed=config['seed_start']+index*10000+attempt
            values=target(kind,random.Random(seed));family=digest(canonical_shape(values))
            split='validation' if int(family[:16],16)%5==0 else 'train'
            if family in seen or counts[split]>=config['families_per_kind'][split]:continue
            seen.add(family);counts[split]+=1
            result.append(dict(kind=kind,target14=values,seed=seed,canonical_shape_family=family,
                group_id='targeted-win-v1:'+family,split=split))
            if dict(counts)==config['families_per_kind']:break
        if dict(counts)!=config['families_per_kind']:raise ValueError('Insufficient canonical source families')
    if len(result)!=36 or len({p['canonical_shape_family'] for p in result})!=36:
        raise ValueError('Exactly36 fresh unique canonical families required')
    return result

