"""Outcome-blind coverage, paired uncertainty targets and fixed source mixing."""
from collections import defaultdict
import math
import random


def category(payload,rank):
    q=payload['questions']['action'];kinds={v['type'] for v in q['criteria'].values()}
    if rank<=2:return 'hold' if 'hora' not in kinds else None
    if rank==3:return 'third' if 'hora' not in kinds else None
    if 'hora' in kinds:
        from .public_policy import win_pass_allowed
        return 'win' if win_pass_allowed(payload) else None
    if kinds & {'ankan','kakan','daiminkan'}:return 'kan'
    if 'pon' in kinds:return 'pon'
    if 'chi' in kinds:return 'chi'
    return 'discard'


def public_cell(payload,kind,rank):
    s=payload['state'];seat=s['seat'];order=sorted(range(4),key=lambda p:(-s['scores'][p],p))
    gap=s['scores'][order[1]]-s['scores'][seat]
    band='le3000' if gap<=3000 else '3001_8000' if gap<=8000 else 'gt8000'
    types={v['type'] for v in payload['questions']['action']['criteria'].values()}
    kan=','.join(sorted(types & {'ankan','kakan','daiminkan'})) or '-'
    return ':'.join((kind,band,'dealer' if seat==s['dealer'] else 'nondealer',
        'threat' if any(p!=seat for p in s.get('riichi',[])) else 'quiet',kan))


def select_manifest(rows,quotas):
    """No terminal outcomes accepted; distinct source game per selected menu."""
    result=[]
    for split,wanted in quotas.items():
        pools=defaultdict(lambda:defaultdict(list))
        for r in rows:
            if r['split']==split:pools[r['category']][r['cell']].append(r)
        for cells in pools.values():
            for values in cells.values():values.sort(key=lambda r:r['selection_key'])
        used=set()
        # Fill scarce kinds first so common discards cannot consume their games.
        order=sorted(wanted,key=lambda k:(sum(len(v) for v in pools[k].values())/wanted[k],k))
        for kind in order:
            selected=[];cells=pools[kind]
            while len(selected)<wanted[kind]:
                changed=False
                for cell in sorted(cells):
                    while cells[cell] and cells[cell][0]['seed'] in used:cells[cell].pop(0)
                    if cells[cell] and len(selected)<wanted[kind]:
                        row=cells[cell].pop(0);selected.append(row);used.add(row['seed']);changed=True
                if not changed:raise ValueError(f'Insufficient distinct public {split}:{kind} coverage')
            result.extend(selected)
    if len({r['seed'] for r in result})!=len(result):raise ValueError('Source groups overlap')
    return result


def world_design(index):
    if type(index) is not int or not 0<=index<64:raise ValueError('Expected one of64 fixed worlds')
    return dict(block=index//32,policy=index%2,opponents=(index//2)%2)


def seeded_future(env,wall_seed,future_seed):
    from jevmahjong.training.futures import sample_future
    from jevmahjong.training.collector import call_observation_digest
    sampled=sample_future(env,wall_seed,'unseen')
    before=call_observation_digest(sampled.clone().get_observations())
    sampled.seed=future_seed
    if call_observation_digest(sampled.clone().get_observations())!=before:
        raise ValueError('Resampling future-hand RNG changed current observations')
    return sampled


def reliable_target(prior,utilities,eligible,temperature=.1,z=1.96,raw_fraction=.5,exploration=.1):
    """Shrink uncertain paired advantages; never discard rare menus or one-hot a winner.

    utilities maps each action to64 paired binary final-top2 outcomes. The SE
    shrinkage is a heuristic, not a confidence guarantee or expert annotation.
    """
    keys=list(prior)
    if (set(utilities)!=set(keys) or not eligible or not set(eligible)<=set(keys)
        or any(len(v)!=64 or any(x not in (0,1) for x in v) for v in utilities.values())
        or any(not math.isfinite(p) or p<0 for p in prior.values()) or abs(sum(prior.values())-1)>1e-5):
        raise ValueError('Complete normalized prior and paired64 binary outcomes required')
    if not (temperature>0 and z>=0 and 0<raw_fraction<=1 and 0<=exploration<1):raise ValueError('Invalid uncertainty recipe')
    mass=sum(prior[k] for k in eligible)
    p={k:(prior[k]/mass if mass>0 else 1/len(eligible)) for k in eligible}
    baseline=max(eligible,key=p.get);means={k:sum(v)/64 for k,v in utilities.items()};details={};adv={}
    for k in eligible:
        d=[a-b for a,b in zip(utilities[k],utilities[baseline])];mean=sum(d)/64
        se=math.sqrt(sum((x-mean)**2 for x in d)/(63*64))
        shrunk=math.copysign(max(0.,abs(mean)-z*se),mean)
        adv[k]=raw_fraction*mean+(1-raw_fraction)*shrunk
        details[k]=dict(paired_advantage=mean,se=se,shrunk_advantage=shrunk,
            block_advantages=[sum(d[i:i+32])/32 for i in (0,32)],
            negative_evidence_retained=mean<0,uncertain_evidence_not_filtered=True)
    maximum=max(adv.values())
    weights={k:((1-exploration)*p[k]+exploration/len(eligible))*math.exp((adv[k]-maximum)/temperature) for k in eligible}
    q={k:weights[k]/sum(weights.values()) for k in eligible}
    fallback=sum(q[k]*means[k] for k in eligible)<sum(p[k]*means[k] for k in eligible)-1e-12
    if fallback:q=p
    return {k:q.get(k,0.) for k in keys},dict(baseline_action=baseline,action_means=means,
        paired=details,finite_expectation_fallback=fallback,rare_successes_kept=True,
        current_hands_are_not_a_public_posterior=True)


def balanced_batches(records,quotas,steps,seed=2026111920):
    pools=defaultdict(list)
    for i,row in enumerate(records):pools[row['questions']['action']['src']].append(i)
    if set(pools)!=set(quotas) or any(v<1 for v in quotas.values()):raise ValueError('Every training source needs a batch quota')
    rng=random.Random(seed);cursor={k:0 for k in pools}
    for values in pools.values():rng.shuffle(values)
    batches=[]
    for _ in range(steps):
        names=list(quotas);rng.shuffle(names);batch=[]
        for name in names:
            for _ in range(quotas[name]):
                if cursor[name]==len(pools[name]):rng.shuffle(pools[name]);cursor[name]=0
                batch.append(pools[name][cursor[name]]);cursor[name]+=1
        batches.append(batch)
    return batches
