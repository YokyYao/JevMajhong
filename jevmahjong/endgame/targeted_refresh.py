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
            if kind=='kan' and wanted[kind]>=3:
                for subtype in ('ankan','daiminkan','kakan'):
                    offered=[r for values in cells.values() for r in values if r['seed'] not in used
                        and subtype in {a['type'] for a in r['request']['questions']['action']['criteria'].values()}]
                    if not offered:raise ValueError(f'Insufficient public {split}:{subtype} coverage')
                    row=min(offered,key=lambda r:r['selection_key']);selected.append(row);used.add(row['seed'])
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


class FutureHandsWorld:
    """Reinitialize only NEW hand boundaries with independently seeded native deals.

    RiichiEnv0.4.11 has no writable seed on a live environment. Its supported
    reset API can preserve the next hand's public match state at a boundary.
    Keep the already completed MJAI prefix for one continuous score ledger.
    """
    def __init__(self,env,future_seed,prefix=None):
        self._env=env;self._future_seed=future_seed;self._prefix=prefix or []

    def __getattr__(self,name):return getattr(self._env,name)

    @property
    def mjai_log(self):
        log=self._env.mjai_log
        if not self._prefix:return log
        if log[0]['type']!='start_game':raise ValueError('Expected native new-hand game prefix')
        return self._prefix+log[1:]

    def clone(self):
        import copy
        return FutureHandsWorld(self._env.clone(),self._future_seed,copy.deepcopy(self._prefix))

    def step(self,actions):
        old=(self._env.round_wind,self._env.kyoku_idx,self._env.honba)
        previous_log_size=len(self.mjai_log)
        observations=self._env.step(actions)
        current=(self._env.round_wind,self._env.kyoku_idx,self._env.honba)
        if self._env.done() or current==old:return observations
        from riichienv import RiichiEnv,GameRule
        from .collect import digest
        expected=(self._env.oya,*current,self._env.riichi_sticks,list(self._env.scores()))
        log=self.mjai_log
        # Some engine transitions can expose the new marker before emitting
        # its start event. Never cut away the completed current hand.
        boundary=next((i for i in range(previous_log_size,len(log)) if log[i]['type']=='start_kyoku'),len(log))
        prefix=log[:boundary]
        seed=int(digest([self._future_seed,list(current),'next-hand-native-reset'])[:16],16)
        fresh=RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=seed)
        observations=fresh.reset(oya=self._env.oya,round_wind=current[0],honba=current[2],
                                 kyotaku=self._env.riichi_sticks,scores=list(self._env.scores()),seed=seed)
        actual=(fresh.oya,fresh.round_wind,fresh.kyoku_idx,fresh.honba,fresh.riichi_sticks,list(fresh.scores()))
        if actual!=expected:raise ValueError('Fresh future-hand reset changed public match state')
        self._prefix=prefix;self._env=fresh
        return observations


def seeded_future(env,wall_seed,future_seed):
    from jevmahjong.training.futures import sample_future
    from jevmahjong.training.collector import call_observation_digest
    sampled=sample_future(env,wall_seed,'unseen')
    before=call_observation_digest(sampled.clone().get_observations())
    result=FutureHandsWorld(sampled,future_seed)
    if call_observation_digest(result.clone().get_observations())!=before:
        raise ValueError('Resampling future-hand RNG changed current observations')
    return result


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
