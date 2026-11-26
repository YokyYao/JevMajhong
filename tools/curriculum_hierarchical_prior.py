"""A declared hierarchical public prior, not weighted uniform-joint inference.

Uniformly shuffle the riichi-seat order, then sample each riichi hand uniformly
from the remaining physical inventory conditional on native tenpai. Partition
the remaining hands/wall uniformly, and reject public-history incompatibility.
For2/3 riichi this is a different prior from uniform-joint rejection. Equal
weights refer ONLY to draws from this declared prior; no calibrated posterior
or importance correction is claimed. Current known opponent hands are absent.
"""
import random

from tools.curriculum_decision_history import history_flags
from tools.curriculum_public_worlds import SamplingExhausted

METHOD='hierarchical_random_riichi_order_public_prior_v1'


def sample(snapshot,history,riichi,seed,count,max_attempts,waits):
    if (type(seed) is not int or type(count) is not int or not 1<=count<=128
        or type(max_attempts) is not int or not count<=max_attempts<=2000000):
        raise ValueError('Explicit bounded prior budget required')
    own=snapshot['own_seat'];sizes=snapshot['opponent_concealed_counts']
    required=snapshot['riichi_tenpai_required'];known=snapshot['known_tiles'];unseen=snapshot['unseen_tiles']
    if (snapshot['schema']!='public-win-inventory-v1' or type(own) is not int or own not in range(4)
        or set(sizes)!=set(range(4))-{own} or any(type(p) is not int for p in sizes)
        or any(type(n) is not int or n not in (1,4,7,10,13) for n in sizes.values())
        or not set(required)<=set(sizes) or len(set(required))!=len(required)
        or any(type(p) is not int for p in required)
        or len(riichi)!=4 or any(type(v) is not bool for v in riichi)
        or set(required)!={p for p in sizes if riichi[p]}
        or any(type(t) is not int for t in [*known,*unseen])
        or len(set(known))!=len(known) or len(set(unseen))!=len(unseen)
        or set(known)&set(unseen) or set(known)|set(unseen)!=set(range(136))
        or len(unseen)<sum(sizes.values())
        or len(unseen)-sum(sizes.values())!=snapshot['unknown_nonhand_tile_count']):
        raise ValueError('Invalid public inventory/riichi constraints')
    rng=random.Random(seed);samples=[];attempts=0;history_rejections=0;world_proposals=0
    while attempts<max_attempts:
        world_proposals+=1;order=sorted(required);rng.shuffle(order)
        available=set(unseen);hands={};complete=True
        for player in order:
            found=False
            while attempts<max_attempts:
                proposed=rng.sample(sorted(available),sizes[player]);attempts+=1
                if waits(player,proposed):
                    hands[player]=proposed;available.difference_update(proposed);found=True;break
            if not found:complete=False;break
        if not complete:break
        if not required:attempts+=1
        remaining=sorted(available);rng.shuffle(remaining);offset=0
        for player in sorted(set(sizes)-set(required)):
            hands[player]=remaining[offset:offset+sizes[player]];offset+=sizes[player]
        candidate=dict(opponent_hands=hands,unknown_nonhand_tiles=remaining[offset:],proposal_riichi_order=order)
        flags=history_flags(history,own,riichi,candidate,waits)
        if not flags['compatible']:history_rejections+=1;continue
        candidate['opponent_temporary_furiten']=flags['opponent_temporary_furiten']
        samples.append(candidate)
        if len(samples)==count:
            return dict(method=METHOD,seed=seed,attempts=attempts,world_proposals=world_proposals,
                history_rejections=history_rejections,requested=count,allocations=samples,
                uniform_joint_target_for_multi_riichi=False,
                uniform_joint_equivalent_when_opponent_riichi_le1=len(required)<=1,
                equal_weight_draws_from_declared_hierarchical_prior=True,
                importance_weights_for_uniform_joint_not_provided=True,
                calibrated_public_history_posterior=False,training_labels=False)
    raise SamplingExhausted(attempts,len(samples),count)
