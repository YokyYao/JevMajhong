"""Bounded public-inventory prior draws; not native continuation environments.

Uniform physical allocations are rejected as a whole unless every declared
riichi hand is tenpai. No actual opponent hand is an input. This does not
condition on the full public action history, infer furiten, refresh native
wait caches or establish a calibrated posterior.
"""
import random

METHOD = 'uniform_physical_allocation_conditioned_on_riichi_tenpai_v1'


class SamplingExhausted(RuntimeError):
    def __init__(self, attempts, accepted, requested):
        super().__init__(f'Public-prior budget exhausted: {accepted}/{requested} draws after {attempts} attempts')
        self.attempts, self.accepted, self.requested = attempts, accepted, requested


def sample_allocations(snapshot, seed, count, max_attempts, shanten):
    if (type(count) is not int or not 1 <= count <= 128 or type(max_attempts) is not int
        or not count <= max_attempts <= 2000000 or type(seed) is not int):
        raise ValueError('Explicit bounded sampling budget and integer seed required')
    own = snapshot['own_seat']
    known, unseen = snapshot['known_tiles'], snapshot['unseen_tiles']
    sizes = snapshot['opponent_concealed_counts']
    required = snapshot['riichi_tenpai_required']
    if (snapshot['schema'] != 'public-win-inventory-v1' or type(own) is not int or own not in range(4)
        or set(sizes) != set(range(4))-{own} or any(type(n) is not int or n not in (1,4,7,10,13) for n in sizes.values())
        or not set(required) <= set(sizes) or len(set(required)) != len(required) or any(type(p) is not int for p in required)
        or any(type(tile) is not int for tile in [*known,*unseen])
        or len(set(known)) != len(known) or len(set(unseen)) != len(unseen)
        or set(known) & set(unseen) or set(known) | set(unseen) != set(range(136))
        or len(unseen) < sum(sizes.values())
        or len(unseen)-sum(sizes.values()) != snapshot['unknown_nonhand_tile_count']):
        raise ValueError('Invalid public inventory partition or opponent constraints')
    rng = random.Random(seed)
    samples = []
    for attempt in range(1,max_attempts+1):
        tiles = list(unseen)
        rng.shuffle(tiles)
        hands = {}
        offset = 0
        for player,n in sorted(sizes.items()):
            hands[player] = tiles[offset:offset+n]
            offset += n
        if any(shanten(hands[player]) != 0 for player in required):
            continue
        samples.append(dict(opponent_hands=hands, unknown_nonhand_tiles=tiles[offset:]))
        if len(samples) == count:
            return dict(method=METHOD, seed=seed, requested=count, attempts=attempt,
                acceptance_fraction=count/attempt, allocations=samples,
                all_declared_riichi_tenpai=True, calibrated_public_history_posterior=False,
                native_environment_constructed=False, training_labels=False)
    # Partial allocations are never returned as a complete labelled case.
    raise SamplingExhausted(max_attempts,len(samples),count)
