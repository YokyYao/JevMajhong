"""Public-only physical tile inventory for future endgame belief sampling.

This derives missing tile identities and opponent concealed counts, not a
posterior over hands, sampled worlds, labels or a runnable environment.
"""
from collections import defaultdict

SCHEMA = 'public-win-inventory-v1'


def inventory(own_seat, own_hand, discards, melds, indicators, riichi, win_kind):
    if type(own_seat) is not int or own_seat not in range(4) or win_kind not in ('ron','tsumo'):
        raise ValueError('A verified native win window is required')
    if len(discards) != 4 or len(melds) != 4 or len(riichi) != 4:
        raise ValueError('Four public player records are required')
    if any(type(v) is not bool for v in riichi):
        raise ValueError('Explicit public riichi flags required')
    owners = defaultdict(list)
    def see(tile, location):
        if type(tile) is not int or tile not in range(136):
            raise ValueError('Invalid physical public tile id')
        owners[tile].append(location)
    for tile in own_hand: see(tile, ('hand',own_seat))
    for seat, river in enumerate(discards):
        for tile in river: see(tile, ('river',seat))
    called = {}
    for seat, group in enumerate(melds):
        if len(group) > 4: raise ValueError('Too many public melds')
        for index, meld in enumerate(group):
            tiles = meld['tiles']
            if len(tiles) not in (3,4) or len(set(tiles)) != len(tiles):
                raise ValueError('Invalid public meld tiles')
            taken = meld.get('called_tile')
            if taken is not None:
                if taken not in tiles or taken in called:
                    raise ValueError('Invalid duplicate called tile')
                called[taken] = (seat,index)
            for tile in tiles: see(tile, ('meld',seat,index))
    for tile in indicators: see(tile, ('dora',))
    for tile, places in owners.items():
        if len(places) == 1: continue
        allowed = (len(places) == 2 and tile in called
            and sum(place[0] == 'river' for place in places) == 1
            and ('meld',*called[tile]) in places
            and next(place[1] for place in places if place[0] == 'river') != called[tile][0])
        if not allowed: raise ValueError('Public physical tile duplicated outside a called river/meld pair')
    expected_own = (14 if win_kind == 'tsumo' else 13)-3*len(melds[own_seat])
    if len(own_hand) != expected_own:
        raise ValueError('Own concealed count differs from verified win kind')
    concealed = {seat:13-3*len(group) for seat,group in enumerate(melds) if seat != own_seat}
    unseen = sorted(set(range(136))-set(owners))
    remainder = len(unseen)-sum(concealed.values())
    if remainder < 0: raise ValueError('Public inventory leaves too few unknown tiles')
    return dict(schema=SCHEMA, own_seat=own_seat, known_tiles=sorted(owners), unseen_tiles=unseen,
        opponent_concealed_counts=concealed, unknown_nonhand_tile_count=remainder,
        riichi_tenpai_required=[seat for seat in concealed if riichi[seat]],
        public_called_tile_duplicates=sum(len(v)==2 for v in owners.values()),
        posterior_or_world=False)


def from_observation(obs, win_kind):
    """Reject accidental omniscient input rather than quietly ignoring it."""
    value = obs.to_dict()
    seat = obs.player_id
    if any(hand for player,hand in enumerate(value['hands']) if player != seat):
        raise ValueError('An opponent concealed hand leaked into the observation')
    if list(value['hands'][seat]) != list(obs.hand):
        raise ValueError('Own public observation hand differs')
    melds = [[dict(tiles=list(m.tiles), called_tile=m.called_tile) for m in group] for group in value['melds']]
    return inventory(seat,list(obs.hand),value['discards'],melds,value['dora_indicators'],value['riichi_declared'],win_kind)
