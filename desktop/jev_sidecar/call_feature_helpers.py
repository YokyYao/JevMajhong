"""Portable equivalents of the frozen native call helpers; checked against TRAIN/validation."""
from enum import IntEnum
from .tiles import index, normal

class MeldType(IntEnum):
    Ankan = 3

def mpsz_to_tid(tile):
    tid = index(tile) * 4
    return tid + (tile[0] == '5' and tile[1] in 'mps')


def tile_ids(text):
    # State uses mpsz (0p,5z), native action criteria use MJAI (5pr,P).
    honors = dict(zip(('E','S','W','N','P','F','C'), ('1z','2z','3z','4z','5z','6z','7z')))
    normalized = [honors.get(t, '0' + t[1] if len(t) == 3 and t.endswith('r') else t) for t in text.split()]
    return [mpsz_to_tid(t) for t in normalized]


def dora_type(indicator):
    t = indicator // 4
    if t < 27: return t // 9 * 9 + (t % 9 + 1) % 9
    if t < 31: return 27 + (t - 27 + 1) % 4
    return 31 + (t - 31 + 1) % 3


def call_discards(hand, criterion):
    """MJ Soul kuikae exclusions for a legal chi/pon, using owned tiles only."""
    if criterion['type'] not in ('chi', 'pon'): raise ValueError('Call required')
    remaining = list(hand)
    consumed = tile_ids(' '.join(criterion['consumed']))
    if len(consumed) != 2: raise ValueError('Native call must consume two owned tiles')
    for tile in consumed: remaining.remove(tile)
    called = tile_ids(criterion['pai'])[0] // 4
    forbidden = {called}
    low, high = sorted(t // 4 for t in consumed)
    if criterion['type'] == 'chi':
        if low == called + 1 and high == called + 2 and called % 9 <= 5: forbidden.add(called + 3)
        if low == called - 2 and high == called - 1 and called % 9 >= 3: forbidden.add(called - 3)
    candidates = []
    for tile in sorted(set(remaining)):
        if tile // 4 not in forbidden:
            after = list(remaining); after.remove(tile); candidates.append((tile, after))
    if not candidates: raise ValueError('Call leaves no kuikae-legal discard')
    return candidates
