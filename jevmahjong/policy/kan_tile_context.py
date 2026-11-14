"""Versioned public tile identities for complete kan decision menus."""
from collections import Counter
from .call_helpers import tile_ids

VERSION = 'public-kan-tile-context-v1'
CHANNELS = ('own_hand_tiles', 'remaining_owned_tiles', 'consumed_tiles',
    'incoming_discard', 'dora_indicator_tiles',
    *(f'river_relative_{i}' for i in range(4)),
    *(f'meld_relative_{i}' for i in range(4)))
NAMES = tuple(f'{channel}_{tile}' for channel in CHANNELS for tile in range(34))


def vectors(request):
    """Describe owned consumption before any unknown replacement draw.

    A kakan consumes its one added owned tile; its consumed MJAI field describes
    the existing public pon. Reach declares intent and leaves owned tiles intact
    until a subsequent discard. Rivers/melds are separate observation channels,
    since called discards can occur in both; they are not remaining-copy counts.
    """
    state = request['state']; seat = state['seat']
    hand = tile_ids(state['hand']); own = Counter(t // 4 for t in hand)
    criteria = list(request['questions']['action']['criteria'].values())
    kinds = {c['type'] for c in criteria}
    if not kinds & {'ankan', 'daiminkan', 'kakan'}:
        raise ValueError('A legal kan window is required for kan tile channels')
    if not kinds <= {'none', 'chi', 'pon', 'ankan', 'daiminkan', 'kakan', 'dahai', 'reach'}:
        raise ValueError('Unsupported action in kan tile channels')

    def one_tile(criterion):
        tiles = tile_ids(criterion.get('pai', ''))
        if len(tiles) != 1:
            raise ValueError('One action tile is required')
        return tiles[0]

    incoming = {one_tile(c) // 4 for c in criteria if c['type'] in ('chi', 'pon', 'daiminkan')}
    if len(incoming) > 1:
        raise ValueError('Mixed calls must refer to one observed discard')
    incoming_channel = [float(t in incoming) for t in range(34)]

    def histogram(tiles):
        counts = Counter(t // 4 for t in tiles)
        return [counts[t] / 4 for t in range(34)]

    indicators = histogram(tile_ids(state['dora_indicators']))
    rivers = [histogram(tile_ids(state['rivers'][(seat + i) % 4])) for i in range(4)]
    melds = [histogram([t for m in state['melds'][(seat + i) % 4]
                       for t in tile_ids(m['tiles'])]) for i in range(4)]
    result = []
    for criterion in criteria:
        kind = criterion['type']
        if kind in ('dahai', 'kakan'):
            consumed = [one_tile(criterion)]
        elif kind in ('chi', 'pon', 'ankan', 'daiminkan'):
            consumed = tile_ids(' '.join(criterion['consumed']))
            expected = {'chi': 2, 'pon': 2, 'ankan': 4, 'daiminkan': 3}[kind]
            if len(consumed) != expected:
                raise ValueError('Invalid owned consumption in kan menu')
        else:
            consumed = []
        used = Counter(t // 4 for t in consumed)
        if any(n > own[t] for t, n in used.items()):
            raise ValueError('Consumed tile is not owned')
        remaining = [(own[t] - used[t]) / 4 for t in range(34)]
        channels = [histogram(hand), remaining, histogram(consumed),
                    incoming_channel, indicators, *rivers, *melds]
        flat = [value for channel in channels for value in channel]
        if len(flat) != len(NAMES):
            raise ValueError('Kan tile channel schema differs')
        result.append(flat)
    return result
