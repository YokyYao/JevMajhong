"""Versioned, player-visible tile channels for pure calling decisions."""
from collections import Counter
from .call_helpers import tile_ids

VERSION = 'public-call-tile-context-v1'
CHANNELS = ('own_hand_tiles', 'remaining_owned_tiles', 'consumed_tiles', 'called_tile',
    'dora_indicator_tiles', *(f'river_relative_{i}' for i in range(4)),
    *(f'meld_relative_{i}' for i in range(4)))
NAMES = tuple(f'{channel}_{tile}' for channel in CHANNELS for tile in range(34))


def vectors(request):
    """Use own tiles and public observations; never other concealed hands or walls.

    Rivers and melds are separate observation channels, not a claimed remaining
    tile count: a called discard can be visible in both channels.
    """
    state=request['state'];seat=state['seat'];hand=tile_ids(state['hand'])
    own=Counter(t//4 for t in hand)
    criteria=list(request['questions']['action']['criteria'].values())
    called={tile_ids(c['pai'])[0]//4 for c in criteria if c['type'] in ('chi','pon')}
    if len(called)!=1:raise ValueError('One observed discard required for a pure-call tile context')
    incoming=next(iter(called))
    def histogram(tiles):
        counts=Counter(t//4 for t in tiles)
        return [counts[t]/4 for t in range(34)]
    indicators=histogram(tile_ids(state['dora_indicators']))
    rivers=[histogram(tile_ids(state['rivers'][(seat+i)%4])) for i in range(4)]
    melds=[histogram([t for m in state['melds'][(seat+i)%4] for t in tile_ids(m['tiles'])]) for i in range(4)]
    result=[]
    for criterion in criteria:
        kind=criterion['type']
        if kind not in ('none','chi','pon'):raise ValueError('Pure chi/pon/pass tile context required')
        consumed=tile_ids(' '.join(criterion['consumed'])) if kind in ('chi','pon') else []
        used=Counter(t//4 for t in consumed)
        if any(n>own[t] for t,n in used.items()):raise ValueError('Consumed tile not owned')
        remaining=[(own[t]-used[t])/4 for t in range(34)]
        channels=[histogram(hand),remaining,histogram(consumed),
            [float(t==incoming) for t in range(34)],indicators,*rivers,*melds]
        flat=[value for channel in channels for value in channel]
        if len(flat)!=len(NAMES):raise ValueError('Public tile channel schema differs')
        result.append(flat)
    return result
