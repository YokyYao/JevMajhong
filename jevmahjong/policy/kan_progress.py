"""Player-visible progress bounds for legacy requests without chi provenance."""
from collections import Counter
from riichienv import MeldType
from jevmahjong.core.state import shanten
from .call_helpers import tile_ids

VERSION = 'public-kan-context-v2'
NAMES = ('ukeire_lower', 'ukeire_upper', 'improving_types_lower',
    'improving_types_upper', 'concealed_dora_after', 'concealed_red_after')


def visible_bounds(state):
    """Bound unique visible counts; called tiles also occur in source rivers.

    Old requests omit which tile was called for each chi. River membership
    can resolve some calls; otherwise retain bounds instead of guessing.
    Bounds for different types need not be simultaneously attainable.
    """
    rivers = Counter(t // 4 for river in state['rivers'] for t in tile_ids(river))
    counts = Counter(t // 4 for t in tile_ids(state['hand'])) + rivers
    counts.update(t // 4 for t in tile_ids(state['dora_indicators']))
    uncertain = Counter()
    for melds in state['melds']:
        for meld in melds:
            types = [t // 4 for t in tile_ids(meld['tiles'])]
            counts.update(types)
            if meld['type'] == int(MeldType.Ankan):
                continue
            candidates = set(types) & rivers.keys()
            if len(candidates) == 1:
                counts[next(iter(candidates))] -= 1
            elif candidates:
                uncertain.update(candidates)
    lower = [counts[t] - uncertain[t] for t in range(34)]
    if any(v < 0 or v > 4 for v in lower):
        raise ValueError('Inconsistent player-visible tile inventory')
    return lower, [min(4, counts[t]) for t in range(34)]


def option_progress(choices, bounds, dora, known_ukeire=None):
    """Separate best-shanten choice summaries, before unknown kan replacement.

    The existing discard criterion provides exact ukeire where available.
    Other actions use public inventory bounds, with no other hands or wall.
    Dora/red retention maxima need not belong to the max-ukeire discard.
    """
    low, high = bounds
    profiles = []
    for discard, remaining in choices:
        types = tuple(sorted(t // 4 for t in remaining))
        distance = shanten(types)
        improving = [t for t in range(34) if low[t] < 4
            and shanten(tuple(sorted((*types, t)))) < distance]
        lo = sum(4 - high[t] for t in improving)
        hi = sum(4 - low[t] for t in improving)
        if known_ukeire is not None and discard in known_ukeire:
            exact = known_ukeire[discard]
            if not isinstance(exact, (int, float)) or not 0 <= exact <= 136:
                raise ValueError('Invalid public discard ukeire')
            lo = hi = exact
        profiles.append((lo / 136, hi / 136,
            sum(high[t] < 4 for t in improving) / 34, len(improving) / 34,
            sum(dora[t // 4] for t in remaining) / 12,
            sum(t in (16, 52, 88) for t in remaining) / 3))
    return [max(values) for values in zip(*profiles)]
