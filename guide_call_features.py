"""Public-state call features motivated by the user-supplied strategy guide.

These are covariates for point regression, not rewards, win probabilities or
hand value predictions. No guide slogan is a legality filter or a fixed bonus.
"""
from collections import Counter
from riichienv import MeldType
from riichienv.convert import mpsz_to_tid
from semantic import shanten

VERSION = 'guide-public-call-v1'
NAMES = (
    'call', 'chi', 'pon', 'first_open', 'open_melds_after',
    'shanten_before', 'best_shanten_after', 'shanten_gain', 'tenpai_after',
    'secured_yakuhai_han', 'all_simples_material', 'single_suit_material',
    'triplet_melds', 'retained_dora', 'retained_red',
    'riichi_threats', 'dealer_riichi', 'max_opponent_open_melds',
    'safe_tiles_before', 'safe_tiles_after', 'safe_forced_discard_available',
    'turn', 'late_turn', 'self_dealer', 'score_above_start',
    'gap_to_leader', 'gap_to_last', 'honba', 'sticks',
    'threat_x_distance', 'threat_x_no_simple_or_yakuhai',
    'threat_x_safe_tiles_lost', 'late_x_tenpai', 'dora_x_simple_or_yakuhai',
)


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


def strategy_features(request):
    state = request['state']; criteria = list(request['questions']['action']['criteria'].values())
    kinds = {c['type'] for c in criteria}
    if not kinds <= {'chi', 'pon', 'none'} or 'none' not in kinds or len(kinds) < 2:
        raise ValueError('Only a native call-versus-pass choice is supported')
    pid, dealer = state['seat'], state['dealer']
    hand = tile_ids(state['hand']); own_melds = state['melds'][pid]
    rivers = [set(t // 4 for t in tile_ids(r)) for r in state['rivers']]
    threats = [p for p in state['riichi'] if p != pid]
    def safe(tiles):
        # Intersection across threats, never the union of their rivers.
        return sum(bool(threats) and all(t // 4 in rivers[p] for p in threats) for t in tiles)
    own_open = sum(m['type'] != int(MeldType.Ankan) for m in own_melds)
    distance = shanten(tuple(sorted(t // 4 for t in hand)))
    dora = Counter(dora_type(t) for t in tile_ids(state['dora_indicators']))
    valued = Counter([31, 32, 33, 27 + state['round_wind'], 27 + (pid - dealer) % 4])
    opponent_open = max(sum(m['type'] != int(MeldType.Ankan) for m in ms)
                        for p, ms in enumerate(state['melds']) if p != pid)
    turn = len(state['rivers'][pid].split()); scores = state['scores']
    output = []
    for criterion in criteria:
        kind = criterion['type']
        if kind == 'none': output.append([0.] * len(NAMES)); continue
        discards = call_discards(hand, criterion)
        distances = [shanten(tuple(sorted(t // 4 for t in rest))) for _, rest in discards]
        best = min(distances)
        choices = [(t, rest) for (t, rest), d in zip(discards, distances) if d == best]
        new_meld = tile_ids(' '.join([criterion['pai'], *criterion['consumed']]))
        meld_tiles = [tile_ids(m['tiles']) for m in own_melds] + [new_meld]
        flat_melds = [t for meld in meld_tiles for t in meld]
        secured = sum(valued[meld[0] // 4] for meld in meld_tiles
                      if len({t // 4 for t in meld}) == 1)
        materials = [rest + flat_melds for _, rest in choices]
        simple = max(all(t // 4 < 27 and t // 4 % 9 not in (0, 8) for t in tiles) for tiles in materials)
        single_suit = max(len({t // 36 for t in tiles if t < 108}) == 1 for tiles in materials)
        red = max(sum(t in (16, 52, 88) for t in tiles) for tiles in materials)
        retained_dora = max(sum(dora[t // 4] + (t in (16, 52, 88)) for t in tiles) for tiles in materials)
        safe_before = safe(hand); safe_after = max(safe(rest) for _, rest in choices)
        support = bool(simple or secured)
        values = [1., kind == 'chi', kind == 'pon', own_open == 0, (own_open + 1) / 4,
            distance / 6, best / 6, (distance - best) / 3, best == 0,
            secured / 4, simple, single_suit,
            sum(len({t // 4 for t in meld}) == 1 for meld in meld_tiles) / 4,
            retained_dora / 6, red / 3,
            len(threats) / 3, dealer in threats, opponent_open / 4,
            safe_before / 13, safe_after / 13, any(safe([t]) for t, _ in discards),
            turn / 18, turn >= 12, pid == dealer, (scores[pid] - 25000) / 10000,
            (max(scores) - scores[pid]) / 10000, (scores[pid] - min(scores)) / 10000,
            state['honba'] / 4, state['sticks'] / 4,
            len(threats) * best / 6, len(threats) * (not support) / 3,
            len(threats) * (safe_before - safe_after) / 13,
            (turn >= 12) and best == 0, retained_dora * support / 6]
        if len(values) != len(NAMES): raise AssertionError('Strategy feature schema mismatch')
        output.append([float(v) for v in values])
    return output
