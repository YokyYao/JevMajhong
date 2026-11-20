"""Versioned, label-independent public summaries for a legal reach branch."""
from collections import Counter
import copy
import math

VERSION = 'public-riichi-context-v1'
FIELD = 'riichi_public_context'
HONORS = dict(zip(('E', 'S', 'W', 'N', 'P', 'F', 'C'), range(27, 34)))


def tile(token):
    if token in HONORS:
        return HONORS[token], False
    if len(token) == 3 and token[0] == '5' and token[1] in 'mps' and token[2] == 'r':
        return 'mps'.index(token[1])*9+4, True
    if len(token) != 2 or not token[0].isdigit():
        raise ValueError('Single-tile mpsz/MJAI notation required')
    rank, suit = int(token[0]), token[1]
    if suit == 'z' and rank in range(1, 8):
        return 26+rank, False
    if suit in 'mps' and rank in range(10):
        return 'mps'.index(suit)*9+(4 if rank == 0 else rank-1), rank == 0
    raise ValueError('Invalid tile token')


def tiles(text):
    tokens = text.split() if isinstance(text, str) else text
    if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
        raise ValueError('Public tile-name string/list required')
    return [tile(t) for t in tokens]


def dora_kind(kind):
    if kind < 27:
        return kind//9*9+(kind % 9+1) % 9
    return 27+(kind-27+1) % 4 if kind < 31 else 31+(kind-31+1) % 3


def public_context(request):
    s = request['state']
    q = request['questions']['action']
    pid, dealer, wind = s['seat'], s['dealer'], s['round_wind']
    assert type(pid) is int and pid in range(4) and type(dealer) is int and dealer in range(4)
    assert type(wind) is int and wind in range(4)
    scores = s['scores']
    assert len(scores) == 4 and all(type(v) in (int, float) and math.isfinite(v) for v in scores)
    declared = s['riichi']
    assert isinstance(declared, list) and all(type(p) is int and p in range(4) for p in declared)
    assert pid not in declared, 'Reach branch after self-declaration is invalid'
    threats = set(declared)-{pid}
    own = s['melds'][pid]
    hand = tiles(s['hand'])
    meld_tiles = [tiles(m['tiles']) for m in own]
    assert all(n <= 4 for n in Counter(k for k, _ in hand+[t for m in meld_tiles for t in m]).values())
    closed = not any(m['type'] not in (3, 'ankan') for m in own)
    assert closed, 'Reach branch in an open hand is invalid'
    indicators = s.get('dora_indicators')
    dora = None if indicators is None else Counter(dora_kind(k) for k, _ in tiles(indicators))
    valued = Counter([27+wind, 27+(pid-dealer) % 4, 31, 32, 33])
    ordered = sorted(range(4), key=lambda p: (-scores[p], p))
    river = s['rivers'][pid]
    turn = len(river.split()) if isinstance(river, str) else len(river)
    options = []
    for c in q['criteria'].values():
        if c['type'] != 'dahai' or c.get('sh') != 0:
            continue
        after = list(hand)
        discard = tile(c['pai'])
        if discard not in after:
            raise ValueError('Tenpai discard does not match an owned tile, including red identity')
        after.remove(discard)
        counts = Counter(k for k, _ in after)
        sets = {k for k, n in counts.items() if n >= 3}
        sets.update(m[0][0] for m in meld_tiles if m and len({k for k, _ in m}) == 1)
        safe = c.get('safe')
        if safe is not None:
            assert isinstance(safe, list) and all(type(p) is int and p in range(4) for p in safe)
        ukeire = c.get('ukeire')
        assert ukeire is None or type(ukeire) in (int, float) and math.isfinite(ukeire) and ukeire >= 0
        owned = after+[t for m in meld_tiles for t in m]
        options.append(dict(ukeire=ukeire, safe=None if safe is None else threats <= set(safe),
            value_honor_roles=sum(valued[k] for k in sets),
            dora=None if dora is None else sum(dora[k] for k, _ in owned),
            red=sum(is_red for _, is_red in owned)))
    def bound(key, function):
        values = [o[key] for o in options]
        return None if not values or any(v is None for v in values) else function(values)
    safe_known = bool(options) and all(o['safe'] is not None for o in options)
    safe_options = [o for o in options if o['safe']] if safe_known and threats else []
    safe_ukeire = [o['ukeire'] for o in safe_options]
    return dict(schema=VERSION, self_is_dealer=pid == dealer, seat_wind_kind=27+(pid-dealer) % 4,
        round_wind_kind=27+wind, self_score=scores[pid], score_rank=ordered.index(pid)+1,
        gap_to_second=scores[ordered[1]]-scores[pid], lead_over_fourth=scores[pid]-scores[ordered[-1]],
        turn=turn, opponent_riichi_count=len(threats), closed=closed,
        honba=s['honba'], riichi_sticks=s['sticks'], visible_dora_indicators=None if indicators is None else len(tiles(indicators)),
        tenpai_discard_options=len(options), tenpai_ukeire_min=bound('ukeire', min), tenpai_ukeire_max=bound('ukeire', max),
        safe_tenpai_discard_options=None if not threats or not safe_known else len(safe_options),
        safe_tenpai_ukeire_max=max(safe_ukeire) if safe_ukeire and all(v is not None for v in safe_ukeire) else None,
        value_honor_roles_min=bound('value_honor_roles', min), value_honor_roles_max=bound('value_honor_roles', max),
        dora_after_discard_min=bound('dora', min), dora_after_discard_max=bound('dora', max),
        red_after_discard_min=bound('red', min), red_after_discard_max=bound('red', max),
        complete_tenpai_summary=bool(options) and all(o['ukeire'] is not None for o in options))


def enrich_request(request):
    """Return a copy, preserving action keys/labels and selecting no action."""
    result = copy.deepcopy(request)
    criteria = result['questions']['action']['criteria']
    reach = [c for c in criteria.values() if c['type'] == 'reach']
    if not reach:
        return result
    context = public_context(result)
    for c in reach:
        if FIELD in c and c[FIELD] != context:
            raise ValueError('Conflicting/stale riichi input schema')
        c[FIELD] = copy.deepcopy(context)
    return result
