"""Public kan-option features and a small paired-value decision component."""
from collections import Counter
import copy
import math
import torch
from riichienv import MeldType
from jevmahjong.core.state import shanten
from .call_helpers import tile_ids, dora_type, call_discards
from .kan import is_kan_request
from .routing import is_call_request
from .defense import VERSION as DEFENSE_VERSION,is_defense_request
from .kan_progress import VERSION as PROGRESS_VERSION, NAMES as PROGRESS_NAMES, visible_bounds, option_progress

VERSION = 'public-kan-context-v1'
KINDS = ('none', 'dahai', 'reach', 'chi', 'pon', 'ankan', 'daiminkan', 'kakan')
STATE_NAMES = ('shanten', 'closed', 'meld_count', 'own_riichi', 'threats', 'discards',
    'dealer', 'honba', 'sticks', 'gap_leader', 'gap_last', 'own_dora', 'own_red',
    'pairs', 'table_kans', 'indicators')
AFTER_NAMES = ('after_shanten', 'shanten_gain', 'closed_after', 'loses_closed',
    'meld_count_after', 'dragon_material', 'wind_material', 'simples_material',
    'meld_dora', 'meld_red', 'remaining_pairs', 'pre_rinshan', 'unsafe_discard',
    'consumed_dora', 'consumed_red', 'call_tile_honor')
NAMES = (*KINDS, *AFTER_NAMES, *(f'{kind}/{name}' for kind in KINDS for name in STATE_NAMES))


def feature_names(version=VERSION):
    if version == VERSION: return NAMES
    if version in (PROGRESS_VERSION,DEFENSE_VERSION): return (*NAMES, *PROGRESS_NAMES)
    raise ValueError('Context feature schema differs')


def feature_vectors(request, version=VERSION):
    names = feature_names(version)
    accepted = (is_defense_request(request) if version == DEFENSE_VERSION else
        is_kan_request(request) or (version == PROGRESS_VERSION and is_call_request(request)))
    if not accepted:
        raise ValueError('Legal kan or versioned pure-call window required')
    s = request['state']; pid = s['seat']; hand = tile_ids(s['hand'])
    own = [(m['type'], tile_ids(m['tiles'])) for m in s['melds'][pid]]
    closed = all(kind == int(MeldType.Ankan) for kind, _ in own)
    before = shanten(tuple(sorted(t // 4 for t in hand)))
    indicators = tile_ids(s['dora_indicators']); dora = Counter(dora_type(t) for t in indicators)
    threats = set(s['riichi']) - {pid}; rivers = [{t // 4 for t in tile_ids(r)} for r in s['rivers']]
    winds = Counter((27 + s['round_wind'], 27 + (pid - s['dealer']) % 4))
    flat = hand + [t for _, tiles in own for t in tiles]
    def red(tiles): return sum(t in (16, 52, 88) for t in tiles)
    def pairs(tiles): return sum(n >= 2 for n in Counter(t // 4 for t in tiles).values())
    def doras(tiles): return sum(dora[t // 4] for t in tiles)
    kans = {int(MeldType.Ankan), int(MeldType.Daiminkan), int(MeldType.Kakan)}
    state = [before / 6, float(closed), len(own) / 4, float(pid in s['riichi']), len(threats) / 3,
        s['discards_count'] / 72, float(pid == s['dealer']), s['honba'] / 4, s['sticks'] / 4,
        (s['scores'][pid] - max(s['scores'])) / 40000,
        (s['scores'][pid] - min(s['scores'])) / 40000, doras(flat) / 12, red(flat) / 3,
        pairs(hand) / 7, sum(m['type'] in kans for ms in s['melds'] for m in ms) / 4,
        len(indicators) / 5]
    result = []
    bounds = visible_bounds(s) if version in (PROGRESS_VERSION,DEFENSE_VERSION) else None
    for c in request['questions']['action']['criteria'].values():
        kind = c['type']
        if kind not in KINDS: raise ValueError('Unsupported action in kan context: ' + kind)
        remaining = list(hand); melds = list(own); consumed = []; pre_draw = kind in ('ankan', 'daiminkan', 'kakan')
        call_tile = tile_ids(c['pai'])[0] if 'pai' in c else None
        if kind in ('chi', 'pon'):
            consumed = tile_ids(' '.join(c['consumed']))
            melds.append((int(MeldType.Chi if kind == 'chi' else MeldType.Pon), consumed + [call_tile]))
            choices = call_discards(hand, c)
        elif kind in ('ankan', 'daiminkan'):
            consumed = tile_ids(' '.join(c['consumed']))
            if len(consumed) != (4 if kind == 'ankan' else 3): raise ValueError('Invalid kan consumption')
            for tile in consumed: remaining.remove(tile)
            melds.append((int(MeldType.Ankan if kind == 'ankan' else MeldType.Daiminkan),
                consumed if kind == 'ankan' else consumed + [call_tile]))
            choices = [(None, remaining)]
        elif kind == 'kakan':
            # MJAI consumed contains the existing pon, not three owned tiles.
            consumed = [call_tile]; remaining.remove(call_tile)
            indexes = [i for i, (m, tiles) in enumerate(melds)
                if m == int(MeldType.Pon) and tiles[0] // 4 == call_tile // 4]
            if len(indexes) != 1: raise ValueError('Kakan requires one existing pon')
            i = indexes[0]; melds[i] = (int(MeldType.Kakan), melds[i][1] + [call_tile])
            choices = [(None, remaining)]
        elif kind == 'dahai':
            remaining.remove(call_tile); choices = [(call_tile, remaining)]
        elif kind == 'reach':
            choices = []
            for tile in sorted(set(hand)):
                after = list(hand); after.remove(tile); choices.append((tile, after))
        else: choices = [(None, remaining)]
        distances = [shanten(tuple(sorted(t // 4 for t in rest))) for _, rest in choices]
        best = min(distances); best_choices = [v for v, d in zip(choices, distances) if d == best]
        after_closed = all(m == int(MeldType.Ankan) for m, _ in melds)
        meld_tiles = [t for _, tiles in melds for t in tiles]
        triplets = [tiles[0] // 4 for _, tiles in melds if len({t // 4 for t in tiles}) == 1]
        safe = any(tile is not None and all(tile // 4 in rivers[p] for p in threats)
            for tile, _ in best_choices) if threats else True
        after = [best / 6, (before - best) / 3, float(after_closed), float(closed and not after_closed),
            len(melds) / 4, sum(t >= 31 for t in triplets) / 3, sum(winds[t] for t in triplets) / 2,
            float(any(all(t // 4 < 27 and t // 4 % 9 not in (0, 8) for t in rest + meld_tiles)
                for _, rest in best_choices)), doras(meld_tiles) / 8, red(meld_tiles) / 3,
            max(pairs(rest) for _, rest in best_choices) / 7, float(pre_draw),
            float(bool(threats) and not pre_draw and kind != 'none' and not safe),
            doras(consumed) / 4, float(red(consumed)), float(call_tile is not None and call_tile // 4 >= 27)]
        vector = [float(kind == k) for k in KINDS] + after + [v if kind == k else 0. for k in KINDS for v in state]
        if bounds is not None:
            known = ({tile_ids(c['pai'])[0]: c['ukeire']} if kind == 'dahai' and 'ukeire' in c else None)
            vector += option_progress(best_choices, bounds, dora, known)
        if len(vector) != len(names) or not all(math.isfinite(v) for v in vector): raise ValueError('Invalid context features')
        result.append(vector)
    return result


def context_scores(request, context):
    names = feature_names(context['version'])
    if tuple(context['names']) != names: raise ValueError('Context feature schema differs')
    x = torch.tensor(feature_vectors(request,context['version']),dtype=torch.float64)
    if context.get('model_kind','ridge') == 'mlp':
        h = context['hidden']; weights = context['state_dict']
        if not isinstance(h,int) or not 1 <= h <= 128:raise ValueError('Invalid context hidden width')
        shapes = {'0.weight':(h,len(names)),'0.bias':(h,),'2.weight':(1,h)}
        if set(weights) != set(shapes) or any(v.shape != shapes[k] or not torch.isfinite(v).all() for k,v in weights.items()):
            raise ValueError('Invalid context network parameters')
        hidden = torch.tanh(x @ weights['0.weight'].double().cpu().T + weights['0.bias'].double().cpu())
        return (hidden @ weights['2.weight'].double().cpu().T).flatten()
    if context.get('model_kind','ridge') != 'ridge': raise ValueError('Unknown context model kind')
    coef = context['coefficients'].double().cpu()
    if coef.shape != (len(names),) or not torch.isfinite(coef).all(): raise ValueError('Invalid context coefficients')
    return x @ coef


def probabilities(request, reference, context):
    scores = context_scores(request,context)
    prior = context.get('reference_prior_strength')
    if prior is not None and (not math.isfinite(prior) or prior <= 0):
        raise ValueError('Finite positive reference prior strength required')
    q = torch.as_tensor(reference, dtype=torch.float64)
    if q.ndim != 1 or len(q) != len(request['questions']['action']['criteria']) or not torch.isfinite(q).all() or (q < 0).any() or q.sum() <= 0:
        raise ValueError('Invalid reference distribution')
    q = q / q.sum()
    if not torch.count_nonzero(scores): return q.tolist()
    if prior is not None:
        return (scores / prior + q.clamp_min(1e-30).log()).softmax(-1).tolist()
    # The tiny prior resolves value ties; it is not a fixed call bonus.
    return (scores + 1e-8 * q.clamp_min(1e-30).log()).softmax(-1).tolist()


class KanContextPolicyClient:
    accepts = staticmethod(is_kan_request)
    def __init__(self, frozen, context): self.frozen, self.context = frozen, context

    def system_one(self, **request):
        answer = self.frozen.system_one(**request)
        if not self.accepts(request): return answer
        keys = list(request['questions']['action']['criteria'])
        reference = answer['answers']['action']['probabilities']
        values = probabilities(request, [reference[k] for k in keys], self.context)
        result = copy.deepcopy(answer); result['answers']['action']['probabilities'] = dict(zip(keys, values))
        return result


class CallContextPolicyClient(KanContextPolicyClient):
    """Public progress residual only for chi/pon/pass, preserving kan routing."""
    accepts = staticmethod(is_call_request)


class DefenseContextPolicyClient(KanContextPolicyClient):
    """Ordinary push/fold residual; all call and kan menus keep their routing."""
    accepts = staticmethod(is_defense_request)
