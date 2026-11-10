"""Answer-independent policy phases and identifiable replay pass supervision."""
import copy
from semantic import describe

POLICY_INSTRUCTION = ('Choose the best legal action in four-player Mahjong Soul East-only riichi mahjong. '
    'Maximize expected net points at the end of the complete match. Consider hand value, winning chances, '
    'deal-in risk, scores, dealer position, honba and riichi sticks. '
    'sh=shanten (lower is closer to tenpai); ukeire=visible remaining improving tiles; '
    'safe=opponent seats for which this tile is genbutsu. Probabilities are action preferences.')


def policy_request(request, view='native'):
    """Shared train/inference transform; confirmed OCR counters stay available."""
    result = copy.deepcopy(request)
    result['questions']['action']['instructions'] = POLICY_INSTRUCTION
    if view == 'native':
        return result
    if view not in ('screen_confirmed', 'screen_unknown'):
        raise ValueError('Unsupported view')
    state = result['state']
    for field in ('tsumogiri_flags', 'riichi_sutehais', 'last_tedashis', 'last_discard'):
        state[field] = None
    state['screen_observation'] = {'temporal_information':'unknown',
        'metadata_confirmed':view == 'screen_confirmed', 'confirmed_metadata':['honba', 'sticks']}
    if phase(result) == 'reaction':
        from desktop.jev_sidecar.tiles import HONORS
        action = next(x for x in result['questions']['action']['criteria'].values() if x['type'] in ('chi','pon'))
        tile = HONORS.get(action['pai'], '0'+action['pai'][1] if action['pai'].endswith('r') else action['pai'])
        source = request['_meta']['reaction_relative_seat']
        state['screen_observation']['reaction'] = {'tile':tile, 'seat':source,
            'offered':sorted({v['type'] for v in result['questions']['action']['criteria'].values()}-{'none'})}
    if view == 'screen_unknown':
        for field in ('scores','dealer','round_wind','hand_number','riichi'):
            state[field] = None
    return result


def pass_outcomes(events):
    """Map each public discard prefix to its next effective log event.

    Another winning/meld claim censors unrecorded choices by other players.
    Never train those implicit replay passes as observed expert decisions.
    Future events control sample admissibility only, never model features or
    legal candidate reconstruction.
    """
    result = {}
    pending = None
    context = None
    count = 0
    for event in events:
        typ = event['type']
        if typ == 'start_kyoku':
            context = ({'E': 0, 'S': 1, 'W': 2, 'N': 3}[event['bakaze']], event['oya'], event['honba'])
            count = 0
            pending = None
        elif typ in ('reach_accepted', 'dora', 'reach'):
            continue
        else:
            if pending is not None:
                result[pending] = typ
                pending = None
            if typ == 'dahai':
                count += 1
                pending = (*context, count)
    return result


def pass_is_identified(obs, outcomes):
    state, _, _ = describe(obs)
    key = (state['round_wind'], state['dealer'], state['honba'], state['discards_count'])
    return outcomes.get(key) == 'tsumo'


def phase(request):
    kinds = {v['type'] for v in request['questions']['action']['criteria'].values()}
    if kinds <= {'none', 'chi', 'pon'} and kinds & {'chi', 'pon'}:
        return 'reaction'
    if kinds <= {'dahai', 'reach'} and 'dahai' in kinds:
        return 'open_discard' if request['state']['melds'][request['state']['seat']] else 'closed_discard'
    raise ValueError('Unsupported clean-policy phase')
