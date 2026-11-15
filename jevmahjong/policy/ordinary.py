"""Public ordinary-discard windows using the deployed shanten/safety guard."""


def is_ordinary_request(request):
    try:
        state=request['state'];criteria=list(request['questions']['action']['criteria'].values())
        return (state['riichi'] is not None and state['seat'] not in state['riichi']
            and len(criteria)>1 and {c['type'] for c in criteria}<={'dahai','reach'}
            and any(c['type']=='dahai' for c in criteria))
    except (KeyError,TypeError):
        return False


def is_attack_request(request,opened=None):
    """Select visible non-riichi-threat states, without reading outcomes."""
    if not is_ordinary_request(request):return False
    state=request['state'];seat=state['seat']
    if any(state.get(key) is None for key in ('scores','melds','rivers','honba','sticks','hand_number')):
        return False
    if set(state['riichi'])-{seat}:return False
    if opened is None:return True
    actual=any(m['type'] not in (3,'ankan') for m in state['melds'][seat])
    return actual==opened


def request_eligible(request):
    """Mirror PolicyAgent admission for closed and open ordinary decisions."""
    if not is_ordinary_request(request):raise ValueError('Ordinary discard decision required')
    criteria=list(request['questions']['action']['criteria'].values())
    state=request['state'];threats=set(state['riichi'])-{state['seat']}
    if not threats and any(c['type']=='reach' for c in criteria):return list(range(len(criteria)))
    discards=[(i,c) for i,c in enumerate(criteria) if c['type']=='dahai']
    fold=bool(threats) and min(c['sh'] for _,c in discards)>=1
    _,ref=min(discards,key=lambda item:(len(threats-set(item[1]['safe'])) if fold else 0,
        item[1]['sh'],-item[1]['ukeire'],item[1]['red'],item[0]))
    risk=len(threats-set(ref['safe']))
    return [i for i,c in enumerate(criteria) if c['type']!='dahai' or (
        c['sh']<=ref['sh'] and (not fold or len(threats-set(c['safe']))<=risk))]
