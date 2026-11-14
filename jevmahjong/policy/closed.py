"""Public ordinary closed-hand opportunities, excluding kan and forced turns."""


def is_closed_request(request):
    state=request['state'];criteria=list(request['questions']['action']['criteria'].values())
    return (state['seat'] not in state['riichi'] and len(criteria)>1
        and not any(m['type'] not in (3,'ankan') for m in state['melds'][state['seat']])
        and {c['type'] for c in criteria}<={'dahai','reach'}
        and any(c['type']=='dahai' for c in criteria))


def request_eligible(request):
    """Match the deployed shanten/safety guard, retaining legal reach choices."""
    if not is_closed_request(request):raise ValueError('Ordinary closed-hand decision required')
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
