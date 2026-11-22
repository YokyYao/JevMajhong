"""Public-only risk-window selection and unchanged serving-guard diagnostics."""


def category(types,state,existing=False):
    seat=state['seat'];threats={p for p in state['riichi'] if p!=seat}
    if (existing or types!={'dahai'} or not threats or seat in state['riichi'] or state['melds'][seat]):return None
    return 'closed_discard_under_riichi'


def guard_diagnostic(obs,actions,features,request,label,guarded_indices,heuristic_index):
    keys=list(request['questions']['action']['criteria']);index=keys.index(label);seat=request['state']['seat']
    threats={p for p in request['state']['riichi'] if p!=seat};reference=heuristic_index(obs,actions,features)
    eligible=set(guarded_indices(obs,actions,features))|{i for i,f in enumerate(features) if not f}
    target=features[index];ref=features[reference]
    if not target or not ref:raise ValueError('Ordinary risk discards require actual native features')
    unsafe=len(threats-set(target['safe']));ref_unsafe=len(threats-set(ref['safe']))
    fold=bool(threats) and min(f['sh'] for f in features if f)>=1
    reason='allowed'
    if index not in eligible:
        if target['sh']>ref['sh']:reason='blocked_shanten_increase'
        elif fold and unsafe>ref_unsafe:reason='blocked_rule_risk'
        else:raise ValueError('Unexplained serving-guard exclusion')
    return dict(teacher_target_allowed=index in eligible,reason=reason,opponent_riichi=len(threats),
        teacher_shanten=target['sh'],reference_shanten=ref['sh'],teacher_unsafe_threat_count=unsafe,
        reference_unsafe_threat_count=ref_unsafe,fold_rule_active=fold,
        feature_safe_to_all_threats=unsafe==0,reference_key=keys[reference])
