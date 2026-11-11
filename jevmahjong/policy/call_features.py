"""Label-independent, public call-option summaries for expert supervision.

No expected points or fixed call bonus. Shape/material flags do not guarantee
an eventual yaku. Availability/ukeire is deliberately omitted: this compact
state lacks the provenance needed to de-duplicate called tiles in old rivers.
"""
from collections import Counter
import copy

from jevmahjong.policy.routing import is_call_request
from jevmahjong.policy.call_helpers import call_discards, dora_type, tile_ids
from riichienv import MeldType
from jevmahjong.core.state import shanten

VERSION='public-call-option-v1'
NAMES=('is_call','shanten_before','best_shanten_after','shanten_gain','tenpai_after',
       'loses_closed_status','closed_after','secured_dragon_han','secured_wind_han',
       'all_simples_material_after','dora_locked_in_melds','red_locked_in_melds',
       'riichi_threat_count','safe_discard_at_best_shanten')
VECTOR_NAMES=('kind_chi','kind_pon',*NAMES)
SCALES=(1.,1.,1.,6.,6.,3.,1.,1.,1.,3.,2.,1.,6.,3.,3.,1.)
NOTE=(' Call summaries use public information and the best immediate kuikae-legal '
      'discard. Material flags describe current tiles and do not guarantee a completed yaku.')


def option_features(request):
    if not is_call_request(request):raise ValueError('Chi/pon/pass request required')
    state=request['state'];pid=state['seat'];hand=tile_ids(state['hand'])
    own=state['melds'][pid];existing=[tile_ids(m['tiles']) for m in own]
    closed=not any(m['type']!=int(MeldType.Ankan) for m in own)
    before=shanten(tuple(sorted(t//4 for t in hand)))
    indicators=state.get('dora_indicators');dora=None if indicators is None else Counter(dora_type(t) for t in tile_ids(indicators))
    riichi=state.get('riichi');threats=None if riichi is None else [p for p in riichi if p!=pid]
    rivers=[{t//4 for t in tile_ids(r)} for r in state['rivers']]
    valued_winds=None
    if state.get('round_wind') is not None and state.get('dealer') is not None:
        valued_winds=Counter([27+state['round_wind'],27+(pid-state['dealer'])%4])
    def all_simples(tiles):return all(t//4<27 and t//4%9 not in (0,8) for t in tiles)
    output={}
    for key,criterion in request['questions']['action']['criteria'].items():
        call=criterion['type'] in ('chi','pon');melds=list(existing)
        if call:
            melds.append(tile_ids(' '.join([criterion['pai'],*criterion['consumed']])))
            discards=call_discards(hand,criterion)
            distances=[shanten(tuple(sorted(t//4 for t in rest))) for _,rest in discards]
            best=min(distances);choices=[(t,rest) for (t,rest),d in zip(discards,distances) if d==best]
            safe=None if not threats else any(all(t//4 in rivers[p] for p in threats) for t,_ in choices)
        else:
            best=before;choices=[(None,hand)];safe=None
        triplets=[m[0]//4 for m in melds if len({t//4 for t in m})==1]
        flat=[t for m in melds for t in m]
        values={'is_call':call,'shanten_before':before,'best_shanten_after':best,'shanten_gain':before-best,
            'tenpai_after':best==0,'loses_closed_status':call and closed,'closed_after':closed and not call,
            'secured_dragon_han':sum(t>=31 for t in triplets),
            'secured_wind_han':None if valued_winds is None else sum(valued_winds[t] for t in triplets),
            'all_simples_material_after':any(all_simples(rest+flat) for _,rest in choices),
            'dora_locked_in_melds':None if dora is None else sum(dora[t//4] for t in flat),
            'red_locked_in_melds':sum(t in (16,52,88) for t in flat),
            'riichi_threat_count':None if threats is None else len(threats),'safe_discard_at_best_shanten':safe}
        if tuple(values)!=NAMES:raise AssertionError('Call feature schema differs')
        output[key]=values
    return output


def enrich_request(request):
    result=copy.deepcopy(request)
    if not is_call_request(result):return result
    features=option_features(result)
    for key,values in features.items():result['questions']['action']['criteria'][key]['call_features']=values
    if NOTE not in result['questions']['action']['instructions']:
        result['questions']['action']['instructions']+=NOTE
    return result


def feature_vectors(request):
    values=option_features(request);criteria=request['questions']['action']['criteria'];output=[]
    for key in criteria:
        raw=[criteria[key]['type']=='chi',criteria[key]['type']=='pon',*[values[key][n] for n in NAMES]]
        output.append([0. if value is None else float(value)/scale for value,scale in zip(raw,SCALES)])
    return output
