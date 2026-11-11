"""Rebuild only already-revealed dora from chronological public MJAI events.

RiichiEnv0.4.11 initializes replay rounds with their final dora list. Never use
that list's length to determine what a player could see at a decision prefix.
Ambiguous prefixes around a reveal are excluded rather than guessed.
"""
import base64
from collections import defaultdict
import json

from riichienv import Observation
from riichienv.convert import mpsz_to_tid, tid_to_mpsz


class UnsafeDoraPrefix(ValueError):
    pass


def tile_name(value):
    honors={'E':'1z','S':'2z','W':'3z','N':'4z','P':'5z','F':'6z','C':'7z'}
    if value in honors:return honors[value]
    if value.endswith('r'):return '0'+value[1]
    return value


def meld_key(melds):
    return tuple(tuple((m['meld_type'],tuple(sorted(m['tiles']))) for m in seat) for seat in melds)


class DoraChronology:
    def __init__(self, events):
        self.prefixes=defaultdict(set)
        context=None;count=0;melds=[[] for _ in range(4)];doras=[]
        for event in events:
            typ=event['type']
            if typ=='start_kyoku':
                context=({'E':0,'S':1,'W':2,'N':3}[event['bakaze']],event['oya'],event['honba'])
                count=0;melds=[[] for _ in range(4)];doras=[tile_name(event['dora_marker'])]
            elif context is None:
                continue
            elif typ=='dahai':count+=1
            elif typ=='dora':doras.append(tile_name(event['dora_marker']))
            elif typ in ('chi','pon','daiminkan','ankan'):
                tiles=[tile_name(v) for v in event['consumed']]
                if typ!='ankan':tiles.append(tile_name(event['pai']))
                melds[event['actor']].append({'meld_type':{'chi':'Chi','pon':'Pon','daiminkan':'Daiminkan','ankan':'Ankan'}[typ],'tiles':tiles})
            elif typ=='kakan':
                pai=tile_name(event['pai']);same=lambda x:x.replace('0','5')==pai.replace('0','5')
                candidates=[m for m in melds[event['actor']] if m['meld_type']=='Pon' and all(same(t) for t in m['tiles'])]
                if len(candidates)!=1:raise UnsafeDoraPrefix('kakan_source_pon_unresolved')
                candidates[0]['meld_type']='Kakan';candidates[0]['tiles'].append(pai)
            key=(*context,count,meld_key(melds))
            self.prefixes[key].add(tuple(doras))

    def repair(self, obs):
        public=json.loads(base64.b64decode(obs.serialize_to_base64()))
        melds=[[{'meld_type':m['meld_type'],'tiles':[tid_to_mpsz(int(t)) for t in m['tiles']]}
                for m in seat] for seat in public['melds']]
        key=(public['round_wind'],public['oya'],public['honba'],sum(map(len,public['discards'])),meld_key(melds))
        options=self.prefixes.get(key,set())
        if not options:raise UnsafeDoraPrefix('missing_dora_chronology_prefix')
        if len(options)!=1:raise UnsafeDoraPrefix('ambiguous_dora_reveal_prefix')
        expected=next(iter(options))
        previous=tuple(tid_to_mpsz(int(t)) for t in public['dora_indicators'])
        if previous[:len(expected)]!=expected:raise UnsafeDoraPrefix('dora_prefix_conflicts_with_log')
        public['dora_indicators']=[mpsz_to_tid(t) for t in expected]
        encoded=base64.b64encode(json.dumps(public).encode()).decode()
        result=Observation.deserialize_from_base64(encoded)
        return result, {'dora_prefix_checked':1,'future_or_duplicate_dora_removed':len(previous)-len(expected)}
