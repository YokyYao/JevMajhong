"""Read-only hand-structure diagnostics. No future hand type becomes an input."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mahjong.shanten import Shanten

def tile_index(tile):
    if tile in ('E','S','W','N','P','F','C'):return 27+('E','S','W','N','P','F','C').index(tile)
    if tile.endswith('r'):tile=tile[:-1]
    value=int(tile[0]) or 5
    return {'m':0,'p':9,'s':18,'z':27}[tile[1]]+value-1

def counts(tiles):
    result=[0]*34
    for t in tiles:result[tile_index(t)]+=1
    return result

def distances(tiles):
    c=counts(tiles)
    regular=Shanten.calculate_shanten(c,use_chiitoitsu=False,use_kokushi=False)
    seven=Shanten.calculate_shanten_for_chiitoitsu_hand(c)
    return regular,seven

def remove_tile(hand,tile):
    # Preserve red identity when possible; compare structural identity otherwise.
    chosen=tile if tile in hand else next((x for x in hand if tile_index(x)==tile_index(tile)),None)
    if chosen is None:raise ValueError('Logged action removes an absent tile')
    hand.remove(chosen)

def winning_shapes(path):
    stats=defaultdict(Counter);schema=None
    if not path:return {},None
    for line in Path(path).open():
        game=json.loads(line);schema=schema or list(game)
        events=game.get('mjai_log',game.get('events',game.get('mjai')))
        if events is None:continue
        if isinstance(events,str):events=json.loads(events)
        arm=game.get('arm',game.get('policy','unknown'));seat=game.get('seat',game.get('player_id'))
        hands=None;melds=None;last_offered=None
        for raw in events:
            e=json.loads(raw) if isinstance(raw,str) else raw;kind=e['type']
            if kind=='start_kyoku':
                hands=[list(h) for h in e['tehais']];melds=[[] for _ in range(4)]
            elif kind=='tsumo':hands[e['actor']].append(e['pai'])
            elif kind=='dahai':
                remove_tile(hands[e['actor']],e['pai']);last_offered=e['pai']
            elif kind in ('chi','pon','daiminkan','ankan'):
                for tile in e['consumed']:remove_tile(hands[e['actor']],tile)
                melds[e['actor']].append(kind)
            elif kind=='kakan':
                remove_tile(hands[e['actor']],e['pai']);last_offered=e['pai']
            elif kind=='hora' and e['actor']==seat:
                stats[str(arm)]['controlled_wins']+=1
                if melds[seat]:stats[str(arm)]['with_melds']+=1;continue
                hand=list(hands[seat])
                if e['target']!=seat:hand.append(e.get('pai',last_offered))
                if len(hand)!=14:raise ValueError('Winning hand reconstruction is not14 tiles')
                regular,seven=distances(hand)
                label=('seven_and_regular_ambiguous' if regular==-1 else 'seven_shape_only') if seven==-1 else (
                    'regular_shape' if regular==-1 else 'other_closed_shape')
                stats[str(arm)][label]+=1
    return {k:dict(v) for k,v in stats.items()},schema

def audit(data,output,games=None):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS'),'Scheduled GPU required'
    from jevmahjong.policy.closed import request_eligible
    report={};guard=Counter()
    for split in ('train','validation'):
        total=Counter();classes=defaultdict(Counter)
        for line in (data/f'{split}.jsonl').open():
            row=json.loads(line);state=row['state'];meta=row['_meta'];q=row['questions']['action']
            if meta['phase']!='closed_discard':continue
            tiles=state['hand'].split()
            if len(tiles) not in (13,14) or state['melds'][state['seat']]:continue
            regular,seven=distances(tiles)
            cls='seven_closer' if seven<regular else 'equal_distance' if seven==regular else 'regular_closer'
            total['closed_no_meld_menus']+=1;total[cls]+=1
            if sum(v>=2 for v in counts(tiles))>=5:total['ui_five_pair_trigger_before_action']+=1
            classes[meta['label_type']][cls]+=1
            if meta['label_type']=='expert_replay':
                action=q['criteria'][q['label']]
                if action['type'] in ('dahai','reach') and 'pai' in action:
                    after=list(tiles);remove_tile(after,action['pai']);a,b=distances(after)
                    classes['expert_after']['seven_closer' if b<a else 'equal_distance' if a==b else 'regular_closer']+=1
            if split=='train' and state.get('riichi') is None:guard['menus_with_unknown_riichi_skipped']+=1
            if split=='train' and len(tiles)==14 and state.get('riichi') is not None:
                options=list(q['criteria'].values());eligible=set(request_eligible(row));shapes={}
                for i,c in enumerate(options):
                    if c['type'] not in ('dahai','reach') or 'pai' not in c:continue
                    after=list(tiles);remove_tile(after,c['pai']);shapes[i]=distances(after)
                if shapes:
                    best=min(v[0] for v in shapes.values());regular_best={i for i,v in shapes.items() if v[0]==best}
                    guard['audited_menus']+=1
                    if not regular_best&eligible:guard['all_best_regular_route_options_filtered']+=1
                    if not eligible:raise ValueError('Empty serving guard')
        report[split]=dict(total= dict(total),by_source={k:dict(v) for k,v in classes.items()})
    wins,schema=winning_shapes(games)
    result=dict(status='hand_structure_audit_complete',node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'],
        public_decision_shapes=report,guard=dict(guard),historical_win_shapes=wins,historical_log_schema=schema,
        no_model_loaded=True,no_optimizer_updates=True,
        limitations=['decision shanten categories are not actual winning yaku',
                    'winning shapes may have multiple legal decompositions; ambiguous cases reported separately',
                    'this is descriptive evidence, not a causal test of training mix'])
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--games',type=Path);a=p.parse_args();audit(a.data,a.out,a.games)
