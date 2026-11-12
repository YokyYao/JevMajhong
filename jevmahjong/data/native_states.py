"""Validate unlabeled native decision inputs without inventing branch outcomes."""
import torch
from jevmahjong.core.io import request_hash
from jevmahjong.core.scoring import settlement_ledger
from jevmahjong.policy.defense import is_defense_request,request_eligible

METHOD='native_defense_states_v1'


def decision_records(payload):
    return payload['decisions'] if payload.get('collection_method')==METHOD else payload['games']


def validate(payload,seed,config):
    if payload.get('collection_method')!=METHOD or payload['seed']!=seed:raise ValueError('Wrong native-state source')
    controls={r['seat']:r for r in payload['controls']}
    if len(payload['controls'])!=4 or len(controls)!=4 or set(controls)!=set(range(4)):raise ValueError('Four native controls required')
    for game in controls.values():
        ledger=settlement_ledger(game['mjai_log'],[25000+sum(v.values()) for v in game['all_point_components']])
        if (game['seed']!=seed or not game['frozen_control'] or not game['settlement_ledger_passed']
                or ledger['components']!=game['all_point_components']
                or ledger['components'][game['seat']]!=game['point_components']
                or ledger['final_scores'][game['seat']]!=game['score']):raise ValueError('Native control ledger differs')
    decisions={tuple(r['episode_key']):r for r in payload['decisions']}
    if len(decisions)!=len(payload['decisions']):raise ValueError('Duplicate native decision metadata')
    seen=set();public=set()
    for row in payload['rows']:
        key=tuple(row['episode_key']);identity=(key[0],key[1],row['request_sha256'])
        if key in seen or identity in public or key not in decisions or key[0]!=seed or key[1] not in controls:
            raise ValueError('Duplicate or foreign native state')
        seen.add(key);public.add(identity);meta=decisions[key]
        if (len(key)!=3 or key[2]<0 or (meta['seed'],meta['seat'],meta['replica'])!=key
                or row['public_request']['state']['seat']!=key[1]):raise ValueError('Native decision identity differs')
        if (not is_defense_request(row['public_request']) or row['scope']!='ordinary'
                or request_hash(row['public_request'])!=row['request_sha256']
                or row['eligible']!=request_eligible(row['public_request'])):raise ValueError('Defense input or action set differs')
        q=row['reference'];eligible=row['eligible']
        if (q.ndim!=1 or len(q)!=len(row['action_keys']) or len(q)!=len(row['public_request']['questions']['action']['criteria'])
                or row['decide'].ndim!=1 or row['options'].ndim!=2 or row['options'].shape!=(len(q),len(row['decide']))
                or not torch.isfinite(q).all() or (q<0).any() or abs(float(q.sum())-1)>1e-5
                or not torch.isfinite(row['decide']).all() or not torch.isfinite(row['options']).all()
                or row['teacher_action']!=max(eligible,key=lambda i:float(q[i]))
                or row['action'] not in eligible or row['action']==row['teacher_action']):raise ValueError('Native model inputs or teacher differ')
        if (meta['intervention_step']<0 or meta['intervention_step']>=controls[key[1]]['environment_steps']
                or meta['intervention_state_digest']!=row['source_state_digest']
                or meta['hand_index']!=row['hand_index'] or meta['draws_before']!=row['draws_before']):
            raise ValueError('Native replay metadata differs')
    if set(decisions)!=seen:raise ValueError('Native state coverage differs')
    for seat,control in controls.items():
        selected=[key for key in seen if key[1]==seat]
        if (len(selected)!=control['available_intervention_decisions']
                or {key[2] for key in selected}!=set(range(len(selected)))):raise ValueError('Native control decision coverage differs')
    return payload['rows']
