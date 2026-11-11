"""Audit paired future averages without counting rollouts as new states."""
import json
from pathlib import Path
import statistics
import torch
from jevmahjong.core.io import digest, request_hash
from jevmahjong.core.scoring import settlement_ledger
from jevmahjong.policy.kan import is_kan_request
from jevmahjong.training.collector import derived_seed
from jevmahjong.training.rewards import hand_outcomes
from jevmahjong.training.utility import hand_utility

METHOD = 'paired_future_ev_v1'


def mean_credit(worlds, row, config, reward):
    seat = row['episode_key'][1]
    values = {name:[hand_utility(w[name]['mjai_log'], row['hand_index'], seat,
        row['draws_before'], config['fast_gamma'], reward) for w in worlds] for name in ('control','branch')}
    return {name:{key:statistics.mean(v[key] for v in items) for key in items[0]}
        for name,items in values.items()}


def validate_rows(payload, seed, config, protocol, reward=None):
    if payload['collection_method'] != METHOD or payload['seed'] != seed:
        raise ValueError('Wrong future-data method or origin wall')
    source = Path(config['source_root']).resolve() / 'walls-private' / f'{seed}.pt'
    if (Path(payload['source_path']).resolve() != source
            or payload['source_sha256'] != protocol['source_files'][str(source)]
            or digest(source) != payload['source_sha256']):
        raise ValueError('Original source shard changed')
    original = torch.load(source,map_location='cpu',weights_only=False)
    originals = {tuple(r['episode_key']):r for r in original['rows']}
    controls = {c['seat']:c for c in original['controls']}
    games = {(g['seed'],g['seat'],g['replica']):g for g in original['games']}
    expanded = bool(config.get('expand_from')); parents = {}
    if expanded:
        parent = Path(config['parent_source_root']).resolve()/'walls-private'/f'{seed}.pt'
        if (Path(payload['parent_source_path']).resolve()!=parent
                or payload['parent_source_sha256']!=protocol['source_files'][str(parent)]
                or digest(parent)!=payload['parent_source_sha256']):raise ValueError('Parent future shard changed')
        parents={tuple(r['episode_key']):r for r in torch.load(parent,map_location='cpu',weights_only=False)['rows']}
    expected = {tuple(k) for k in protocol['selected_case_keys'] if k[0] == seed}
    if expanded:
        complete=set();seen=set()
        selected_origins=({tuple(k) for k in protocol['selected_origin_keys']} if 'selected_origin_keys' in protocol else None)
        for parent_row in sorted(parents.values(),key=lambda r:tuple(r['episode_key'])):
            identity=(parent_row['episode_key'][0],parent_row['episode_key'][1],parent_row['request_sha256'])
            if identity in seen:continue
            seen.add(identity)
            if selected_origins is not None and tuple(parent_row['episode_key']) not in selected_origins:continue
            complete.update((*parent_row['episode_key'],a) for a in parent_row['eligible'] if a!=parent_row['teacher_action'])
        if expected!=complete:raise ValueError('Expanded plan omits an eligible alternative')
    keys = [tuple(r['episode_key']) for r in payload['rows']]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError('Missing or duplicate revalued state')
    result = []
    for row in payload['rows']:
        key = tuple(row['episode_key']); origin = tuple(row.get('origin_episode_key',key))
        if (len(origin)!=3 or (expanded and (len(key)!=4 or key[:3]!=origin or key[3]!=row['action']))
                or (not expanded and key!=origin)):raise ValueError('Invalid expanded origin identity')
        old = originals[origin]; game = games[origin]
        if not is_kan_request(row['public_request']) or row['scope'] != 'ordinary':
            raise ValueError('Future data must represent a legal kan window')
        if (row['public_request'] != old['public_request'] or request_hash(row['public_request']) != row['request_sha256']
                or row['request_sha256'] != old['request_sha256']
                or row['action_keys'] != old['action_keys'] or row['eligible'] != old['eligible']
                or (not expanded and row['category'] != old['category']) or row['hand_index'] != old['hand_index']
                or row['draws_before'] != old['draws_before']
                or not torch.equal(row['decide'],old['decide']) or not torch.equal(row['options'],old['options'])):
            raise ValueError('Revaluation changed the original model input')
        reference = row['reference']
        if (reference.ndim != 1 or len(reference) != len(row['action_keys'])
                or not torch.isfinite(reference).all() or (reference < 0).any()
                or abs(float(reference.sum())-1.) > 1e-5):
            raise ValueError('Invalid deployed reference probabilities')
        teacher = max(row['eligible'],key=lambda i:float(reference[i]))
        alternate = old['action'] if old['action'] != teacher else old['teacher_action']
        if expanded:
            from jevmahjong.training.sampling import OPERATIONS
            parent_row=parents[origin];alternate=row['action']
            if alternate not in row['eligible']:raise ValueError('Ineligible expanded alternative')
            kinds=[c['type'] for c in row['public_request']['questions']['action']['criteria'].values()]
            category=kinds[alternate] if kinds[alternate] in OPERATIONS else kinds[teacher] if kinds[teacher] in OPERATIONS else 'ordinary'
            if (row['pair_selection']!='all_eligible_alternatives_against_play_v1'
                    or row['category']!=category or row['teacher_action']!=parent_row['teacher_action']
                    or row['prefix_events']!=parent_row['prefix_events']
                    or not torch.equal(row['reference'],parent_row['reference'])):
                raise ValueError('Expanded alternative differs from canonical parent')
        split = 'held' if (seed-config['training_seed_start']) % 4 == 3 else 'train'
        if (row['teacher_action'] != teacher or row['action'] != alternate or alternate == teacher
                or alternate not in row['eligible'] or row['source_split'] != split
                or row['source_state_digest'] != game['intervention_state_digest']):
            raise ValueError('Revalued pair or whole-wall partition changed')
        worlds = row['counterfactual_worlds']
        if len(worlds) != config['worlds']:raise ValueError('Incomplete paired future count')
        prefix = row['prefix_events']; source_log = controls[key[1]]['mjai_log']
        if not 0 < prefix <= len(source_log):raise ValueError('Invalid native prefix length')
        for index,world in enumerate(worlds):
            if world['future_seed'] != derived_seed(config['seed'],seed,origin[1],origin[2],f'future-{index}'):
                raise ValueError('Future seed changed or was selected after outcomes')
            if expanded:
                parent_world=parent_row['counterfactual_worlds'][index]
                if (world['future_seed']!=parent_world['future_seed'] or world['future_wall_sha256']!=parent_world['future_wall_sha256']
                        or world['control']!=parent_world['control']):raise ValueError('Expanded pair changed its parent future or control')
                if alternate==parent_row['action'] and world['branch']!=parent_world['branch']:
                    raise ValueError('Reused parent branch changed')
            wall_hash = world['future_wall_sha256']
            if not isinstance(wall_hash,str) or len(wall_hash) != 64:
                raise ValueError('Paired future wall identity missing')
            for name in ('control','branch'):
                arm=world[name];log=arm['mjai_log']
                if log[:prefix] != source_log[:prefix] or not arm['stopped_after_intervention_hand']:
                    raise ValueError('Future branch changed its historical prefix')
                hands=hand_outcomes(log)
                if not hands[row['hand_index']]['complete'] or sum(h['complete'] for h in hands) != row['hand_index']+1:
                    raise ValueError('Future branch must stop after the selected hand')
                if settlement_ledger(log,arm['scores_at_stop']) != arm['point_ledger']:
                    raise ValueError('Future branch point ledger changed')
        original_means=mean_credit(worlds,row,config,config['reward'])
        if original_means != row['training_credit']:
            raise ValueError('Future mean credit differs from actual native logs')
        if (abs(row['training_advantage']-(original_means['branch']['utility']-original_means['control']['utility'])) > 1e-12
                or abs(row['fast_advantage']-(original_means['branch']['fast_reward']-original_means['control']['fast_reward'])) > 1e-12):
            raise ValueError('Future mean advantage changed')
        selected = dict(row)
        if reward is not None:
            means=mean_credit(worlds,row,config,reward)
            selected.update(training_advantage=means['branch']['utility']-means['control']['utility'],training_credit=means)
        # The audit consumed private rollout records. The fitter needs only the
        # unchanged public representation and one mean target per original pair.
        selected.pop('counterfactual_worlds')
        result.append(selected)
    return result
