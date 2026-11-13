"""Recompute saved call-head choices and reference drift before full-game tests.

This checks cached-input arithmetic and provenance, not independent policy strength.
"""
import argparse
import json
import math
from pathlib import Path
import torch
from jevmahjong.core.io import digest, write
from jevmahjong.policy.routing import make_head
from jevmahjong.training.provenance import snapshot, verify
from jevmahjong.training.call_head import load_inputs, route_groups, anchor_policy


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def manual_logits(head,row):
    z=head(row['decide'],row['options']).double()
    return z+row['fixed_call_context_logits'] if 'fixed_call_context_logits' in row else z


def native_win_credit(log, hand_index, seat, draws_before, reward):
    """Winner/draw counting independent of the fitter's reward transform."""
    hand=-1;draws=0;points=0;own=False;hit=False;speed=0.;ended=False
    for event in log:
        kind=event['type']
        if kind=='start_kyoku':hand+=1;draws=0
        if hand!=hand_index:continue
        if kind=='tsumo':draws+=1
        elif kind=='reach_accepted' and event['actor']==seat:points-=1000
        elif kind in ('hora','ryukyoku'):
            points+=event['deltas'][seat]
            if kind=='hora':
                if event['actor']==seat:
                    if own or draws<draws_before:raise ValueError('Invalid own win timing')
                    own=True;speed=reward['speed_gamma']**(draws-draws_before)
                if event['actor']!=seat and event['target']==seat:hit=True
        elif kind=='end_kyoku':ended=True;break
    if not ended:raise ValueError('Complete intervention hand required')
    return dict(own_win=float(own),fast_reward=speed,dealin=float(hit),net_points=points,
        utility=float(own)+reward['fast_weight']*speed-reward['dealin_penalty']*hit)


def check_win_targets(training,parts):
    expected={tuple(r['episode_key']):r for rows in parts.values() for r in rows};seen=set();arms=0
    for directory in training['data']:
        for path in Path(directory).glob('walls-private/*.pt'):
            for raw in torch.load(path,map_location='cpu',weights_only=False)['rows']:
                key=tuple(raw['episode_key'])
                if key not in expected:continue
                if key in seen:raise ValueError('Repeated win-target pair')
                seen.add(key);row=expected[key];worlds=raw['counterfactual_worlds']
                for name in ('control','branch'):
                    values=[native_win_credit(w[name]['mjai_log'],row['hand_index'],key[1],row['draws_before'],training['reward']) for w in worlds]
                    means={k:sum(v[k] for v in values)/len(values) for k in values[0]}
                    check_close(means,row['training_credit'][name]);arms+=len(values)
    if seen!=set(expected):raise ValueError('Missing native win-target records')
    return arms


@torch.no_grad()
def manual_profile(head, groups):
    totals = dict(label_regret=0.,mean_label_utility_gain=0.,mean_sampled_net_point_gain=0.,
        mean_sampled_fast_gain=0.,mean_sampled_dealin_change=0.,changed_argmax_states=0)
    win_first='own_win' in groups[0]['credit'][groups[0]['teacher']]
    if win_first:totals['mean_sampled_hand_win_gain']=0.
    for group in groups:
        row = group['row']; z = manual_logits(head,row)
        choice = max(row['eligible'],key=lambda i:float(z[i]))
        matches = [r for r in group['records'] if r['action']==choice]
        if choice == row['teacher_action']:
            values = dict(utility=0.,net_points=0.,fast_reward=0.,dealin=0.)
            if win_first:values['own_win']=0.
        else:
            if len(matches)!=1: raise ValueError('Chosen alternative missing')
            c = matches[0]['training_credit']
            values = {k:c['branch'][k]-c['control'][k] for k in c['control']}
        alternatives = [0.] + [r['training_advantage'] for r in group['records']]
        totals['label_regret'] += max(alternatives)-values['utility']
        for name, field in [('mean_label_utility_gain','utility'),('mean_sampled_net_point_gain','net_points'),
                            ('mean_sampled_fast_gain','fast_reward'),('mean_sampled_dealin_change','dealin')]:
            totals[name] += values[field]
        totals['changed_argmax_states'] += choice != row['teacher_action']
        if win_first:totals['mean_sampled_hand_win_gain']+=values['own_win']
    return dict(states=len(groups),**{k:v if k=='changed_argmax_states' else v/len(groups) for k,v in totals.items()})


def check_close(a,b):
    if set(a)!=set(b) or any(abs(a[k]-b[k])>1e-8 for k in a): raise ValueError('Saved metric differs from recomputation')


@torch.no_grad()
def manual_route_profile(head, groups):
    result=manual_profile(head,groups);positive=recovered=negative=skipped=0
    for g in groups:
        row=g['row'];criteria=list(row['public_request']['questions']['action']['criteria'].values())
        if criteria[row['teacher_action']]['type']!='none':continue
        z=manual_logits(head,row);choice=max(row['eligible'],key=lambda i:float(z[i]))
        records=[r for r in g['records'] if criteria[r['action']]['type'] in ('chi','pon')]
        win_first='own_win' in g['credit'][g['teacher']]
        deltas={}
        for r in records:
            b,c=r['training_credit']['branch'],r['training_credit']['control']
            priority=(b['own_win']-c['own_win'],b['fast_reward']-c['fast_reward']) if win_first else (b['net_points']-c['net_points'],0.)
            deltas[r['action']]=(b['utility']-c['utility'],priority)
        good={action for action,(utility,gain) in deltas.items() if utility>0 and gain>(0.,0.)}
        bad=bool(deltas) and all(u<=0 and gain<=(0.,0.) for u,gain in deltas.values())
        positive+=bool(good);recovered+=choice in good;negative+=bad;skipped+=bad and criteria[choice]['type']=='none'
    result.update(missed_positive_call_states=positive,recovered_sample_positive_call_states=recovered,
        all_sample_negative_call_states=negative,negative_states_still_skipped=skipped)
    return result


@torch.no_grad()
def audit(cfg):
    torch.set_num_threads(4)
    fit_file=Path(cfg['fit_config']).resolve();training=read(fit_file)
    if digest(fit_file)!=cfg['fit_config_sha256']: raise ValueError('Fit configuration changed')
    root=Path(training['out']).resolve();fit=read(root/'result.json');registered=read(root/'protocol.json');verify(registered)
    if fit['status']!='trained' or fit['config']!=training or registered['config']!=training:
        raise ValueError('Complete matching call-head fit required')
    if training['fit_model']!='complete_option_call_head' or training['train_scopes']!=['calling']:
        raise ValueError('Only calling pointer-head repair is supported')
    mode=anchor_policy(training)
    parts,groups,hashes,sources,anchors,anchor_meta,data_audits=load_inputs(training,registered)
    if training.get('call_context_baseline'):
        from jevmahjong.policy.call_baseline import load
        from jevmahjong.policy.kan_context import feature_vectors
        context=load(training,registered['model_files']);w=context['state_dict']
        for rows in [*parts.values(),anchors]:
            for row in rows:
                x=torch.tensor(feature_vectors(row['public_request'],context['version']),dtype=torch.float64)
                bias=(torch.tanh(x@w['0.weight'].T+w['0.bias'])@w['2.weight'].T).flatten()/context['reference_prior_strength']
                if not torch.allclose(bias,row['fixed_call_context_logits'],atol=1e-12,rtol=0):
                    raise ValueError('Fixed calling context arithmetic differs')
    win_arms=check_win_targets(training,parts) if training['reward']['mode']=='own_win_first_v1' else 0
    if hashes!=fit['data_sha256'] or sources!=fit['data_sources'] or anchor_meta!=fit['anchor_metadata'] or data_audits!=fit['data_audit_sha256']:
        raise ValueError('Fit or anchor inputs differ')
    expected=cfg['expected']
    for split in ('train','held'):
        if len(groups[split])!=expected[split+'_states'] or len(parts[split])!=expected[split+'_pairs']:
            raise ValueError('Registered complete-option coverage differs')
        if len(groups[split])!=fit[split+'_states'] or len(parts[split])!=fit[split+'_rows']:
            raise ValueError('Recorded fit coverage differs')
        for group in groups[split]:
            group['records']=[r for r in parts[split]
                if (r['episode_key'][0],r['episode_key'][1],r['request_sha256'])==group['key']]
    if len(anchors)!=expected['anchor_states'] or anchor_meta['held_states_used']!=0 or anchor_meta['labels_used']:
        raise ValueError('Train-only anchor coverage differs')
    if mode=='calling_baseline_kl':
        from jevmahjong.training.sampling import state_key
        keys={state_key(row) for row in anchors}
        labelled={g['key'] for gs in groups.values() for g in gs}
        held_walls={g['key'][0] for g in groups['held']}
        if (len(keys)!=len(anchors) or keys!=set(map(tuple,anchor_meta['keys']))
                or keys & labelled or {key[0] for key in keys} & held_walls
                or anchor_meta.get('fixed_calling_context_applied_once') is not True):
            raise ValueError('Fixed-calling anchor identity or train-only separation differs')
    routed={split:route_groups(gs) for split,gs in groups.items()}
    if training.get('route_loss_weights'):
        raw={g['key']:1. for g in groups['train']}
        for name,value in training['route_loss_weights'].items():
            for g in routed['train'].get(name,[]):raw[g['key']]=max(raw[g['key']],float(value))
        expected_weighting=dict(routes=training['route_loss_weights'],overlap_rule='maximum',
            normalizer=sum(raw.values())/len(raw),states=len(raw),upweighted_states=sum(v>1 for v in raw.values()),
            outcome_based_selection=False,held_weighted=False)
        if fit.get('training_route_weighting')!=expected_weighting:
            raise ValueError('Registered train-only route weighting differs')
    heads={s:make_head(torch.load(Path(training[role+'_checkpoint'])/'head.pt',map_location='cpu',weights_only=True),'cpu').eval()
        for s,role in [('ordinary','discard'),('calling','call')]}
    original={s:{k:v.clone() for k,v in h.state_dict().items()} for s,h in heads.items()}
    files={**hashes,**anchor_meta['source_sha256'],**data_audits,str(fit_file):digest(fit_file)}
    for name in ('result.json','protocol.json','anchor-inputs-private.json'):files[str(root/name)]=digest(root/name)
    if [c['epoch'] for c in fit['curve']]!=list(range(len(fit['curve']))): raise ValueError('Noncontiguous epochs')
    curve=[];best=None;stale=0
    for item in fit['curve']:
        if stale>=training['patience_epochs']: raise ValueError('Training exceeded choice-based early stop')
        path=Path(item['checkpoint']);files[str(path)]=digest(path)
        if files[str(path)]!=item['sha256']: raise ValueError('Checkpoint changed')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if (saved['model_files']!=registered['model_files'] or saved['base_and_LoRA_frozen'] is not True
                or saved['trained_scopes']!=['calling'] or saved['decision_filter']!='all'
                or saved['epoch']!=item['epoch'] or saved['model_id']!=training['model_id']):
            raise ValueError('Checkpoint identity differs')
        if saved.get('call_context_baseline')!=registered.get('call_context_baseline'):
            raise ValueError('Checkpoint fixed calling context differs')
        for scope,head in heads.items():
            if set(saved['heads'][scope])!=set(original[scope]): raise ValueError('Head architecture changed')
            for name,value in saved['heads'][scope].items():
                if value.shape!=original[scope][name].shape or not torch.isfinite(value).all():raise ValueError('Invalid head tensor')
                if (scope=='ordinary' or item['epoch']==0) and not torch.equal(value,original[scope][name]):
                    raise ValueError('Frozen ordinary head or reference initialization changed')
            head.load_state_dict(saved['heads'][scope])
        train=manual_profile(heads['calling'],groups['train']);held=manual_profile(heads['calling'],groups['held'])
        check_close(train,item['train_policy']);check_close(held,item['held_policy'])
        route_results={}
        for split in ('train','held'):
            route_results[split]={name:manual_route_profile(heads['calling'],gs) for name,gs in routed[split].items()}
            if set(route_results[split])!=set(item[split+'_routes']):raise ValueError('Route coverage differs')
            for name,actual in route_results[split].items():check_close(actual,item[split+'_routes'][name])
        divergences=[]
        for row in anchors:
            z=manual_logits(heads['calling'],row)[row['eligible']]
            q=row['reference'][row['eligible']].double();q=q/q.sum()
            divergences.append(float((q*(q.clamp_min(1e-30).log()-z.log_softmax(-1))).sum()))
        drift=sum(divergences)/len(divergences) if divergences else 0.
        if abs(drift-item['anchor_kl'])>1e-8: raise ValueError('Anchor KL differs')
        curve.append(dict(epoch=item['epoch'],train_policy=train,held_policy=held,anchor_kl=drift,
            held_routes=route_results['held'],train_routes=route_results['train']))
        if best is None:
            if held['changed_argmax_states'] or abs(held['mean_sampled_net_point_gain'])>1e-10:
                raise ValueError('Starting policy differs from registered baseline')
            best=item;continue
        if training.get('selection_objective','net_points')=='hand_win':
            priority=lambda p:(p['mean_sampled_hand_win_gain'],p['mean_sampled_fast_gain'])
            improved=(drift<=training['anchor_kl_limit'] and priority(held)>priority(best['held_policy'])
                and held['mean_sampled_hand_win_gain']>=fit['curve'][0]['held_policy']['mean_sampled_hand_win_gain']
                and held['mean_label_utility_gain']>fit['curve'][0]['held_policy']['mean_label_utility_gain']+1e-12
                and held['mean_sampled_dealin_change']<=training['max_held_dealin_increase'])
        else:
            improved=(drift<=training['anchor_kl_limit']
                and held['mean_sampled_net_point_gain']>best['held_policy']['mean_sampled_net_point_gain']+1e-12
                and held['mean_label_utility_gain']>fit['curve'][0]['held_policy']['mean_label_utility_gain']+1e-12)
        if improved:best=item;stale=0
        else:stale+=1
    if best!=fit['selected'] or fit['selected_is_starting_model']!=(best['epoch']==0): raise ValueError('Checkpoint selection differs')
    if mode in ('same_data_no_anchors','calling_baseline_kl') and fit.get('trainable_parameter_count')!=sum(p.numel() for p in heads['calling'].parameters()):
        raise ValueError('Matched comparison parameter count differs')
    if len(curve)>training['epochs']+1 or fit['optimizer_updates']!=(len(curve)-1)*math.ceil(len(groups['train'])/training['batch']):
        raise ValueError('Epoch or optimizer update count differs')
    eligible=(best['epoch']>0 and best['held_policy']['mean_sampled_net_point_gain']>0
        and best['held_policy']['mean_label_utility_gain']>0 and best['anchor_kl']<=training['anchor_kl_limit'])
    if training.get('selection_objective')=='hand_win':
        eligible=(best['epoch']>0 and best['held_policy']['mean_sampled_hand_win_gain']>0
            and best['held_policy']['mean_label_utility_gain']>0 and best['anchor_kl']<=training['anchor_kl_limit']
            and best['held_policy']['mean_sampled_dealin_change']<=training['max_held_dealin_increase'])
    route=best['held_routes'].get('closed_chi_tanyao_gain',{})
    route_verified=route.get('recovered_sample_positive_call_states',0)>=cfg['minimum_held_closed_chi_recoveries']
    require_route=cfg.get('require_route_recovery_for_development',True)
    if not isinstance(require_route,bool) or (not require_route and training.get('selection_objective')!='hand_win'):
        raise ValueError('Diagnostic-only route checks require the explicitly registered hand-win objective')
    eligible=eligible and (route_verified or not require_route)
    if fit['held_policy_gain_improved']!=(best['epoch']>0):raise ValueError('Policy improvement flag differs')
    protocol=snapshot(cfg);protocol['source_files']=files
    if protocol['model_files']!=registered['model_files']:raise ValueError('Audit models differ')
    out=Path(cfg['out']);out.mkdir(parents=True,exist_ok=False)
    result=dict(status='passed_complete',selected_sha256=best['sha256'],selected_epoch=best['epoch'],curves=curve,
        train_states=len(groups['train']),held_states=len(groups['held']),anchor_states=len(anchors),
        selection_objective=training.get('selection_objective','net_points'),
        win_reward_arms_recomputed=win_arms,
        anchor_policy=mode,trainable_parameter_count=sum(p.numel() for p in heads['calling'].parameters()),
        route_recovery_verified=route_verified,route_is_diagnostic_only=not require_route,
        eligible_for_development=eligible,strength_improvement_proven=False)
    write(out/'protocol.json',protocol);write(out/'result.json',result);write(out/'status.json',result)
    return result


def checked_receipt(config_path,fit_run,fit,model_files):
    path=Path(config_path).resolve();cfg=read(path);root=Path(cfg['out']).resolve()
    result=read(root/'result.json');protocol=read(root/'protocol.json');verify(protocol)
    fit_file=Path(fit_run).resolve()/'result.json'
    if (result['status']!='passed_complete' or protocol['config']!=cfg or protocol['model_files']!=model_files
            or read(cfg['fit_config'])!=fit['config'] or Path(fit['config']['out']).resolve()!=Path(fit_run).resolve()
            or digest(Path(cfg['fit_config']))!=cfg['fit_config_sha256']
            or result['selected_sha256']!=fit['selected']['sha256']):raise ValueError('Matching call-head audit required')
    if read(fit_file)!=fit or protocol['source_files'].get(str(fit_file))!=digest(fit_file):
        raise ValueError('Audited fit result identity differs')
    for file,sha in protocol['source_files'].items():
        if digest(Path(file))!=sha:raise ValueError('Audited fit input changed')
    return result,{str(file):digest(file) for file in (path,root/'protocol.json',root/'result.json')}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    result=audit(read(p.parse_args().config));print(json.dumps({k:v for k,v in result.items() if k!='curves'}))


if __name__=='__main__': main()
