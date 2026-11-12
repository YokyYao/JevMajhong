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
from jevmahjong.training.call_head import load_inputs, route_groups


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


@torch.no_grad()
def manual_profile(head, groups):
    totals = dict(label_regret=0.,mean_label_utility_gain=0.,mean_sampled_net_point_gain=0.,
        mean_sampled_fast_gain=0.,mean_sampled_dealin_change=0.,changed_argmax_states=0)
    for group in groups:
        row = group['row']; z = head(row['decide'],row['options'])
        choice = max(row['eligible'],key=lambda i:float(z[i]))
        matches = [r for r in group['records'] if r['action']==choice]
        if choice == row['teacher_action']:
            values = dict(utility=0.,net_points=0.,fast_reward=0.,dealin=0.)
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
    return dict(states=len(groups),**{k:v if k=='changed_argmax_states' else v/len(groups) for k,v in totals.items()})


def check_close(a,b):
    if set(a)!=set(b) or any(abs(a[k]-b[k])>1e-8 for k in a): raise ValueError('Saved metric differs from recomputation')


@torch.no_grad()
def manual_route_profile(head, groups):
    result=manual_profile(head,groups);positive=recovered=negative=skipped=0
    for g in groups:
        row=g['row'];criteria=list(row['public_request']['questions']['action']['criteria'].values())
        if criteria[row['teacher_action']]['type']!='none':continue
        z=head(row['decide'],row['options']);choice=max(row['eligible'],key=lambda i:float(z[i]))
        records=[r for r in g['records'] if criteria[r['action']]['type'] in ('chi','pon')]
        deltas={r['action']:(r['training_credit']['branch']['utility']-r['training_credit']['control']['utility'],
            r['training_credit']['branch']['net_points']-r['training_credit']['control']['net_points']) for r in records}
        good={action for action,(utility,points) in deltas.items() if utility>0 and points>0}
        bad=bool(deltas) and all(u<=0 and p<=0 for u,p in deltas.values())
        positive+=bool(good);recovered+=choice in good;negative+=bad;skipped+=bad and criteria[choice]['type']=='none'
    result.update(missed_positive_call_states=positive,recovered_sample_positive_call_states=recovered,
        all_sample_negative_call_states=negative,negative_states_still_skipped=skipped)
    return result


def audit(cfg):
    torch.set_num_threads(4)
    fit_file=Path(cfg['fit_config']).resolve();training=read(fit_file)
    if digest(fit_file)!=cfg['fit_config_sha256']: raise ValueError('Fit configuration changed')
    root=Path(training['out']).resolve();fit=read(root/'result.json');registered=read(root/'protocol.json');verify(registered)
    if fit['status']!='trained' or fit['config']!=training or registered['config']!=training:
        raise ValueError('Complete matching call-head fit required')
    if training['fit_model']!='complete_option_call_head' or training['train_scopes']!=['calling']:
        raise ValueError('Only calling pointer-head repair is supported')
    parts,groups,hashes,sources,anchors,anchor_meta,data_audits=load_inputs(training,registered)
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
    routed={split:route_groups(gs) for split,gs in groups.items()}
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
            z=heads['calling'](row['decide'],row['options'])[row['eligible']].double()
            q=row['reference'][row['eligible']].double();q=q/q.sum()
            divergences.append(float((q*(q.clamp_min(1e-30).log()-z.log_softmax(-1))).sum()))
        drift=sum(divergences)/len(divergences)
        if abs(drift-item['anchor_kl'])>1e-8: raise ValueError('Anchor KL differs')
        curve.append(dict(epoch=item['epoch'],train_policy=train,held_policy=held,anchor_kl=drift,
            held_routes=route_results['held'],train_routes=route_results['train']))
        if best is None:
            if held['changed_argmax_states'] or abs(held['mean_sampled_net_point_gain'])>1e-10:
                raise ValueError('Starting policy differs from Play v1')
            best=item;continue
        improved=(drift<=training['anchor_kl_limit']
            and held['mean_sampled_net_point_gain']>best['held_policy']['mean_sampled_net_point_gain']+1e-12
            and held['mean_label_utility_gain']>fit['curve'][0]['held_policy']['mean_label_utility_gain']+1e-12)
        if improved:best=item;stale=0
        else:stale+=1
    if best!=fit['selected'] or fit['selected_is_starting_model']!=(best['epoch']==0): raise ValueError('Checkpoint selection differs')
    if len(curve)>training['epochs']+1 or fit['optimizer_updates']!=(len(curve)-1)*math.ceil(len(groups['train'])/training['batch']):
        raise ValueError('Epoch or optimizer update count differs')
    eligible=(best['epoch']>0 and best['held_policy']['mean_sampled_net_point_gain']>0
        and best['held_policy']['mean_label_utility_gain']>0 and best['anchor_kl']<=training['anchor_kl_limit'])
    route=best['held_routes'].get('closed_chi_tanyao_gain',{})
    route_verified=route.get('recovered_sample_positive_call_states',0)>=cfg['minimum_held_closed_chi_recoveries']
    eligible=eligible and route_verified
    if fit['held_policy_gain_improved']!=(best['epoch']>0):raise ValueError('Policy improvement flag differs')
    protocol=snapshot(cfg);protocol['source_files']=files
    if protocol['model_files']!=registered['model_files']:raise ValueError('Audit models differ')
    out=Path(cfg['out']);out.mkdir(parents=True,exist_ok=False)
    result=dict(status='passed_complete',selected_sha256=best['sha256'],selected_epoch=best['epoch'],curves=curve,
        train_states=len(groups['train']),held_states=len(groups['held']),anchor_states=len(anchors),
        route_recovery_verified=route_verified,eligible_for_development=eligible,strength_improvement_proven=False)
    write(out/'protocol.json',protocol);write(out/'result.json',result);write(out/'status.json',result)
    return result


def checked_receipt(config_path,fit_run,fit,model_files):
    path=Path(config_path).resolve();cfg=read(path);root=Path(cfg['out']).resolve()
    result=read(root/'result.json');protocol=read(root/'protocol.json');verify(protocol)
    if (result['status']!='passed_complete' or protocol['config']!=cfg or protocol['model_files']!=model_files
            or read(cfg['fit_config'])!=fit['config'] or Path(fit['config']['out']).resolve()!=Path(fit_run).resolve()
            or digest(Path(cfg['fit_config']))!=cfg['fit_config_sha256']
            or result['selected_sha256']!=fit['selected']['sha256']):raise ValueError('Matching call-head audit required')
    for file,sha in protocol['source_files'].items():
        if digest(Path(file))!=sha:raise ValueError('Audited fit input changed')
    return result,dict(config_sha256=digest(path),protocol_sha256=digest(root/'protocol.json'),result_sha256=digest(root/'result.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    result=audit(read(p.parse_args().config));print(json.dumps({k:v for k,v in result.items() if k!='curves'}))


if __name__=='__main__': main()
