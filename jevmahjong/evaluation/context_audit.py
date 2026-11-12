"""Recompute a residual fit's held policy, frozen parameters and inference."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import torch
from jevmahjong.core.io import digest,write
from jevmahjong.data.native_pairs import load_sources,select_decisions
from jevmahjong.policy.kan_context import feature_names,feature_vectors,probabilities
from jevmahjong.policy.routing import make_head
from jevmahjong.training.provenance import snapshot,verify
from jevmahjong.training.sampling import state_key

METRICS=('label_regret','policy_kl','mean_label_utility_gain_vs_play',
    'mean_sampled_net_point_gain_per_state','mean_sampled_dealin_change_per_state',
    'mean_sampled_fast_reward_change_per_state')


def require(condition,message):
    if not condition:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def aggregate(records):
    require(bool(records),'Nonempty held states required')
    return {'states':len(records),**{k:sum(r[k] for r in records)/len(records) for k in METRICS},
        'changed_argmax_states':sum(r['changed_argmax_states'] for r in records)}


def held_groups(rows,sources,version,strength):
    grouped=defaultdict(list)
    for row in rows:grouped[state_key(row)].append(row)
    result=[]
    for key,items in grouped.items():
        first=items[0];eligible=first['eligible'];teacher=first['teacher_action']
        require(len(eligible)==len(set(eligible)) and teacher in eligible,'Invalid eligible set')
        require(len(items)==len(eligible)-1 and {r['action'] for r in items}==set(eligible)-{teacher},
            'Complete distinct alternatives required')
        require(all(r['teacher_action']==teacher and r['eligible']==eligible and r['public_request']==first['public_request']
            and torch.equal(r['reference'],first['reference']) for r in items),'Conflicting public-state inputs')
        values={teacher:{k:0. for k in ('utility','net_points','dealin','fast_reward')}}
        for row in items:
            credit=row['training_credit'];values[row['action']]={k:credit['branch'][k]-credit['control'][k] for k in values[teacher]}
            require(abs(values[row['action']]['utility']-row['training_advantage'])<1e-12,'Credit differs from target')
        q=first['reference'].double()[eligible];q=q/q.sum()
        require(eligible[int(q.argmax())]==teacher,'Teacher and actual expanded prior differ')
        y=torch.tensor([values[a]['utility'] for a in eligible],dtype=torch.float64)
        x=torch.tensor(feature_vectors(first['public_request'],version),dtype=torch.float64)[eligible]
        target=(q.clamp_min(1e-30).log()+y/strength).softmax(-1)
        source=next(s['path'] for s in sources if s['seed_start']<=key[0]<s['seed_start']+s['seeds'])
        result.append((first,eligible,x,q,y,target,values,source))
    return result


def check_profiles(actual,registered):
    require(actual['states']==registered['states'] and actual['changed_argmax_states']==registered['changed_argmax_states'],
        'Registered validation state counts differ')
    errors=[abs(actual[k]-registered[k]) for k in METRICS[:3]]
    require(all(math.isfinite(e) and e<1e-10 for e in errors),'Independent held profile differs')
    return max(errors)


@torch.no_grad()
def audit(cfg):
    torch.set_num_threads(4)
    workspace=Path(cfg.get('workspace','.')).resolve()
    resolve=lambda p:(workspace/Path(p)).resolve()
    training_file=resolve(cfg['fit_config']);training=read(training_file)
    require(digest(training_file)==cfg['fit_config_sha256'],'Registered fit configuration changed')
    fit_run=resolve(training['out']);fit=read(fit_run/'result.json');original=read(fit_run/'protocol.json')
    require(fit['status']=='trained' and fit['config']==training and original['config']==training,'Completed registered fit required')
    require(training['fit_model']=='residual_policy' and training['selection_validation']=='all_sources','Residual pooled-validation fit required')
    mode=training['decision_filter'];activation=('defense_windows' if mode=='open_tenpai_riichi' else
        'call_windows' if training['train_scopes']==['calling'] else 'kan_windows')
    require(fit['decision_filter']==activation and fit['trained_scopes']==training['train_scopes'],'Registered model activation differs')
    require(training.get('policy_state_weighting','stratified')==cfg['expected']['policy_state_weighting'],'Registered state weighting differs')
    verify(original);protocol=snapshot(cfg)
    require(protocol['model_files']==original['model_files'],'Audit frozen encoders differ')
    out=Path(cfg['out']);out.mkdir(parents=True,exist_ok=False)
    hashes={str(training_file):digest(training_file)}
    for name in ('result.json','protocol.json','data-coverage.json'):hashes[str(fit_run/name)]=digest(fit_run/name)
    audit_collections=set()
    for directory in cfg['data_audits']:
        directory=resolve(directory);receipt=read(directory/'result.json');registered=read(directory/'protocol.json');verify(registered)
        require(receipt['status']=='passed_complete' and registered['model_files']==original['model_files'],'Completed matching data audit required')
        for path,h in registered['source_files'].items():
            require(digest(Path(path))==h,'Audited data input changed')
            require(path not in hashes or hashes[path]==h,'Conflicting audit input identity');hashes[path]=h
        for name in ('result.json','protocol.json'):hashes[str(directory/name)]=digest(directory/name)
        audit_collections.add(resolve(registered['config']['collection']))
    parts,data_hashes,sources=load_sources([resolve(p) for p in training['data']],original['model_files'],training['reward'])
    parts=select_decisions(parts,mode,training['train_scopes'])
    require(data_hashes==fit['data_sha256'] and sources==fit['data_sources'],'Registered fit data changed')
    for path,h in data_hashes.items():
        require(path not in hashes or hashes[path]==h,'Fit data and audited inputs differ');hashes[path]=h
    if activation=='defense_windows':
        require(len(sources)==1 and training['train_scopes']==['ordinary'],'One registered defense source required')
        options=Path(sources[0]['path']).parent;option_protocol=read(options/'protocol.json')
        parent=resolve(option_protocol['config']['expand_from']);parent_protocol=read(parent/'protocol.json')
        native=resolve(parent_protocol['config']['data'][0])
        require({options,parent,native}<=audit_collections,'Native, parent and complete-option audits required')
        require(option_protocol['config'].get('max_states') is None and parent_protocol['config'].get('max_states') is None,
            'All registered public defense states required')
        require({tuple(k) for k in option_protocol['selected_origin_keys']}=={tuple(k) for k in parent_protocol['selected_case_keys']},
            'Option expansion omitted registered parent states')
    counts={split:len({state_key(r) for r in rows}) for split,rows in parts.items()}
    require(all(counts.values()),'Nonempty whole-wall train and validation required')
    require(not {r['episode_key'][0] for r in parts['train']} & {r['episode_key'][0] for r in parts['held']},'Whole wall crosses split')
    require(sum(s['walls'] for s in sources)==cfg['expected']['wall_groups'],'Registered wall coverage differs')
    require(all(counts[s]==fit[s+'_states'] and len(parts[s])==fit[s+'_rows'] for s in parts),'Fit coverage differs')
    names=feature_names(training['feature_version']);strength=training['reference_prior_strength']
    require(len(names)==cfg['expected']['features'] and strength>0,'Feature schema or prior differs')
    heads={scope:make_head(torch.load(resolve(training[role+'_checkpoint'])/'head.pt',map_location='cpu',weights_only=True),'cpu').eval()
        for scope,role in [('ordinary','discard'),('calling','call')]}
    cached_error=0.
    for rows in parts.values():
        for row in rows:
            q=heads[row['scope']](row['decide'],row['options']).softmax(-1)
            cached_error=max(cached_error,float((q-row['reference']).abs().max()))
            require(torch.allclose(q,row['reference'],atol=2e-5,rtol=2e-4),'Cached frozen head probability differs')
    groups=held_groups(parts['held'],sources,training['feature_version'],strength)
    coverage=read(fit_run/'data-coverage.json')
    if training.get('policy_state_weighting')=='uniform':
        require(coverage['policy_state_weighting']=='uniform' and coverage['outcome_sign_reweighting'] is False
            and coverage['unique_states']==counts['train'],'Uniform public-state coverage differs')
    curves=[];profile_error=inference_error=zero_error=0.
    require([c['epoch'] for c in fit['curve']]==list(range(len(fit['curve']))),'Contiguous saved epochs required')
    for item in fit['curve']:
        path=Path(item['checkpoint']);require(digest(path)==item['sha256'],'Saved checkpoint changed');hashes[str(path)]=item['sha256']
        model=torch.load(path,map_location='cpu',weights_only=True)
        require(model['epoch']==item['epoch'] and model['model_id']==training['model_id']
            and model['model_files']==original['model_files'] and model['base_and_LoRA_frozen'] is True
            and model['trained_scopes']==training['train_scopes'] and model['decision_filter']==activation,'Checkpoint identity differs')
        for scope,head in heads.items():
            expected=head.state_dict();saved=model['heads'][scope]
            require(set(saved)==set(expected) and all(torch.equal(saved[k],v) for k,v in expected.items()),'Original pointer head changed')
        context=model['kan_context'];w=context['state_dict'];h=training['hidden']
        require(context['version']==training['feature_version'] and tuple(context['names'])==names
            and context['hidden']==h and context['reference_prior_strength']==strength,'Context schema differs')
        shapes={'0.weight':(h,len(names)),'0.bias':(h,),'2.weight':(1,h)}
        require(set(w)==set(shapes) and all(tuple(v.shape)==shapes[k] and torch.isfinite(v).all() for k,v in w.items()),'Context parameters differ')
        require(sum(v.numel() for v in w.values())==cfg['expected']['parameters']==fit['parameter_count'],'Parameter count differs')
        if item['epoch']==0:require(not torch.count_nonzero(w['2.weight']),'Starting residual must output zero')
        if item['epoch']==0:
            from jevmahjong.training.context_network import make_network
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(training['seed']);initial=make_network(h,len(names)).state_dict()
            require(all(torch.equal(w[k],v) for k,v in initial.items()),'Starting parameters differ from registered seed')
        by_source=defaultdict(list);selected=item['sha256']==fit['selected']['sha256']
        for row,eligible,x,q,y,target,values,source in groups:
            z=q.clamp_min(1e-30).log()+(torch.tanh(x@w['0.weight'].T+w['0.bias'])@w['2.weight'].T).flatten()/strength
            index=int(z.argmax());choice=eligible[index]
            by_source[source].append({'label_regret':float(y.max()-y[index]),
                'policy_kl':float((target*(target.clamp_min(1e-300).log()-z.log_softmax(-1))).sum()),
                'mean_label_utility_gain_vs_play':float(y[index]),'changed_argmax_states':int(choice!=row['teacher_action']),
                **{'mean_sampled_'+key+'_per_state':values[choice][credit] for key,credit in
                    [('net_point_gain','net_points'),('dealin_change','dealin'),('fast_reward_change','fast_reward')]}})
            if selected or item['epoch']==0:
                actual=torch.tensor(probabilities(row['public_request'],row['reference'].tolist(),context),dtype=torch.float64)
                eligible_p=actual[eligible]/actual[eligible].sum();error=float((eligible_p-z.softmax(-1)).abs().max())
                inference_error=max(inference_error,error)
                require(error<1e-10 and eligible[int(eligible_p.argmax())]==choice,'Actual inference differs from manual policy')
                if item['epoch']==0:zero_error=max(zero_error,float((actual-row['reference'].double()/row['reference'].double().sum()).abs().max()))
        profiles=[{'source':s['path'],**aggregate(by_source[s['path']])} for s in sources]
        require([p['source'] for p in profiles]==[p['source'] for p in item['validation_sources']],'Validation sources differ')
        for actual,registered in zip(profiles,item['validation_sources']):profile_error=max(profile_error,check_profiles(actual,registered))
        combined=aggregate([r for records in by_source.values() for r in records])
        profile_error=max(profile_error,check_profiles(combined,item['validation_selection']))
        require(abs(combined['label_regret']-item['held_loss'])<1e-10,'Held selection loss differs')
        curves.append({'epoch':item['epoch'],**combined,'validation_sources':profiles})
    require(zero_error<1e-12,'Starting model does not preserve original probabilities')
    best=min(fit['curve'],key=lambda c:(c['held_loss'],c['validation_selection']['policy_kl']))
    require(best==fit['selected'],'Selected checkpoint differs from registered selection')
    chosen=curves[best['epoch']];improved=chosen['label_regret']<curves[0]['label_regret']
    require(fit['held_label_policy_gain_improved']==improved and fit['selected_is_starting_model']==(best['epoch']==0),'Held gain flags differ')
    require(fit['optimizer_updates']==(len(fit['curve'])-1)*math.ceil(counts['train']/training['batch']),'Optimizer update count differs')
    require(len(fit['curve'])<=training['epochs']+1,'Registered epoch limit exceeded')
    stale=0;best_so_far=fit['curve'][0]
    for item in fit['curve'][1:]:
        require(stale<training['patience_epochs'],'Fit continued beyond registered early stop')
        if (item['held_loss'],item['validation_selection']['policy_kl'])<(best_so_far['held_loss'],best_so_far['validation_selection']['policy_kl']):
            best_so_far=item;stale=0
        else:stale+=1
    require(fit['stop_reason']==('held_policy_regret_and_kl_stalled' if stale>=training['patience_epochs'] else 'epoch_limit'),'Stop reason differs')
    protocol['source_files']=hashes;write(out/'protocol.json',protocol);verify(protocol);verify(original)
    for path,h in hashes.items():require(digest(Path(path))==h,'Fit audit input changed')
    result={'status':'passed_complete','model_id':training['model_id'],'checkpoint_sha256':best['sha256'],
        'fit_result_sha256':digest(fit_run/'result.json'),'selected':chosen,'curves':curves,'checkpoints_audited':len(curves),
        'activation':activation,'train_pairs':len(parts['train']),'held_pairs':len(parts['held']),
        'train_states':counts['train'],'held_states':counts['held'],'frozen_original_heads':True,'whole_wall_disjoint':True,
        'max_cached_head_probability_error':cached_error,'max_profile_recomputation_error':profile_error,
        'max_actual_inference_error':inference_error,'max_zero_probability_error':zero_error,
        'eligible_for_development':improved and chosen['mean_sampled_net_point_gain_per_state']>0,
        'independent_strength_proven':False,'new_full_matches':0}
    write(out/'result.json',result);write(out/'status.json',result);return result


def checked_receipt(config_path,fit_run,fit,model_files):
    """Bind development to the audited fit, inputs, config and selected weights."""
    config_path=Path(config_path).resolve();registered=read(config_path)
    root=Path(registered['out']).resolve();protocol=read(root/'protocol.json');result=read(root/'result.json')
    require(protocol['config']==registered and protocol['model_files']==model_files,'Fit audit registration differs')
    verify(protocol)
    fit_run=Path(fit_run).resolve();fit_file=fit_run/'result.json'
    require(result['status']=='passed_complete' and result['model_id']==fit['model_id']
        and result['checkpoint_sha256']==fit['selected']['sha256']
        and result['fit_result_sha256']==digest(fit_file)
        and protocol['source_files'].get(str(fit_file))==digest(fit_file),'Audited fit identity differs')
    require(result['activation']==fit['decision_filter'] and result['frozen_original_heads'] is True
        and result['whole_wall_disjoint'] is True,'Audited fit scope differs')
    for path,h in protocol['source_files'].items():require(digest(Path(path))==h,'Audited fit input changed before evaluation')
    receipt={str(path):digest(path) for path in (config_path,root/'protocol.json',root/'result.json')}
    return result,receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    cfg=json.loads(parser.parse_args().config.read_text());result=audit(cfg)
    print(json.dumps({k:v for k,v in result.items() if k!='curves'}))


if __name__=='__main__':main()
