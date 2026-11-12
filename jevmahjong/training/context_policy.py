"""Distill complete option utilities into a policy retaining the frozen prior."""
import copy
import math
import random
from collections import defaultdict
from pathlib import Path
import torch
from jevmahjong.core.io import digest, write
from jevmahjong.policy.kan_context import VERSION, feature_names, feature_vectors
from .context_fit import prepare
from .context_network import make_network
from .provenance import verify
from .sampling import state_key, weights


def pack_options(rows, pair_weights=None, feature_version=VERSION):
    """Require every eligible alternative, rather than treating missing Q as zero."""
    grouped=defaultdict(list)
    for index,row in enumerate(rows):grouped[state_key(row)].append((index,row))
    result=[]
    for key,items in grouped.items():
        first=items[0][1];eligible=list(first['eligible']);teacher=first['teacher_action']
        if len(set(eligible))!=len(eligible) or teacher not in eligible:raise ValueError('Invalid eligible actions')
        seen=set();utility={teacher:0.}
        for _,row in items:
            if (row['teacher_action']!=teacher or row['eligible']!=eligible
                    or row['public_request']!=first['public_request']
                    or not torch.equal(row['reference'],first['reference'])
                    or row['action'] in seen or row['action']==teacher):
                raise ValueError('Conflicting complete-option state')
            seen.add(row['action']);utility[row['action']]=row['training_advantage']
        if seen!=set(eligible)-{teacher}:raise ValueError('Complete eligible option utilities required')
        q=first['reference'][eligible].double();q=q/q.sum()
        y=torch.tensor([utility[a] for a in eligible],dtype=torch.float64)
        if (not torch.isfinite(q).all() or (q<0).any() or not torch.isfinite(y).all()
                or eligible[int(q.argmax())]!=teacher):raise ValueError('Invalid reference or utilities')
        features=torch.tensor(feature_vectors(first['public_request'],feature_version),dtype=torch.float64)[eligible]
        result.append({'key':key,'features':features,'utility':y,'log_prior':q.clamp_min(1e-30).log(),
            'teacher':eligible.index(teacher),'weight':sum(pair_weights[i] for i,_ in items) if pair_weights is not None else 1.})
    return result


def policy_logits(net,group,strength):
    if not math.isfinite(strength) or strength<=0:raise ValueError('Positive prior strength required')
    return group['log_prior']+net(group['features']).flatten()/strength


def target(group,strength):
    return (group['log_prior']+group['utility']/strength).softmax(-1)


@torch.no_grad()
def policy_profile(net,groups,sources,strength):
    result=[]
    for source in sources:
        selected=[g for g in groups if source['seed_start']<=g['key'][0]<source['seed_start']+source['seeds']]
        if not selected:raise ValueError('Every source needs held states')
        kl=[];regret=[];gain=[];changes=0
        for g in selected:
            z=policy_logits(net,g,strength);p=target(g,strength);choice=int(z.argmax())
            kl.append(float((p*(p.clamp_min(1e-300).log()-z.log_softmax(-1))).sum()))
            regret.append(float(g['utility'].max()-g['utility'][choice]));gain.append(float(g['utility'][choice]))
            changes+=int(choice!=g['teacher'])
        result.append({'source':source['path'],'states':len(selected),'policy_kl':sum(kl)/len(kl),
            'label_regret':sum(regret)/len(regret),'mean_label_utility_gain_vs_play':sum(gain)/len(gain),
            'changed_argmax_states':changes,'aggregation':'equal public-state means; sampled current-hand utilities'})
    return result


def selection_profile(profiles, mode):
    if not profiles or any(p['states'] <= 0 for p in profiles):
        raise ValueError('Nonempty validation states required')
    if mode == 'latest_source': return profiles[-1]
    if mode != 'all_sources': raise ValueError('Unknown validation selection')
    count = sum(p['states'] for p in profiles)
    return {'source':'all_sources','states':count,
        **{k:sum(p[k]*p['states'] for p in profiles)/count for k in
            ('policy_kl','label_regret','mean_label_utility_gain_vs_play')},
        'changed_argmax_states':sum(p['changed_argmax_states'] for p in profiles),
        'aggregation':'equal public-state means across all validation sources'}


def fit(cfg):
    strength=cfg['reference_prior_strength']
    version=cfg.get('feature_version',VERSION);names=feature_names(version)
    selection=cfg.get('selection_validation','latest_source')
    if not math.isfinite(strength) or strength<=0:raise ValueError('Positive prior strength required')
    out,protocol,parts,hashes,sources,heads=prepare(cfg);torch.manual_seed(cfg['seed'])
    w,coverage=weights(parts['train'],cfg['action_mix'],cfg['max_example_weight'],cfg.get('context_mix'),True)
    train=pack_options(parts['train'],w,version);held=pack_options(parts['held'],feature_version=version)
    mean=sum(g['weight'] for g in train)/len(train)
    for g in train:g['weight']=min(cfg['max_example_weight'],g['weight']/mean)
    coverage.update(policy_loss_unit='one complete eligible action set per state',
        policy_state_weight_cap=cfg['max_example_weight'],policy_training_states=len(train))
    write(out/'data-coverage.json',coverage)
    net=make_network(cfg['hidden'],len(names));optimizer=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    curve=[];updates=0;stale=0;reason='epoch_limit'
    @torch.no_grad()
    def save(epoch):
        prof=policy_profile(net,held,sources,strength);checkpoint=out/f'{cfg["model_id"]}-e{epoch:02d}.pt'
        selected_profile=selection_profile(prof,selection)
        context={'version':version,'names':list(names),'model_kind':'mlp','hidden':cfg['hidden'],
            'reference_prior_strength':strength,'state_dict':{k:v.detach().clone() for k,v in net.state_dict().items()}}
        torch.save({'model_id':cfg['model_id'],'model_files':protocol['model_files'],'base_and_LoRA_frozen':True,
            'heads':{s:copy.deepcopy(h.state_dict()) for s,h in heads.items()},'trained_scopes':['ordinary'],
            'decision_filter':'kan_windows','kan_context':context,'epoch':epoch},checkpoint)
        item={'epoch':epoch,'held_loss':selected_profile['label_regret'],'validation_sources':prof,
            'validation_selection':selected_profile,
            'checkpoint':str(checkpoint.resolve()),'sha256':digest(checkpoint)}
        curve.append(item);return item
    def selection_key(item):return item['held_loss'],item['validation_selection']['policy_kl']
    best=save(0);baseline=best['held_loss']
    for epoch in range(1,cfg['epochs']+1):
        order=list(range(len(train)));random.Random(cfg['seed']+epoch).shuffle(order)
        for start in range(0,len(order),cfg['batch']):
            groups=[train[i] for i in order[start:start+cfg['batch']]]
            losses=[-g['weight']*(target(g,strength)*policy_logits(net,g,strength).log_softmax(-1)).sum() for g in groups]
            loss=torch.stack(losses).sum()/sum(g['weight'] for g in groups)
            if not torch.isfinite(loss):raise ValueError('Nonfinite policy loss')
            optimizer.zero_grad();loss.backward();gradient=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            if not torch.isfinite(gradient):raise ValueError('Nonfinite policy gradient')
            optimizer.step();updates+=1
        item=save(epoch)
        if selection_key(item)<selection_key(best):best=item;stale=0
        else:stale+=1
        write(out/'status.json',{'status':'training','updates':updates,'curve':curve,'selected':best})
        if stale>=cfg['patience_epochs']:reason='held_policy_regret_and_kl_stalled';break
    verify(protocol)
    for path,h in hashes.items():
        if digest(Path(path))!=h:raise ValueError('Input changed during policy fit')
    result={'status':'trained','model_id':cfg['model_id'],'config':cfg,'selected':best,'curve':curve,
        'fit_method':'complete-option-reference-prior-residual-policy-distillation',
        'loss_definition':'state-weighted cross entropy to softmax(log prior + sampled utility / prior strength)',
        'selection_metric':selection+' state-mean label regret; policy KL breaks ties',
        'selection_validation':selection,'optimizer_updates':updates,'stop_reason':reason,
        'train_rows':len(parts['train']),'held_rows':len(parts['held']),'train_states':len(train),'held_states':len(held),
        'parameter_count':sum(p.numel() for p in net.parameters()),'baseline_held_loss':baseline,
        'held_label_policy_gain_improved':best['held_loss']<baseline,
        'selected_is_starting_model':best['epoch']==0,'data':[s['path'] for s in sources],
        'data_sources':sources,'data_sha256':hashes,'human_test_opened':False,'deployed':False,'strength_improvement_proven':False}
    write(out/'result.json',result);write(out/'status.json',result)
