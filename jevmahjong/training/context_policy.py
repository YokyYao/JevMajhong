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
        win_credit='own_win' in first.get('training_credit',{}).get('control',{})
        credit={teacher:{k:0. for k in ('own_win','net_points','dealin','fast_reward')}} if win_credit else None
        for _,row in items:
            if (row['teacher_action']!=teacher or row['eligible']!=eligible
                    or row['public_request']!=first['public_request']
                    or not torch.equal(row['reference'],first['reference'])
                    or row['action'] in seen or row['action']==teacher):
                raise ValueError('Conflicting complete-option state')
            seen.add(row['action']);utility[row['action']]=row['training_advantage']
            if win_credit:
                c=row['training_credit']
                if c['control']!=first['training_credit']['control']:
                    raise ValueError('Win targets require the same paired control futures')
                if abs(c['branch']['utility']-c['control']['utility']-row['training_advantage'])>1e-12:
                    raise ValueError('Win credit and utility differ')
                credit[row['action']]={k:c['branch'][k]-c['control'][k] for k in credit[teacher]}
        if seen!=set(eligible)-{teacher}:raise ValueError('Complete eligible option utilities required')
        q=first['reference'][eligible].double();q=q/q.sum()
        y=torch.tensor([utility[a] for a in eligible],dtype=torch.float64)
        if (not torch.isfinite(q).all() or (q<0).any() or not torch.isfinite(y).all()
                or eligible[int(q.argmax())]!=teacher):raise ValueError('Invalid reference or utilities')
        features=torch.tensor(feature_vectors(first['public_request'],feature_version),dtype=torch.float64)[eligible]
        result.append({'key':key,'features':features,'utility':y,'log_prior':q.clamp_min(1e-30).log(),
            'teacher':eligible.index(teacher),'weight':sum(pair_weights[i] for i,_ in items) if pair_weights is not None else 1.})
        if win_credit:result[-1]['credit']=[credit[a] for a in eligible]
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
        kl=[];regret=[];gain=[];changes=0;credits=[]
        for g in selected:
            z=policy_logits(net,g,strength);p=target(g,strength);choice=int(z.argmax())
            kl.append(float((p*(p.clamp_min(1e-300).log()-z.log_softmax(-1))).sum()))
            regret.append(float(g['utility'].max()-g['utility'][choice]));gain.append(float(g['utility'][choice]))
            changes+=int(choice!=g['teacher'])
            if 'credit' in g:credits.append(g['credit'][choice])
        result.append({'source':source['path'],'states':len(selected),'policy_kl':sum(kl)/len(kl),
            'label_regret':sum(regret)/len(regret),'mean_label_utility_gain_vs_play':sum(gain)/len(gain),
            'changed_argmax_states':changes,'aggregation':'equal public-state means; sampled current-hand utilities'})
        if credits:
            if len(credits)!=len(selected):raise ValueError('Mixed win and non-win targets')
            result[-1].update({name:sum(c[key] for c in credits)/len(credits) for name,key in
                [('mean_sampled_hand_win_gain_per_state','own_win'),
                 ('mean_sampled_net_point_gain_per_state','net_points'),
                 ('mean_sampled_dealin_change_per_state','dealin'),
                 ('mean_sampled_fast_reward_change_per_state','fast_reward')]})
    return result


def selection_profile(profiles, mode):
    if not profiles or any(p['states'] <= 0 for p in profiles):
        raise ValueError('Nonempty validation states required')
    if mode == 'latest_source': return profiles[-1]
    if mode != 'all_sources': raise ValueError('Unknown validation selection')
    count = sum(p['states'] for p in profiles)
    metrics=['policy_kl','label_regret','mean_label_utility_gain_vs_play']
    if 'mean_sampled_hand_win_gain_per_state' in profiles[0]:
        metrics+=['mean_sampled_hand_win_gain_per_state','mean_sampled_net_point_gain_per_state',
            'mean_sampled_dealin_change_per_state','mean_sampled_fast_reward_change_per_state']
    return {'source':'all_sources','states':count,
        **{k:sum(p[k]*p['states'] for p in profiles)/count for k in metrics},
        'changed_argmax_states':sum(p['changed_argmax_states'] for p in profiles),
        'aggregation':'equal public-state means across all validation sources'}


def win_policy_improved(candidate,best,baseline,risk_limit):
    """Choose sampled own wins, then speed; points do not select weights."""
    p=candidate['validation_selection'];b=best['validation_selection'];start=baseline['validation_selection']
    win='mean_sampled_hand_win_gain_per_state';fast='mean_sampled_fast_reward_change_per_state'
    risk='mean_sampled_dealin_change_per_state';utility='mean_label_utility_gain_vs_play'
    if not all(math.isfinite(q[k]) for q in (p,b,start) for k in (win,fast,risk,utility)):
        return False
    better=(p[win]>b[win]+1e-12 or (abs(p[win]-b[win])<=1e-12 and p[fast]>b[fast]+1e-12))
    return (better and p[win]>=start[win]-1e-12 and p[utility]>start[utility]+1e-12
        and p[risk]<=risk_limit+1e-12)


def training_groups(rows,cfg,version):
    mode=cfg.get('policy_state_weighting','stratified')
    if mode=='public_routes':
        from .call_head import training_weights
        if cfg.get('train_scopes')!=['calling'] or not cfg.get('route_loss_weights'):
            raise ValueError('Explicit public call-route loss weights required')
        groups=pack_options(rows,feature_version=version)
        first={}
        for row in rows:first.setdefault(state_key(row),row)
        routed=[{'key':g['key'],'row':first[g['key']]} for g in groups]
        values,report=training_weights(routed,cfg['route_loss_weights'])
        cap=cfg['max_example_weight']
        if not math.isfinite(cap) or cap<1 or any(w>cap+1e-12 for w in values):
            raise ValueError('Public route weights exceed the registered state cap')
        for group,value in zip(groups,values):group['weight']=value
        return groups,{'policy_state_weighting':mode,'unique_states':len(groups),
            'outcome_sign_reweighting':False,'rows':len(rows),'public_route_loss':report}
    if mode=='uniform':
        groups=pack_options(rows,feature_version=version)
        return groups,{'policy_state_weighting':'uniform','unique_states':len(groups),
            'outcome_sign_reweighting':False,'rows':len(rows)}
    if mode!='stratified':raise ValueError('Unknown policy state weighting')
    pair_weights,coverage=weights(rows,cfg['action_mix'],cfg['max_example_weight'],cfg.get('context_mix'),True)
    groups=pack_options(rows,pair_weights,version)
    mean=sum(g['weight'] for g in groups)/len(groups)
    for g in groups:g['weight']=min(cfg['max_example_weight'],g['weight']/mean)
    return groups,coverage


def fit(cfg):
    strength=cfg['reference_prior_strength']
    version=cfg.get('feature_version',VERSION);names=feature_names(version)
    selection=cfg.get('selection_validation','latest_source')
    objective=cfg.get('selection_objective','utility')
    if objective not in ('utility','hand_win'):raise ValueError('Unknown residual selection objective')
    if objective=='hand_win' and (cfg['train_scopes']!=['calling'] or selection!='all_sources'
            or cfg.get('reward',{}).get('mode')!='own_win_first_v1'
            or not math.isfinite(cfg.get('max_held_dealin_increase',float('nan')))
            or not 0<=cfg['max_held_dealin_increase']<=1):
        raise ValueError('Explicit pure-call win reward, pooled validation and risk cap required')
    activation=('defense_windows' if cfg['decision_filter']=='open_tenpai_riichi' else
        'call_windows' if cfg['train_scopes'] == ['calling'] else 'kan_windows')
    if not math.isfinite(strength) or strength<=0:raise ValueError('Positive prior strength required')
    out,protocol,parts,hashes,sources,heads=prepare(cfg);torch.manual_seed(cfg['seed'])
    train,coverage=training_groups(parts['train'],cfg,version);held=pack_options(parts['held'],feature_version=version)
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
            'heads':{s:copy.deepcopy(h.state_dict()) for s,h in heads.items()},'trained_scopes':cfg['train_scopes'],
            'decision_filter':activation,'kan_context':context,'epoch':epoch},checkpoint)
        item={'epoch':epoch,'held_loss':selected_profile['label_regret'],'validation_sources':prof,
            'validation_selection':selected_profile,
            'checkpoint':str(checkpoint.resolve()),'sha256':digest(checkpoint)}
        curve.append(item);return item
    def selection_key(item):return item['held_loss'],item['validation_selection']['policy_kl']
    best=initial=save(0);baseline=best['held_loss']
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
        improved=(win_policy_improved(item,best,initial,cfg['max_held_dealin_increase'])
            if objective=='hand_win' else selection_key(item)<selection_key(best))
        if improved:best=item;stale=0
        else:stale+=1
        write(out/'status.json',{'status':'training','updates':updates,'curve':curve,'selected':best})
        if stale>=cfg['patience_epochs']:
            reason='held_actual_hand_wins_stalled' if objective=='hand_win' else 'held_policy_regret_and_kl_stalled';break
    verify(protocol)
    for path,h in hashes.items():
        if digest(Path(path))!=h:raise ValueError('Input changed during policy fit')
    result={'status':'trained','model_id':cfg['model_id'],'config':cfg,'selected':best,'curve':curve,
        'decision_filter':activation,'trained_scopes':cfg['train_scopes'],
        'fit_method':'complete-option-reference-prior-residual-policy-distillation',
        'loss_definition':'state-weighted cross entropy to softmax(log prior + sampled utility / prior strength)',
        'selection_metric':('state-mean sampled own wins, then speed, with positive utility and deal-in guard'
            if objective=='hand_win' else selection+' state-mean label regret; policy KL breaks ties'),
        'selection_objective':objective,
        'selection_validation':selection,'optimizer_updates':updates,'stop_reason':reason,
        'train_rows':len(parts['train']),'held_rows':len(parts['held']),'train_states':len(train),'held_states':len(held),
        'parameter_count':sum(p.numel() for p in net.parameters()),'baseline_held_loss':baseline,
        'held_label_policy_gain_improved':best['held_loss']<baseline,
        'selected_is_starting_model':best['epoch']==0,'data':[s['path'] for s in sources],
        'data_sources':sources,'data_sha256':hashes,'human_test_opened':False,'deployed':False,'strength_improvement_proven':False}
    write(out/'result.json',result);write(out/'status.json',result)
