"""Full-training-source descent directions and unchanged placement guards."""
import math
import copy


def updated_head_metadata(parent,head_state,step):
    metadata=copy.deepcopy(parent)
    metadata.update(head=head_state,optimizer_step=step,
        training_backend='full_batch_constrained_existing_head',formal_parent_step=2088)
    return metadata


def projected_descent(gradients,masses,*,margin=.01,max_cycles=2048,tolerance=1e-8):
    import torch
    if set(gradients)!=set(masses) or abs(sum(masses.values())-1.)>1e-9:
        raise ValueError('Complete normalized training-source masses are required')
    total=sum(masses[k]*v.double() for k,v in gradients.items())
    norm=torch.linalg.vector_norm(total)
    if not torch.isfinite(total).all() or norm<=1e-12:
        raise ValueError('No finite nonzero full gradient')
    active={k:v.double()/torch.linalg.vector_norm(v.double()) for k,v in gradients.items()
            if torch.linalg.vector_norm(v.double())>1e-12}
    units=torch.stack(list(active.values()));d0=total/norm
    gram=(units@units.T).cpu().tolist();offset=(units@d0).cpu().tolist()
    alpha=[0.]*len(active);converged=False
    # Hildreth dual coordinate updates for projection onto linear halfspaces.
    for cycle in range(max_cycles):
        change=0.
        for i in range(len(alpha)):
            residual=margin-offset[i]-sum(gram[i][j]*alpha[j] for j in range(len(alpha)))
            value=max(0.,alpha[i]+residual/gram[i][i]);change=max(change,abs(value-alpha[i]));alpha[i]=value
        violation=max(margin-offset[i]-sum(gram[i][j]*alpha[j] for j in range(len(alpha))) for i in range(len(alpha)))
        if not all(math.isfinite(v) and v<1e6 for v in alpha):break
        if violation<=tolerance and change<=tolerance:
            converged=True;break
    if not converged:raise ValueError('Source-descent projection did not converge; do not relax constraints')
    direction=d0+units.T@torch.tensor(alpha,device=units.device,dtype=torch.float64)
    minimum=float((units@direction).min())
    if not torch.isfinite(direction).all() or minimum<margin-tolerance*2:
        raise ValueError('Projected direction failed actual source constraints')
    return direction,{'cycles':cycle+1,'active_sources':list(active),'normalized_margin':margin,
        'minimum_actual_source_inner_product':minimum,'full_gradient_norm':float(norm),
        'direction_norm':float(torch.linalg.vector_norm(direction))}


def guarded_validation(step,current,initial,config,*,exposure_met=True):
    """Same thresholds, cohorts and zero-anchor semantics as the round14 fit."""
    if (config.get('enforce_placement_strata_guard') is not True
            or config.get('enforce_win_stratum_guard') is not False
            or config.get('reference_kl_guard_relative_to_start') is not True
            or config.get('fourth_escape_constraint') is not True):
        raise ValueError('Expected the unchanged round14 guard protocol')
    floor=lambda entry,anchor:max(entry['original_policy'],anchor['candidate'])
    ce=current['original_expert_ce']<=initial['original_expert_ce']+.10
    strata=current['utility_by_stratum'];anchor=initial['utility_by_stratum']
    placement=(strata['third']>=anchor['third']-1e-12 and strata['hold']>=anchor['hold']-.02-1e-12)
    replay=current['kl']['original']['mean']<=initial['kl']['original']['mean']+config['reference_kl_guard']
    native=current['kl']['trailing']['mean']<=config['native_reference_kl_guard']
    leading=current['kl']['leading']['mean']<=config['leading_native_reference_kl_guard']
    escape={k:current['fourth_escape'][k]['candidate']>=floor(current['fourth_escape'][k],initial['fourth_escape'][k])-1e-12
            for k in ('existing','native_fourth')}
    third=current['third_pursuit']['candidate']>=floor(current['third_pursuit'],initial['third_pursuit'])-1e-12
    raw=ce and placement and replay and native and leading and all(escape.values()) and third
    flags={'replay_guard_passed':ce,'placement_strata_guard_passed':placement,
        'reference_guard_passed':replay,'native_reference_guard_passed':step==0 or native,
        'leading_native_reference_guard_passed':step==0 or leading,
        'fourth_escape_guard_passed':step==0 or escape['existing'],
        'native_fourth_escape_guard_passed':step==0 or escape['native_fourth'],
        'third_pursuit_guard_passed':step==0 or third,'exposure_guard_passed':step==0 or exposure_met,
        'win_stratum_guard_passed':True}
    return dict(flags,all_guards_passed=all(flags.values()),raw_constraints_met=raw,
        eligible_nonzero=step>0 and raw and exposure_met,
        fourth_escape_floors={k:floor(current['fourth_escape'][k],initial['fourth_escape'][k]) for k in escape},
        third_pursuit_floor=floor(current['third_pursuit'],initial['third_pursuit']))


def source_loss_values(head,records,features):
    import torch
    from collections import defaultdict
    groups=defaultdict(list)
    with torch.no_grad():
        for row in records:
            q=row['questions']['action'];value=features[row['_meta']['id']]
            logits=head(value['decide'],value['options']).float()
            target=torch.tensor([q['target'][k] for k in value['keys']],device=logits.device,dtype=torch.float32)
            target=target/target.sum();groups[q['src']].append(-(target*logits.log_softmax(-1)).sum())
        return {source:float(torch.stack(values).mean()) for source,values in groups.items()}


def constrained_update(head,records,features,direction,before_losses,*,relative_step=.005,max_halvings=12,tolerance=1e-7):
    import torch
    parameters=list(head.parameters());before=torch.cat([p.detach().reshape(-1) for p in parameters]).clone()
    scale=relative_step*torch.linalg.vector_norm(before.double())/torch.linalg.vector_norm(direction.double())
    def assign(flat):
        offset=0
        with torch.no_grad():
            for p in parameters:
                p.copy_(flat[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
    try:
        for halvings in range(max_halvings+1):
            candidate=(before.double()-scale/(2**halvings)*direction.double()).to(before.dtype)
            assign(candidate);after=source_loss_values(head,records,features)
            if set(after)!=set(before_losses) or not all(math.isfinite(v) for v in after.values()):
                raise ValueError('Source loss coverage changed or became nonfinite')
            if all(after[k]<=before_losses[k]+tolerance for k in after):
                actual=float(torch.linalg.vector_norm(candidate.double()-before.double())/torch.linalg.vector_norm(before.double()))
                if actual<=0:raise ValueError('Projected update rounded to no parameter change')
                return {'accepted':True,'halvings':halvings,'actual_relative_head_l2_change':actual,
                        'before_losses':dict(before_losses),'losses':after}
        assign(before)
        return {'accepted':False,'halvings':max_halvings,'reason':'no finite step preserves every training source',
                'before_losses':dict(before_losses),'losses':dict(before_losses)}
    except BaseException:
        assign(before);raise
