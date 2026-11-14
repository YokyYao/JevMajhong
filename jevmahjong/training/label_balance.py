"""Balance gradient mass by audited label source, without copying positions."""
from collections import Counter
import json
import math
from pathlib import Path

LABELS={'expert_replay','paired_future_masked'}
PHASES={'closed_discard','open_discard','call_window','kan_window'}
PHASE_PREFIX='paired_future_masked:'


def source_weights(path,mass,expected_records,MC_phase_mass=None):
    if (not isinstance(mass,dict) or set(mass)!=LABELS or
            any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in mass.values())
            or abs(sum(mass.values())-1)>1e-10):raise ValueError('Two positive normalized label-source masses required')
    if MC_phase_mass is not None and (not isinstance(MC_phase_mass,dict) or set(MC_phase_mass)!=PHASES
            or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in MC_phase_mass.values())
            or abs(sum(MC_phase_mass.values())-1)>1e-10):
        raise ValueError('Four positive normalized MC phase masses required')
    counts=Counter();phases=Counter()
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():continue
            r=json.loads(line);label=r['_meta']['label_type'];q=r['questions']['action']
            if isinstance(q.get('src'),str) and q['src'].startswith(PHASE_PREFIX):
                raise ValueError('Raw label-source declaration contains a loss-only phase tag')
            if label not in LABELS or (q.get('src')=='paired_future_masked')!=(label=='paired_future_masked'):
                raise ValueError('Raw label-source declaration differs from actual question')
            counts[label]+=1
            if MC_phase_mass is not None and label=='paired_future_masked':phases[public_phase(r)]+=1
    total=sum(counts.values())
    if total!=expected_records or set(counts)!=LABELS:raise ValueError('Complete prepared label-source coverage required')
    weights={k:mass[k]*total/counts[k] for k in LABELS}
    phase_weights=None
    if MC_phase_mass is not None:
        if set(phases)!=PHASES:raise ValueError('Complete MC phase coverage required')
        phase_weights={p:mass['paired_future_masked']*MC_phase_mass[p]*total/phases[p] for p in sorted(PHASES)}
    if max([*weights.values(),*(phase_weights.values() if phase_weights else [])])>8:
        raise ValueError('Requested label-source/phase gradient mass exceeds eightfold cap')
    result=dict(records=total,counts=dict(counts),target_mass=mass,loss_weights=weights,
        actual_weighted_mass={k:counts[k]*weights[k]/total for k in LABELS},
        mean_example_weight=sum(counts[k]*weights[k] for k in LABELS)/total,
        copies_added=0,reward_sign_selection=False,validation_weighted=False)
    if phase_weights is not None:
        applied={p:phases[p]*phase_weights[p]/total for p in phase_weights}
        result.update(MC_phase_counts=dict(phases),target_MC_phase_mass=MC_phase_mass,
            MC_phase_loss_weights=phase_weights,actual_weighted_mass_by_MC_phase=applied,
            training_source_tag_schema='loss_only_source_phase_v1')
        result['actual_weighted_mass']['paired_future_masked']=sum(applied.values())
        result['mean_example_weight']=sum(result['actual_weighted_mass'].values())
    return result


def public_phase(record):
    from jevmahjong.data.base_lora_pool import phase
    actual=phase(record)
    if actual not in PHASES or record['_meta'].get('phase')!=actual:
        raise ValueError('Training phase differs from public legal menu/hand')
    return actual


def tag_training_sources(records,expected_counts):
    """src survives upstream materialization, but is excluded from model input."""
    counts=Counter();pending=[]
    for record in records:
        q=record['questions']['action'];label=record['_meta']['label_type']
        if isinstance(q.get('src'),str) and q['src'].startswith(PHASE_PREFIX):
            raise ValueError('Unexpected existing loss-only phase tag')
        if label not in LABELS or (q.get('src')=='paired_future_masked')!=(label=='paired_future_masked'):
            raise ValueError('Raw source differs before loss-only phase tagging')
        if label=='paired_future_masked':
            name=public_phase(record);counts[name]+=1;pending.append((q,name))
    if dict(counts)!=expected_counts:raise ValueError('Actual phase-tagged training coverage differs')
    for q,name in pending:q['src']=PHASE_PREFIX+name
    return dict(schema='loss_only_source_phase_v1',tagged_MC_phase_counts=dict(counts),
        training_only=True,source_tag_is_model_input=False,validation_weighted=False)


def weighted_loss(logits,question,device,weights,phase_weights=None):
    from .base_lora import question_loss
    src=question.get('src','')
    if isinstance(src,str) and src.startswith(PHASE_PREFIX):
        name=src[len(PHASE_PREFIX):]
        if phase_weights is None or name not in phase_weights:raise ValueError('Unexpected loss-only source phase')
        # Restore the exact masked-label identity before computing CE.
        return phase_weights[name]*question_loss(logits,dict(question,src='paired_future_masked'),device)
    label='paired_future_masked' if src=='paired_future_masked' else 'expert_replay'
    if phase_weights is not None and label=='paired_future_masked':raise ValueError('Missing MC phase loss tag')
    return weights[label]*question_loss(logits,question,device)
