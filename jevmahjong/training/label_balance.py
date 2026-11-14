"""Balance gradient mass by audited label source, without copying positions."""
from collections import Counter
import json
import math
from pathlib import Path

LABELS={'expert_replay','paired_future_masked'}


def source_weights(path,mass,expected_records):
    if (not isinstance(mass,dict) or set(mass)!=LABELS or
            any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in mass.values())
            or abs(sum(mass.values())-1)>1e-10):raise ValueError('Two positive normalized label-source masses required')
    counts=Counter()
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():continue
            r=json.loads(line);label=r['_meta']['label_type'];q=r['questions']['action']
            if label not in LABELS or (q.get('src')=='paired_future_masked')!=(label=='paired_future_masked'):
                raise ValueError('Raw label-source declaration differs from actual question')
            counts[label]+=1
    total=sum(counts.values())
    if total!=expected_records or set(counts)!=LABELS:raise ValueError('Complete prepared label-source coverage required')
    weights={k:mass[k]*total/counts[k] for k in LABELS}
    if max(weights.values())>8:raise ValueError('Requested label-source gradient mass exceeds eightfold cap')
    return dict(records=total,counts=dict(counts),target_mass=mass,loss_weights=weights,
        actual_weighted_mass={k:counts[k]*weights[k]/total for k in LABELS},
        mean_example_weight=sum(counts[k]*weights[k] for k in LABELS)/total,
        copies_added=0,reward_sign_selection=False,validation_weighted=False)


def weighted_loss(logits,question,device,weights):
    from .base_lora import question_loss
    label='paired_future_masked' if question.get('src')=='paired_future_masked' else 'expert_replay'
    return weights[label]*question_loss(logits,question,device)
