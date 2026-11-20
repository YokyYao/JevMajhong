"""Observe actual accumulation composition; propose sizes without refitting."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import socket

ROOT=Path(__file__).resolve().parents[1]


def suggest_accumulation(records,maximum,minimum_stratum_count,min_tail_fraction=.75):
    """Largest size within the declared range with no tiny final update."""
    for size in range(maximum,max(2,maximum//2)-1,-1):
        updates=math.ceil(records/size);tail=records-(updates-1)*size
        if tail/size>=min_tail_fraction and updates<=minimum_stratum_count:
            return dict(accumulation=size,updates_per_epoch=updates,final_records=tail,
                final_fraction=tail/size,selection='largest_half_to_max_size_with_tail_ge75pct_and_distinct_rare_coverage')
    raise ValueError('No size satisfies the registered count-only constraints')


def audit(registered,out):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-','login'))
    reg=json.loads(registered.read_text());c=reg['curriculum'];fit=ROOT/c['out']
    for path,expected in reg['inputs'].items():
        with Path(path).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected,path
    raw=(fit/'metrics/train_metrics.jsonl').read_text();values=raw.splitlines()
    if raw and not raw.endswith('\n'):values=values[:-1]
    logs=[json.loads(s) for s in values if s];assert [r['local_step'] for r in logs]==list(range(1,len(logs)+1))
    size=c['batch']*c['accumulation'];per_epoch=math.ceil(reg['train_records']/size)
    weights=reg['stage1_loss_weights'];events=[]
    for row in logs:
        if row['local_step']%per_epoch:continue
        counts=row['source_counts'];total=sum(counts.values());mass={k:count*weights[k] for k,count in counts.items()}
        norm=sum(mass.values());shares={k:v/norm for k,v in mass.items()}
        events.append(dict(step=row['local_step'],records=total,source_counts=counts,
            coefficient_mass_shares=shares,riichi_mass_share=sum(v for k,v in shares.items() if k.startswith('riichi_')),
            grad_norm_pre_clip=row['grad_norm'],logged_post_scheduler_next_lr=row['lr']))
    data=json.loads((ROOT/c['data_root']/'data-ready.json').read_text())
    rare=min(data['stage1_strata']['train'][k] for k in ('riichi_yes','riichi_no'))
    proposal=suggest_accumulation(reg['train_records'],c['accumulation'],rare)
    result=dict(status='actual_accumulation_composition_audited',node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'],
        observed_updates=len(logs),registered_records=reg['train_records'],registered_accumulation=size,
        updates_per_epoch=per_epoch,tail_records=reg['train_records']-(per_epoch-1)*size,
        registered_riichi_loss_mass=sum(v for k,v in reg['stage1_loss_mass'].items() if k.startswith('riichi_')),
        completed_epoch_end_events=events,count_only_later_recipe_candidate=proposal,
        next_course_count_only_candidate=suggest_accumulation(20237,128,194),
        current_recipe_modified=False,new_fit_started=False,weight_coefficients_not_gradient_norm_shares=True,
        note='Short groups are normalized by their actual record count in the pinned upstream trainer. These are loss-coefficient composition shares, not measured per-source gradient contributions or proof of the cause of policy oscillation. Candidate sizes are selected from counts only; do not change a frozen active fit or claim an unrun fix improves play.')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registered',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit(a.registered,a.out)
