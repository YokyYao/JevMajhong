"""Audit fixed RI calibration/margin and loss-mix metrics on saved snapshots."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read,sha
from tools.curriculum_riichi_mechanisms import score_rows,update_mass,summarize_mass


def audit(registered,config,out):
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS')
        or socket.gethostname().startswith(('lo-','login'))):raise RuntimeError('Bounded scheduler GPU required')
    reg=read(registered);c=reg['curriculum'];fit=ROOT/c['out'];cfg=read(config)
    if cfg!=dict(method='fixed_riichi_snapshot_mechanisms_v1',steps=[0,76,152,228],class_threshold=.5,
        calibration_bins=10,mix_extreme_bounds=[.25,.75],no_threshold_search=True,no_model_or_policy_changes=True):
        raise ValueError('Prospective diagnostic configuration changed')
    if not all(sha(p)==h for p,h in reg['inputs'].items()):raise ValueError('Active registered inputs changed')
    sys.path.insert(0,reg['source']);sys.path.insert(0,reg['training_config']['upstream_source'])
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.training.base_lora import package_digest
    if package_digest(Path(reg['source'])/'jevmahjong')!=reg['source_sha256']:raise ValueError('Pinned source changed')
    data=read(ROOT/c['data_root']/'data-ready.json');spec=data['stages']['stage1']['validation'];path=ROOT/spec['path']
    if sha(path)!=spec['sha256']:raise ValueError('Validation source changed')
    records=[json.loads(s) for s in path.open()];out.mkdir(parents=True,exist_ok=False)
    def write(p,value):
        with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    observed_train=(fit/'metrics/train_metrics.jsonl').read_text()
    updates=[json.loads(s) for s in observed_train.splitlines() if json.loads(s)['local_step']<=228]
    if [r['local_step'] for r in updates]!=list(range(1,229)):raise ValueError('Missing actual optimizer updates1..228')
    validation=read(fit/'validation-curve.json');points=[p for p in validation if p['step'] in cfg['steps']]
    if [p['step'] for p in points]!=cfg['steps']:raise ValueError('All fixed completed validation points required')
    write(out/'observed-validation-private.json',points)
    with (out/'observed-train-private.jsonl').open('x') as stream:
        for row in updates:stream.write(json.dumps(row)+'\n')
    binding={str(p.resolve()):sha(p) for p in (registered,config,path,Path(__file__),ROOT/'tools/curriculum_riichi_mechanisms.py',
        ROOT/'tools/curriculum_calls_repair.py',ROOT/'tools/audit_curriculum_riichi_execution.py',
        ROOT/'docs/experiments/plans/curriculum-riichi-mechanisms-20261122.md',out/'observed-validation-private.json',out/'observed-train-private.jsonl')}
    summaries={};identities={};all_cases=[]
    for point in points:
        step=point['step'];predpath=fit/'training-run/training_eval'/f'step-{step:06d}'/'development/rows.json'
        predictions=read(predpath);summary,cases=score_rows(records,predictions)
        if summary['pooled']['windows']!=632:raise ValueError('Actual RI validation population changed')
        for source,expected in point['riichi_by_source'].items():
            actual=summary['sources'][source]['confusion']['raw_argmax']
            if any(actual[k]!=expected.get(k,0) for k in ('tp','fn','tn','fp')):raise ValueError('Raw confusion does not reproduce saved validation')
        for item in cases:item['step']=step
        all_cases.extend(cases);summaries[str(step)]=summary;binding[str(predpath.resolve())]=sha(predpath)
        identities[str(step)]=bundle(fit/'training-run/checkpoints'/f'step-{step:06d}')
    mass=update_mass(updates,reg['stage1_loss_weights'],data['stage1_strata']['train'])
    exposure=Counter()
    for row in updates[:152]:exposure.update(row['source_counts'])
    if exposure!=data['stage1_strata']['train']:raise ValueError('Actual first full epoch does not consume each source count once')
    intervals={f'{start+1}-{end}':summarize_mass([r for r in mass if start<r['step']<=end]) for start,end in ((0,76),(76,152),(152,228))}
    intervals['full_epoch1_1-152']=summarize_mass(mass[:152])
    if abs(intervals['full_epoch1_1-152']['aggregate_riichi_positive_fraction']-.5)>1e-12:
        raise ValueError('Complete epoch fails its registered positive/negative RI loss balance')
    for name,values in (('cases-private.jsonl',all_cases),('update-mass-private.jsonl',mass)):
        with (out/name).open('x') as stream:
            for row in values:stream.write(json.dumps(row)+'\n')
    if not all(sha(p)==h for p,h in binding.items()) or not all(sha(p)==h for p,h in reg['inputs'].items()):
        raise ValueError('Frozen snapshots or registered inputs changed')
    for step,identity in identities.items():
        if bundle(fit/'training-run/checkpoints'/f'step-{int(step):06d}')!=identity:raise ValueError('Saved model changed')
    result=dict(status='fixed_cached_riichi_mechanisms_and_actual_update_mass_audited',steps=cfg['steps'],
        snapshots=summaries,actual_loss_weight_intervals=intervals,actual_updates_checked=228,
        actual_first_complete_epoch_source_exposure=dict(exposure),
        no_new_model_inference_labels_games_or_updates=True,no_threshold_selected_or_policy_changed=True,
        correlational_not_a_causal_training_ablation=True,new_raw_cases_and_weights_private=True,final_reserved_grid_unused=True)
    write(out/'binding-private.json',dict(files=binding,checkpoint_identities=identities,job_id=os.environ['SLURM_JOB_ID']))
    write(out/'summary-private.json',result)
    print(json.dumps(dict(status=result['status'],steps=cfg['steps'],actual_updates_checked=228,intervals=intervals)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('registered','config','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();audit(a.registered.resolve(),a.config.resolve(),a.out.resolve())
