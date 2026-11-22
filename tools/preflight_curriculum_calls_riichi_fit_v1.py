"""Check actual calls menus against the retained loss and metric reconstruction."""
import argparse
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read,sha

PUBLIC_FILES=(
    'tools/curriculum_calls_riichi_fit_v1.py',
    'tools/register_curriculum_calls_riichi_retention_v1.py',
    'tools/train_curriculum_calls_riichi_retention_v1.py',
    'tools/audit_curriculum_calls_riichi_retention_v1.py',
    'tools/preflight_curriculum_calls_riichi_fit_v1.py',
    'configs/training/curriculum-calls-riichi-retention-fit-v1.json',
    'tests/core/test_curriculum_calls_riichi_fit_v1.py',
    'docs/experiments/plans/curriculum-calls-riichi-retention-fit-20261123.md',
)


def preflight(out):
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS')
        or socket.gethostname().startswith(('lo-','login'))):raise RuntimeError('Scheduled GPU required')
    out.mkdir(parents=True,exist_ok=False)
    old_path=ROOT/'runs/curriculum-basic-repair-v1/fit-plan/registered.json';old=read(old_path)
    sys.path.insert(0,old['source']);sys.path.insert(0,old['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from jevany.data import materialize
    from jevmahjong.training.base_lora import target_cross_entropy
    from tools.curriculum_calls_riichi_fit_v1 import require_recipe,profile,COMPONENTS
    from tools.curriculum_calls_riichi_retention_v1 import annotate,parts
    for module in ('tools.register_curriculum_calls_riichi_retention_v1',
                   'tools.train_curriculum_calls_riichi_retention_v1',
                   'tools.audit_curriculum_calls_riichi_retention_v1'):importlib.import_module(module)
    ready_path=ROOT/'runs/curriculum-calls-repair-data-v1/data/data-ready-private.json';data=read(ready_path)
    config=ROOT/'configs/training/curriculum-calls-riichi-retention-fit-v1.json'
    proposal=require_recipe(read(config),data,data['categories']['train'])
    files={str((ROOT/name).resolve()):sha(ROOT/name) for name in PUBLIC_FILES}
    for name in ('runs/curriculum-riichi-class-control-v1/launch-binding-private.json',
                 'runs/curriculum-calls-riichi-retention-v1/launch-binding-private.json',
                 'runs/curriculum-calls-riichi-retention-v1/preflight-v1/binding-private.json'):
        path=ROOT/name;bound=read(path);files.update(bound.get('files',bound));files[str(path)]=sha(path)
    files[str(ready_path)]=sha(ready_path)
    if any(sha(path)!=digest for path,digest in files.items()):raise ValueError('Bound source changed')
    for name in PUBLIC_FILES:
        if name.endswith('.py'):compile((ROOT/name).read_text(),str(ROOT/name),'exec')
    validation_path=ROOT/data['files']['validation']['path']
    if sha(validation_path)!=data['files']['validation']['sha256']:raise ValueError('Actual validation changed')
    files[str(validation_path)]=sha(validation_path)
    validation=[json.loads(line) for line in validation_path.open()];predictions=[];observed={}
    counts=Counter();maximum_error=0.;comparisons=0
    for index,row in enumerate(validation):
        encoded=annotate(materialize(row),row);q=encoded['questions'][0]
        assert len(encoded['questions'])==1
        raw=row['questions']['action'];keys=q['keys'];source=q['src'].removeprefix('curriculum2:')
        z=torch.linspace(-1.3,1.1,len(keys),device='cuda',dtype=torch.float64)
        if index%2:z=-z
        p=z.softmax(0).cpu().tolist();predictions.append(dict(id=row['_meta']['id'],keys=keys,p=p))
        actual=parts(z,q,'cuda');actual={name:float(value) for name,value in actual.items()}
        # This is deliberately a synthetic-score accounting check, not inference.
        if source.startswith('basic:'):
            from tools.curriculum_riichi_control import cached_parts
            expected=dict(cached_parts(dict(raw,src='curriculum1:'+source.removeprefix('basic:')),keys,p,target_cross_entropy),advanced_action_ce=0.)
        else:
            value=target_cross_entropy(raw,keys,p);expected={name:0. for name in COMPONENTS}
            expected.update(objective=value,standard_action_ce=value,advanced_action_ce=value)
        error=max(abs(actual[name]-expected[name]) for name in COMPONENTS)
        if error>5e-6:raise ValueError('Actual loss and cached objective differ')
        maximum_error=max(maximum_error,error);comparisons+=1;counts[source]+=1
        for name in COMPONENTS:observed.setdefault(source,Counter())[name]+=actual[name]
    assert counts==Counter(data['categories']['validation']) and len(validation)==12404
    base=dict(weighted_target_ce=sum(target_cross_entropy(row['questions']['action'],p['keys'],p['p']) for row,p in zip(validation,predictions))/len(validation))
    result=profile(validation,predictions,data['loss_mass'],base,target_cross_entropy)
    profile_error=max(abs(result['source_loss_components'][s][name]-observed[s][name]/counts[s]) for s in counts for name in COMPONENTS)
    assert profile_error<5e-6
    assert abs(result['weighted_class_objective']-sum(data['loss_mass'][s]*observed[s]['objective']/counts[s] for s in counts))<5e-6
    if any(sha(path)!=digest for path,digest in files.items()):raise ValueError('Frozen source changed during preflight')
    summary=dict(status='calls_RI_retention_full_entry_and_cached_metric_preflight_not_fit_admitted',
        actual_validation_menus_checked=comparisons,maximum_per_menu_component_error=maximum_error,
        maximum_aggregate_component_error=profile_error,all_twelve_sources_checked=True,
        objective_and_original_action_ce_separate=True,synthetic_CUDA_logits_not_model_predictions=True,
        conditional_recipe_updates=proposal['final_step'],conditional_recipe_exposures=proposal['record_exposures'],
        language_model_parameters_loaded=False,student_updates=0,parent_admitted=False,
        full_child_registration_or_fit_performed=False,active_and_completed_bound_files_unchanged=True)
    for name,value in [('summary-private.json',summary),('binding-private.json',dict(files=files,job_id=os.environ['SLURM_JOB_ID'],node=socket.gethostname()))]:
        with (out/name).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();preflight(args.out.resolve())
