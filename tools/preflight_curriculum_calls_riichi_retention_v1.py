"""Full prepared-data/input parity and CUDA loss checks; no parent or fit admission."""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read,sha


def preflight(out):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS')
    assert not socket.gethostname().startswith(('lo-','login'))
    out.mkdir(parents=True,exist_ok=False)
    registered=ROOT/'runs/curriculum-basic-repair-v1/fit-plan/registered.json';reg=read(registered)
    sys.path.insert(0,reg['source']);sys.path.insert(0,reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from jevany.data import materialize
    from jevany.model import DecisionModel,load_preprocessor
    from tools.curriculum_calls_riichi_retention_v1 import FIELD,MARKER,annotate,weighted_loss
    from tools.curriculum_riichi_class_loss import weighted_loss as basic_weighted_loss
    from tools.train_curriculum_calls_repair import weighted_loss as old_weighted_loss
    from tools.build_curriculum_calls_repair import MASS
    data_path=ROOT/'runs/curriculum-calls-repair-data-v1/data/data-ready-private.json';data=read(data_path)
    assert data['status']=='calls_repair12_source_data_prepared_not_parent_or_fit_admitted'
    files={str(path.resolve()):sha(path) for path in (Path(__file__),registered,data_path,
        ROOT/'tools/curriculum_calls_riichi_retention_v1.py',ROOT/'tests/core/test_curriculum_calls_riichi_retention_v1.py',
        ROOT/'docs/experiments/plans/curriculum-calls-riichi-retention-20261123.md')}
    for name in ('runs/curriculum-riichi-class-control-v1/launch-binding-private.json',
                 'runs/curriculum-calls-repair-fit-preflight-v1/attempt-315589/binding-private.json'):
        path=ROOT/name;record=read(path);files.update(record.get('files',record));files[str(path)]=sha(path)
    assert all(sha(path)==digest for path,digest in files.items())
    tokenizer=load_preprocessor(reg['training_config']['base_load_path'],backbone_adapter='text')
    proxy=SimpleNamespace(decision_mode='pointer',option_isolation=False)
    counts={};representatives={};max_packed={}
    for split,spec in data['files'].items():
        path=ROOT/spec['path'];assert sha(path)==spec['sha256'];files[str(path)]=spec['sha256']
        categories=Counter();packed_max=0
        for line in path.open():
            row=json.loads(line);raw_questions=copy.deepcopy(row['questions'])
            encoded=materialize(row);before=copy.deepcopy(encoded);annotate(encoded,row)
            assert row['questions']==raw_questions
            stripped=copy.deepcopy(encoded)
            for question in stripped['questions']:question.pop(FIELD);question.pop(MARKER)
            if stripped!=before:raise ValueError('Nonprivate materialized field changed')
            a=DecisionModel.encode(proxy,tokenizer,before,max_state=4096,max_branch=8192,strict=True)
            b=DecisionModel.encode(proxy,tokenizer,encoded,max_state=4096,max_branch=8192,strict=True)
            if a!=b or a['state_truncated'] or len(a['ids'])>8192:
                raise ValueError('Actual packed input changed/truncated/reordered')
            packed_max=max(packed_max,len(a['ids']))
            assert len(encoded['questions'])==1
            q=encoded['questions'][0];key=q['src'].removeprefix('curriculum2:');categories[key]+=1
            representatives.setdefault((split,key),copy.deepcopy(q))
            if sum(categories.values())%10000==0:
                print(json.dumps(dict(stage='calls_riichi_retention_input_preflight',split=split,rows=sum(categories.values()))),flush=True)
        assert dict(categories)==data['categories'][split] and sum(categories.values())==spec['records']
        counts[split]=dict(categories);max_packed[split]=packed_max
    assert sum(counts['train'].values())==48710 and sum(counts['validation'].values())==12404
    assert set(key for _,key in representatives)==set(MASS) and len(representatives)==24
    weights=data['example_weights'];comparisons=0;largest_value_error=0.;largest_gradient_error=0.
    for (split,key),question in representatives.items():
        for dtype in (torch.float64,torch.float32):
            x=torch.linspace(-.7,.8,len(question['keys']),device='cuda',dtype=dtype,requires_grad=True)
            y=x.detach().clone().requires_grad_();observed=[]
            actual=weighted_loss(x,question,'cuda',weights,observed)
            if key.startswith('basic:') and key.rsplit(':',1)[1] in ('riichi_yes','riichi_no'):
                mapped=dict(question,src='curriculum1:'+key.removeprefix('basic:'))
                expected=basic_weighted_loss(y,mapped,'cuda',{key.removeprefix('basic:'):weights[key]})
            else:expected=old_weighted_loss(y,question,'cuda',weights)
            ga=torch.autograd.grad(actual,x)[0];gb=torch.autograd.grad(expected,y)[0]
            ve=abs(float(actual.detach())-float(expected.detach()));ge=float((ga-gb).abs().max())
            largest_value_error=max(largest_value_error,ve);largest_gradient_error=max(largest_gradient_error,ge)
            tolerance=1e-12 if dtype==torch.float64 else 1e-6
            if ve>tolerance or ge>tolerance or len(observed)!=1:raise ValueError('Loss/gradient/source observation differs')
            comparisons+=1
    assert all(sha(path)==digest for path,digest in files.items())
    result=dict(status='calls_RI_retention_adapter_all_inputs_and_CUDA_loss_preflighted_not_fit_admitted',
        checked_materialized_and_actual_packed_inputs={split:sum(c.values()) for split,c in counts.items()},
        original_source_label_candidate_order_and_splits_preserved=True,max_packed=max_packed,
        source_categories=counts,actual_row_cuda_value_gradient_comparisons=comparisons,
        maximum_value_error=largest_value_error,maximum_gradient_error=largest_gradient_error,
        basic_RI_loss_matches_active_six_source_control=True,plain_and_advanced_CE_match_old_calls=True,
        deterministic_synthetic_logits_not_model_predictions=True,language_model_parameters_loaded=False,
        student_optimizer_updates=0,parent_model_admitted=False,trainer_or_fit_recipe_registered=False,
        active_control_and_prepared_calls_data_unchanged=True,new_post_reach_rows_admitted=False,
        full_four_course_goal_unachieved=True)
    for name,value in (('summary-private.json',result),('binding-private.json',dict(files=files,node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID']))):
        with (out/name).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();preflight(args.out.resolve())
