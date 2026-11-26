"""Scheduled one-shot registration and actual model input-path parity check."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(config,registered,base,reference,out):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-','login'))
    from tools.register_curriculum_context import register,read,write
    register(config,base,reference)
    reg=read(registered);sys.path.insert(0,reg['source']);sys.path.insert(0,reg['training_config']['upstream_source'])
    from jevmahjong.evaluation.unified_contract import bundle,MODEL_FILES
    from jevmahjong.policy.routing import load_model
    from jevmahjong.policy.unified import UnifiedPolicyClient,serving_request
    from jevany.data import api_request
    from tools.curriculum_input_contract import seal_checkpoint,load_context_client
    from tools.curriculum_riichi_context import enrich_request,FIELD
    import torch
    torch.set_num_threads(4)
    # Copy six immutable files, never add a sidecar to an active V2 checkpoint.
    temporary=out.parent/'input-parity-checkpoint-private';temporary.mkdir(exist_ok=False)
    source=ROOT/'runs/curriculum-optimization-v2/stage1-fit/training-run/checkpoints/step-000000'
    for name in MODEL_FILES:shutil.copyfile(source/name,temporary/name)
    contract=seal_checkpoint(temporary,reg,bundle)
    raw_client=UnifiedPolicyClient(load_model(temporary));client=load_context_client(raw_client,temporary,reg,bundle)
    spec=read(ROOT/reg['curriculum']['data_root']/'data-ready.json')['stages']['stage1']['validation']
    values=[json.loads(s) for s in (ROOT/spec['path']).read_text().splitlines() if s]
    error=0.;count=0
    for row in values:
        if row['questions']['action']['src']=='curriculum1:plain':continue
        enriched=api_request(row);raw=copy.deepcopy(enriched)
        for value in raw['questions']['action']['criteria'].values():value.pop(FIELD,None)
        assert enrich_request(raw)==enriched
        assert serving_request(enriched)==serving_request(enrich_request(raw))
        assert any(FIELD in value for value in serving_request(enriched)['questions']['action']['criteria'].values())
        expected=raw_client.system_one(**enriched)['answers']['action']['probabilities']
        actual=client.system_one(**raw)['answers']['action']['probabilities']
        assert set(actual)==set(expected)
        error=max(error,max(abs(actual[k]-expected[k]) for k in actual));count+=1
    assert count==107 and error<5e-5
    write(out,dict(status='context_input_preflight_passed',node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'],
        validation_menus=count,max_probability_error=error,registered=str(registered),input_contract=contract,
        active_v2_modified=False,new_fit_started=False,diagnostic_step0_copy_not_training_initializer=True))
    print(out.read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--registered',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.config,a.registered,a.base,a.reference,a.out)
