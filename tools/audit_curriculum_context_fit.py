"""Audit the full fit plus its versioned raw-public serving path."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_curriculum_fit import audit,read,rows,sha,write
from tools.curriculum_riichi_context import VERSION,FIELD,enrich_request
from tools.curriculum_input_contract import verify_checkpoint,load_context_client


def context_audit(registered,destination,binding=None):
    reg=read(registered);c=reg['curriculum'];out=ROOT/c['out'];last=reg['registered_final_step']
    if c.get('riichi_input_schema')!=VERSION:raise ValueError('Missing context schema')
    if binding:
        for p,h in read(binding).items():assert sha(p)==h,p
    sys.path.insert(0,reg['source']);sys.path.insert(0,reg['training_config']['upstream_source'])
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.policy.routing import load_model
    from jevmahjong.policy.unified import UnifiedPolicyClient
    from jevany.data import api_request
    import torch
    data=read(ROOT/c['data_root']/'data-ready.json');variant=read(data['context_variant'])
    for split,spec in variant['source_files'].items():
        original=rows(ROOT/spec['path']);enriched=rows(ROOT/data['stages']['stage1'][split]['path'])
        assert [enrich_request(r) for r in original]==enriched,'Training transform mismatch'
    contract_steps=sorted({0,last,*range(c['validation_every_steps'],last+1,c['validation_every_steps'])})
    contracts={step:verify_checkpoint(out/'training-run/checkpoints'/f'step-{step:06d}',reg,bundle) for step in contract_steps}
    stored=read(out/'input-contracts.json')
    assert stored=={str(k):v for k,v in contracts.items()}
    common_path=destination.with_name(destination.stem+'-core-private.json')
    audit(registered,common_path,binding)
    evidence=read(common_path);path=out/'training-run/checkpoints'/f'step-{last:06d}'
    client=load_context_client(UnifiedPolicyClient(load_model(path)),path,reg,bundle)
    predictions={p['id']:p for p in read(out/'training-run/training_eval'/f'step-{last:06d}'/'development/rows.json')}
    error=0.;count=0
    for row in rows(ROOT/variant['source_files']['validation']['path']):
        if row['questions']['action']['src']=='curriculum1:plain':continue
        raw=api_request(row);assert all(FIELD not in v for v in raw['questions']['action']['criteria'].values())
        actual=client.system_one(**raw)['answers']['action']['probabilities'];p=predictions[row['_meta']['id']]
        assert set(actual)==set(p['keys'])
        error=max(error,max(abs(actual[k]-v) for k,v in zip(p['keys'],p['p'])));count+=1
    assert error<5e-5
    del client;gc.collect();torch.cuda.empty_cache()
    for step in (0,146,292,438,last):
        protocol=read(out/'evaluations'/f'step-{step:06d}'/'protocol.json')
        assert protocol['input_contract']==contracts[step]
    for p,h in reg['inputs'].items():assert sha(p)==h,p
    if binding:
        for p,h in read(binding).items():assert sha(p)==h,p
    evidence.update(input_schema=VERSION,all_saved_checkpoint_contracts_verified=True,
        final_raw_public_serving_menus=count,final_raw_public_max_probability_error=error,
        final_policy_identity_sha256=contracts[last]['policy_identity_sha256'])
    write(destination,evidence);print(json.dumps(evidence),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registered',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--binding',type=Path);a=p.parse_args()
    try:context_audit(a.registered,a.out,a.binding)
    except BaseException as e:
        write(a.out.with_name(a.out.stem+'-error.json'),dict(error_type=type(e).__name__,message=str(e)));raise
