"""Audit every collected fresh batch against the one frozen initial teacher."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from jevany.model import PointerHead
from scripts.audit_single_decision import audit_records,require,sha256


def audit(run):
    protocol=json.loads((run/'protocol.json').read_text());config=protocol['config']
    require(hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()==protocol['sha256'],'Protocol changed')
    require(config['teacher_control_stationary_across_stages'] is True,'Teacher must stay fixed')
    for name,expected in protocol['code_sha256'].items():require(sha256(ROOT/name)==expected,f'Numerical code changed:{name}')
    reference=ROOT/config['reference']
    require(sha256(reference/'head.pt')==config['initial_reference_head_sha256']==protocol['reference_head_sha256'],
            'Pinned teacher changed')
    require(sha256(reference/'adapter_model.safetensors')==protocol['reference_adapter_sha256'],'LoRA changed')
    meta=torch.load(reference/'head.pt',map_location='cpu',weights_only=False)
    require(meta['head_residual_dim']==0 and meta['temperature']==1.,'Unexpected pointer head')
    dp,d=meta['head']['q.weight'].shape;head=PointerHead(d,dp).cpu().eval();head.load_state_dict(meta['head'])
    initial=torch.load(run/'checkpoints/update-000000/head.pt',map_location='cpu',weights_only=False)['head']
    require(all(torch.equal(v,initial[k]) for k,v in meta['head'].items()),'Initialization differs')
    summaries=[];seen=set();hashes={}
    for path in sorted(run.glob('training-games-stage-*.json')):
        stage=int(path.stem.rsplit('-',1)[1]);require(1<=stage<=config['max_stages'],'Unexpected stage')
        controls=run/f'reference-controls-stage-{stage:02d}.json';features=run/f'rollout-stage-{stage:02d}.pt'
        games=json.loads(path.read_text());fixed=json.loads(controls.read_text())
        rows=torch.load(features,map_location='cpu',weights_only=False)
        batch_config={**config,'training_seed_start':config['training_seed_start']+(stage-1)*config['training_seeds_per_stage']}
        current={g['seed'] for g in games};require(not seen&current,'Fresh batches share streams');seen|=current
        summary=audit_records(games,fixed,rows,batch_config,head);summary['stage']=stage;summaries.append(summary)
        for p in [path,controls,features]:hashes[p.name]=sha256(p)
    require(bool(summaries),'No completed data batch to audit')
    for prefix in ['development','confirmation']:
        start=config[prefix+'_seed_start'];count=config[prefix+'_seeds']
        require(not seen&set(range(start,start+count)),'Training and evaluation streams overlap')
    return {'audit_status':'passed','protocol_sha256':protocol['sha256'],'teacher_head_sha256':protocol['reference_head_sha256'],
            'initialization_tensor_identical':True,'teacher_fixed_across_stages':True,'completed_batches':len(summaries),
            'logical_proposals':sum(x['logical_proposals'] for x in summaries),
            'single_decision_states':sum(x['single_decision_states'] for x in summaries),
            'independent_training_streams':len(seen),'batches':summaries,'raw_sha256':hashes,
            'scope':'Data consistency, not a performance claim; native full matches required separately'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('Preserve previous batch audits')
    result=audit(args.run);args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
