"""Diagnose the stage gate on practical v8, not an unseen-strength evaluation."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]


def audit(registered,out):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-','login'))
    read=lambda p:json.loads(Path(p).read_text())
    reg=read(registered);c=reg['curriculum'];sys.path.insert(0,reg['source'])
    sys.path.insert(0,reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(1)
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.policy.routing import load_model
    from jevmahjong.policy.unified import UnifiedPolicyClient
    from jevany.data import api_request
    assert package_digest(Path(reg['source'])/'jevmahjong')==reg['source_sha256']
    for path,h in reg['inputs'].items():
        with Path(path).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h,path
    checkpoint=ROOT/c['reference_checkpoint'];identity=bundle(checkpoint)
    assert identity['identity_sha256']==c['reference_identity']
    spec=read(reg['training_config']['data_receipt'])['files']['validation']
    values=[json.loads(s) for s in (ROOT/spec['path']).read_text().splitlines() if s]
    values=[r for r in values if r['questions']['action']['src']!='curriculum1:plain']
    assert len(values)==107
    client=UnifiedPolicyClient(load_model(checkpoint));counts=Counter();predictions=[]
    for row in values:
        q=row['questions']['action'];truth=q['src']=='curriculum1:riichi_yes'
        probabilities=client.system_one(**api_request(row))['answers']['action']['probabilities']
        assert set(probabilities)==set(q['criteria'])
        chosen=max(probabilities,key=probabilities.__getitem__);actual=q['criteria'][chosen]['type']=='reach'
        counts['tp' if truth and actual else 'fn' if truth else 'fp' if actual else 'tn']+=1
        predictions.append(dict(id=row['_meta']['id'],group=row['_meta']['group_id'],probabilities=probabilities))
    accept=counts['tp']/(counts['tp']+counts['fn']);decline=counts['tn']/(counts['tn']+counts['fp'])
    assert bundle(checkpoint)==identity
    result=dict(status='reference_riichi_gate_diagnostic_complete',node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'],
        windows=len(values),groups=len({r['_meta']['group_id'] for r in values}),counts=dict(counts),
        riichi_acceptance=accept,riichi_correct_decline=decline,both_recalls_ge60=accept>=.6 and decline>=.6,
        checkpoint_sha256=identity['identity_sha256'],reference_public_input_unchanged=True,
        flat_pointer_before_native_discard_guard=True,new_games=0,optimizer_updates=0,
        current_fit_or_gate_changed=False,
        note='Same consumed107-window development diagnostic as the curriculum gate. V8 may have trained on some groups before the new curriculum reservation, so this is not held-out generalization, an optimal-action teacher certificate, or independent game strength. It helps assess whether this imitation gate is aligned with the practical comparison model; complete-game evidence and frozen final candidates remain separate.')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.with_name(out.stem+'-predictions-private.json').open('x') as f:json.dump(predictions,f)
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registered',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();audit(a.registered,a.out)
