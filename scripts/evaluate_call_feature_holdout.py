"""Freeze the expert-feature probe before opening genuinely new match validation."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from call_policy import digest, is_call_request
from head_rl import write
from scripts.probe_call_features import tensors, fit, metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',type=Path,default=Path('runs/call-feature-probe-20261110'))
    p.add_argument('--data',type=Path,default=Path('data/call-expert-expansion-20261110'))
    a=p.parse_args();out=ROOT/a.probe;data=ROOT/a.data
    target=out/'new-match-holdout-result.json'
    if target.exists():raise ValueError('New validation was already opened; do not silently refit or repeat')
    registration=json.loads((out/'holdout-evaluation-registration.json').read_text())
    selected=json.loads((out/'result.json').read_text());protocol=json.loads((out/'protocol.json').read_text())
    if digest(out/'linear-probe-private.pt')!=registration['frozen_probe_checkpoint_sha256'] or selected['selected']['l2']!=registration['frozen_selected_l2']:
        raise ValueError('Probe changed after holdout registration')
    for name,h in protocol['code_sha256'].items():
        if digest(ROOT/name)!=h:raise ValueError('Frozen probe code changed')
    source=ROOT/protocol['source_data'];old={}
    for split in ('train','validation'):
        if digest(source/(split+'.jsonl'))!=protocol['source_files'][split]:raise ValueError('Original probe data changed')
        old[split]=[json.loads(l) for l in (source/(split+'.jsonl')).read_text().splitlines()]
        old[split]=[r for r in old[split] if is_call_request(r)]
    torch.set_num_threads(8)
    cfg=protocol['config']
    null=fit(tensors(old['train']),cfg['baseline_l2'],cfg['max_lbfgs_iterations'],[0,1])
    old_null=metrics(tensors(old['validation']),old['validation'],null)
    if abs(old_null['nll']-selected['baseline']['validation']['nll'])>1e-9:
        raise ValueError('Reconstructed fixed action-prior baseline differs')
    checkpoint=torch.load(out/'linear-probe-private.pt',map_location='cpu',weights_only=True)
    torch.save({'coefficient':null,'fit_source':'original TRAIN only'},out/'null-probe-private.pt')
    provenance=json.loads((data/'provenance.json').read_text())
    if not provenance['complete'] or not provenance['chronological_dora_and_identified_passes']:
        raise ValueError('New validation preparation incomplete')
    if digest(data/'validation.jsonl')!=provenance['files']['validation']:
        raise ValueError('New validation file changed')
    old_groups={r['_meta']['group_id'] for rows in old.values() for r in rows}
    new_groups=set(provenance['matches']['validation'])
    if old_groups&new_groups:raise ValueError('New validation shares original probe matches')
    frozen={'full_probe_sha256':digest(out/'linear-probe-private.pt'),
        'null_probe_sha256':digest(out/'null-probe-private.pt'),'new_validation_sha256':digest(data/'validation.jsonl'),
        'evaluation_code_sha256':digest(Path(__file__)),'no_refit_on_new_data':True,'human_test_opened':False}
    write(out/'new-match-holdout-protocol.json',frozen)
    # First access to new per-decision data occurs only after both models are fixed.
    records=[json.loads(l) for l in (data/'validation.jsonl').read_text().splitlines()]
    if len(records)!=1024 or not all(is_call_request(r) for r in records):raise ValueError('New reaction holdout coverage differs')
    batch=tensors(records)
    result={'status':'new_match_holdout_evaluated','n':len(records),'matches':len(new_groups),
        'baseline':metrics(batch,records,null),'feature_probe':metrics(batch,records,checkpoint['coefficient']),
        'parameters_frozen_before_opening':True,'no_refit_on_new_data':True,'human_test_opened':False,
        'not_strength_evidence':True,'deployed':False}
    write(target,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
