"""Audit paired-point fitting and, once complete, native development results."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from diagonal_value import cross_validate,folded_head,ridge_fit
from head_rl import fingerprint,write
from scripts.audit_score_confirmation import compare,require
from scripts.train_diagonal_value import BASE_FILES,ORIGINAL_HEAD,load_contrasts,pointer
from scripts.prepare_preserved_incumbent import validate_frozen_runtime


def close(actual,expected,message):
    require(math.isclose(actual,expected,rel_tol=0,abs_tol=1e-8),message)


def code_unchanged(protocol):
    if 'sha256' in protocol:
        require(hashlib.sha256(json.dumps(protocol['config'],sort_keys=True).encode()).hexdigest()==protocol['sha256'],
                'Protocol config fingerprint changed')
    for name,sha in protocol['code_sha256'].items():
        require(fingerprint(ROOT/name)==sha,f'Frozen code changed:{name}')


def audit_fit(source):
    protocol=json.loads((source/'protocol.json').read_text());config=protocol['config']
    code_unchanged(protocol)
    reference=ROOT/config['reference']
    require(fingerprint(reference/'head.pt')==ORIGINAL_HEAD,'Original benchmark changed')
    metadata=torch.load(reference/'head.pt',map_location='cpu',weights_only=False)
    head=pointer(metadata,'cpu')
    x,y,seeds,states,data=load_contrasts(config,metadata,head)
    recorded=json.loads((source/'training-result.json').read_text())
    require(recorded['data']==data,'Training provenance summary changed')
    private=torch.load(source/'private-regression.pt',map_location='cpu',weights_only=False)
    require(private['seeds']==seeds and torch.equal(private['y'],y),'Saved score labels or grouping changed')
    require(torch.allclose(private['x'],x,atol=1e-4,rtol=1e-4),'Saved features differ from frozen pointer products')
    # CPU/CUDA products can differ in the last bits. Refit the exact saved design,
    # after independently checking its provenance and reconstruction above.
    cv=cross_validate(private['x'],y,seeds,config['ridge_alphas'],config['cross_validation_folds'])
    require(cv==recorded['cross_validation'],'Grouped cross-validation changed')
    coefficient=ridge_fit(private['x'],y,cv['best_alpha'])
    require(torch.equal(coefficient,private['coefficient']),'Ridge coefficients changed')
    candidates=[]
    for item in recorded['candidates']:
        checkpoint=ROOT/item['checkpoint']
        require(fingerprint(checkpoint/'head.pt')==item['head_sha256'],'Candidate head fingerprint changed')
        actual=torch.load(checkpoint/'head.pt',map_location='cpu',weights_only=False)
        validate_frozen_runtime(actual,metadata)
        require(actual['learned_degrees_of_freedom']==256,'Incorrect model capacity claim')
        expected=folded_head(head,coefficient,item['actual_strength']).state_dict()
        require(set(expected)==set(actual['head']),'Candidate head structure changed')
        for name in expected:
            require(torch.allclose(expected[name],actual['head'][name],atol=1e-7,rtol=1e-6),
                    f'Candidate does not equal declared diagonal transform:{name}')
        for name in BASE_FILES:
            require(fingerprint(reference/name)==fingerprint(checkpoint/name),'Frozen model input changed')
        require(float((coefficient*item['actual_strength']).abs().max())<=config['max_relative_q_change']+1e-12,
                'Declared query trust region exceeded')
        require(item['mean_guarded_mixture_kl']<=config['max_reference_kl'] and item['cached_greedy_flips']>0,
                'Candidate did not satisfy predeclared cache checks')
        candidates.append({'checkpoint':item['checkpoint'],'head_sha256':item['head_sha256']})
    return {'audit_status':'passed','scope':'source fingerprints, cached behavior/point labels, grouped ridge fit, native head transform',
            'source_protocol_sha256':protocol['sha256'],'data':data,'candidates':candidates,
            'maximum_cpu_cuda_feature_difference':float((private['x']-x).abs().max()),'performance_claim':False}


def audit_development(source):
    protocol=json.loads((source/'protocol.json').read_text());config=protocol['config'];code_unchanged(protocol)
    fit=json.loads((source/'training-result.json').read_text());out=source/'native-development'
    native=json.loads((out/'protocol.json').read_text());code_unchanged(native)
    require(native['config']==config and native['source_protocol_sha256']==protocol['sha256'],'Native protocol differs')
    require(native['preselected_candidates']==fit['candidates'],'Native candidate set changed after fitting')
    selection=json.loads((out/'selection.json').read_text());require(selection['status']=='complete','Incomplete native study')
    results=json.loads((out/'development-results.json').read_text())
    require(selection['results']==results and len(results)==len(fit['candidates']),'Missing predefined native variants')
    baseline=[json.loads(s) for s in (out/'baseline'/'games.jsonl').read_text().splitlines()]
    audit_config={**config,'confirmation_seed_start':config['development_seed_start'],
                  'confirmation_seeds':config['development_seeds']}
    audited=[];hashes={'baseline':fingerprint(out/'baseline'/'games.jsonl')}
    for index,(entry,item) in enumerate(zip(results,fit['candidates']),1):
        require(entry['variant']==index and entry['checkpoint']==item['checkpoint'],'Variant order or checkpoint changed')
        require(fingerprint(ROOT/item['checkpoint']/'head.pt')==item['head_sha256']==entry['head_sha256'],'Native head changed')
        require(1<=entry['native_parity_checks']<=32,'Native parity was not exercised')
        path=out/f'variant-{index:02d}'/'games.jsonl'
        candidate=[json.loads(s) for s in path.read_text().splitlines()]
        computed=compare(candidate,baseline,audit_config)
        for name,value in computed['candidate'].items():close(entry['metrics'][name],value,'Candidate summary differs from outcomes')
        for name,value in computed['reference'].items():close(selection['baseline'][name],value,'Baseline summary differs from outcomes')
        close(entry['paired_exploratory']['primary_delta'],computed['primary_delta'],'Paired gain differs')
        require(len(entry['paired_exploratory']['primary_delta_ci95'])==2,'Malformed development confidence interval')
        for a,b in zip(entry['paired_exploratory']['primary_delta_ci95'],computed['primary_delta_ci95']):
            close(a,b,'Paired confidence interval differs')
        hashes[f'variant-{index:02d}']=fingerprint(path)
        audited.append({'variant':index,'head_sha256':item['head_sha256'],**computed})
    positive=[r for r in results if r['paired_exploratory']['primary_delta']>0]
    chosen=max(positive,key=lambda r:(r['metrics']['mean_net_points'],-r['variant'])) if positive else None
    require(selection['selected']==chosen,'Selection differs from highest positive complete native result')
    require(selection['independent_improvement_confirmed'] is False,'Development was misrepresented as confirmation')
    return {'audit_status':'passed','selected':chosen,'raw_record_sha256':hashes,'variants':audited,
            'scope':'complete East native development with three rule opponents; not independent confirmation'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--development',action='store_true');parser.add_argument('--out',type=Path)
    args=parser.parse_args();result=audit_development(args.run) if args.development else audit_fit(args.run)
    if args.out:write(args.out,result)
    print(json.dumps(result,indent=2),flush=True)
