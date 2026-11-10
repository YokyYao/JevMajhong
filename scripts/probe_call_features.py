"""CPU-only expert predictivity probe for public call-option features."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from call_policy import digest, is_call_request
from call_option_features import feature_vectors, VECTOR_NAMES
from head_rl import write


def tensors(records):
    features=[feature_vectors(r) for r in records];width=max(map(len,features))
    x=torch.zeros(len(records),width,len(VECTOR_NAMES),dtype=torch.float64)
    valid=torch.zeros(len(records),width,dtype=torch.bool)
    y=torch.empty(len(records),dtype=torch.long)
    for i,(record,values) in enumerate(zip(records,features)):
        x[i,:len(values)]=torch.tensor(values,dtype=x.dtype);valid[i,:len(values)]=True
        q=record['questions']['action'];y[i]=list(q['criteria']).index(q['label'])
    if not torch.isfinite(x).all():raise ValueError('Nonfinite public feature')
    return x,valid,y


def fit(batch,l2,iterations,columns):
    x,valid,y=batch;x=x[:,:,columns];coef=torch.zeros(x.shape[-1],dtype=x.dtype,requires_grad=True)
    optimizer=torch.optim.LBFGS([coef],lr=1.,max_iter=iterations,line_search_fn='strong_wolfe',tolerance_grad=1e-8)
    def closure():
        optimizer.zero_grad(set_to_none=True)
        logits=(x@coef).masked_fill(~valid,-torch.inf)
        loss=torch.nn.functional.cross_entropy(logits,y)+l2*coef.square().sum()/2
        if not torch.isfinite(loss):raise ValueError('Nonfinite feature probe loss')
        loss.backward();return loss
    optimizer.step(closure)
    result=torch.zeros(len(VECTOR_NAMES),dtype=x.dtype);result[columns]=coef.detach()
    return result


def metrics(batch,records,coef):
    x,valid,y=batch;logits=(x@coef).masked_fill(~valid,-torch.inf)
    p=logits.softmax(-1);chosen=p.argmax(-1);gold=Counter();pred=Counter();tp=Counter()
    for i,r in enumerate(records):
        q=r['questions']['action'];keys=list(q['criteria']);g=q['criteria'][q['label']]['type'];k=q['criteria'][keys[int(chosen[i])]]['type']
        gold[g]+=1;pred[k]+=1
        if chosen[i]==y[i]:tp[g]+=1
    return {'n':len(records),'nll':float(torch.nn.functional.cross_entropy(logits,y)),
        'exact_accuracy':float((chosen==y).double().mean()),'gold':dict(gold),'predicted':dict(pred),'exact_tp':dict(tp)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());out=ROOT/cfg['out'];out.mkdir(parents=True,exist_ok=False)
    source_cfg=json.loads((ROOT/cfg['source_config']).read_text());data=ROOT/source_cfg['data']
    provenance=json.loads((data/'provenance.json').read_text());records={}
    for split in ('train','validation'):
        if digest(data/(split+'.jsonl'))!=provenance['files'][split]:raise ValueError('Source changed')
        records[split]=[json.loads(l) for l in (data/(split+'.jsonl')).read_text().splitlines()]
        records[split]=[r for r in records[split] if r['_meta']['view']=='native' and is_call_request(r)]
    if len(records['train'])!=8192 or len(records['validation'])!=192:raise ValueError('Unexpected reaction coverage')
    groups={s:{r['_meta']['group_id'] for r in rows} for s,rows in records.items()}
    if groups['train']&groups['validation']:raise ValueError('Whole-match overlap')
    code=['scripts/probe_call_features.py','call_option_features.py','guide_call_features.py','semantic.py']
    protocol={'config':cfg,'code_sha256':{n:digest(ROOT/n) for n in code},'source_files':provenance['files'],
        'source_data':source_cfg['data'],'counts':{s:len(r) for s,r in records.items()},
        'matches':{s:len(v) for s,v in groups.items()},'columns':list(VECTOR_NAMES),
        'human_test_opened':False,'not_strength_evidence':True,'device':'cpu'}
    write(out/'protocol.json',protocol);torch.set_num_threads(8);torch.manual_seed(cfg['seed'])
    write(out/'status.json',{'status':'computing_public_features'})
    batches={s:tensors(r) for s,r in records.items()}
    null=fit(batches['train'],cfg['baseline_l2'],cfg['max_lbfgs_iterations'],[0,1])
    baseline={s:metrics(batches[s],records[s],null) for s in records}
    trials=[];models=[]
    for l2 in cfg['l2_grid']:
        coef=fit(batches['train'],l2,cfg['max_lbfgs_iterations'],list(range(len(VECTOR_NAMES))))
        item={'l2':l2,**{s:metrics(batches[s],records[s],coef) for s in records}}
        trials.append(item);models.append(coef);write(out/'trials-private.json',trials)
        print(json.dumps(item),flush=True)
    selected=min(range(len(trials)),key=lambda i:trials[i]['validation']['nll']);chosen=trials[selected]
    supported=chosen['validation']['nll']<baseline['validation']['nll']
    exact_calls=all(chosen['validation']['exact_tp'].get(k,0)>0 for k in ('chi','pon'))
    checkpoint={'version':1,'coefficient':models[selected],'columns':list(VECTOR_NAMES),'config':cfg,
        'source_protocol_sha256':digest(out/'protocol.json'),'not_deployed':True}
    torch.save(checkpoint,out/'linear-probe-private.pt')
    for n,h in protocol['code_sha256'].items():
        if digest(ROOT/n)!=h:raise ValueError('Probe source changed while running')
    result={'status':'feature_predictivity_supported' if supported else 'feature_predictivity_not_improved',
        'baseline':baseline,'selected':chosen,'trials':trials,'exact_chi_pon_both_nonzero':exact_calls,
        'checkpoint_sha256':digest(out/'linear-probe-private.pt'),'human_test_opened':False,
        'validation_is_development':True,'strength_improvement_proven':False,'deployed':False}
    write(out/'result.json',result);write(out/'status.json',{'status':result['status'],'selected_l2':chosen['l2'],
        'validation_nll':chosen['validation']['nll'],'baseline_validation_nll':baseline['validation']['nll'],
        'exact_chi_pon_both_nonzero':exact_calls,'strength_improvement_proven':False})
    print(json.dumps({k:v for k,v in result.items() if k not in ('trials','baseline')}),flush=True)


if __name__=='__main__':main()
