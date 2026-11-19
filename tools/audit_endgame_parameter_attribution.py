"""Read-only encoder/head attribution on an already-consumed validation set."""
import argparse
from collections import defaultdict
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def saved_predictions(run,step):
    path=run/'training-run/training_eval'/f'step-{step:06d}'/'development/rows.json'
    rows=json.loads(path.read_text())
    values={r['id']:dict(zip(r['keys'],r['p'])) for r in rows}
    if len(values)!=len(rows):raise ValueError('Repeated saved validation prediction')
    return values


def summarize(records,predictions,reference):
    import math
    from jevmahjong.endgame.fit import target_ce,original_replay_ce,reference_kl_group
    from jevmahjong.endgame.public_policy import action_keys
    from jevmahjong.endgame.reference import restrict_distribution
    ce=defaultdict(list);kl=defaultdict(list);utility=defaultdict(list)
    fourth=[];third=[];holding_changes=0
    for record in records:
        meta,q=record['_meta'],record['questions']['action'];rid=meta['id']
        p=predictions[rid];eligible=action_keys(record)
        if set(p)!=set(q['criteria']):raise ValueError('Attribution prediction menu differs')
        chosen=max(eligible,key=p.get);prior=reference[rid]['probabilities']
        baseline=max(eligible,key=prior.get)
        ce[q['src']].append(target_ce(q,p))
        if q['src'].startswith('replay_'):
            current=restrict_distribution(p,eligible)
            kl[reference_kl_group(q['src'])].append(sum(v*math.log(max(v,1e-30)/max(current[k],1e-30))
                                                       for k,v in prior.items() if v>0))
        if q['src']=='endgame_terminal':
            values=meta['expected_terminal_utilities'];utility[meta['bucket']].append(values[chosen])
            if meta['source_initial_placement']==4:
                fourth.append(meta['terminal_escape4_probabilities'][chosen])
            if meta['source_initial_placement']==3:third.append(values[chosen])
            if meta['bucket']=='hold':holding_changes+=chosen!=baseline
    if (set(utility)!={'win','third','fourth','hold'} or len(fourth)!=43 or len(third)!=23
            or {k:len(v) for k,v in kl.items()}!={'original':64,'trailing':16,'leading':16}):
        raise ValueError('Attribution metric coverage changed')
    means={k:sum(v)/len(v) for k,v in utility.items()}
    return {'original_expert_ce':original_replay_ce(ce),'reference_kl':{k:sum(v)/len(v) for k,v in kl.items()},
        'terminal_utility_by_stratum':means,'equally_stratified_terminal_utility':sum(means.values())/4,
        'fourth_escape_positions':43,'fourth_escape':sum(fourth)/43,
        'third_pursuit_positions':23,'third_pursuit_utility':sum(third)/23,
        'holding_positions':len(utility['hold']),'holding_changes_from_original_policy':holding_changes,
        'prediction_records':len(predictions)}


def control_comparison(records,actual,expected,tolerance):
    from jevmahjong.endgame.public_policy import action_keys
    maximum=0.;flips=0
    for record in records:
        rid=record['_meta']['id'];keys=list(record['questions']['action']['criteria'])
        if set(actual[rid])!=set(keys) or set(expected[rid])!=set(keys):
            raise ValueError('Control action menu changed')
        maximum=max(maximum,max(abs(actual[rid][k]-expected[rid][k]) for k in keys))
        eligible=action_keys(record)
        flips+=max(eligible,key=actual[rid].get)!=max(eligible,key=expected[rid].get)
    result={'records':len(records),'max_absolute_probability_difference':maximum,'greedy_flips':flips,
            'passed':maximum<=tolerance and flips==0}
    return result


def audit(config):
    import torch
    from peft import get_peft_model_state_dict
    from jevmahjong.endgame.collect import digest,write
    from jevmahjong.endgame.prepare import read_jsonl,checksum
    from jevmahjong.endgame.futures import bundle_identity
    from jevmahjong.endgame.reference import inference_request
    from desktop.jev_sidecar.unified_policy import identity
    if (config.get('schema')!='endgame-head-encoder-attribution-v1'
            or config.get('fit_registered') is not False or config.get('export_registered') is not False
            or config.get('candidate_step')!=72 or config.get('validation_records')!=186
            or config.get('control_count')!=8
            or config.get('control_max_absolute_probability_difference')!=1e-4
            or config.get('input_version')!='endgame-trailing-public-v1'):
        raise ValueError('Unregistered parameter-attribution protocol')
    out,run=Path(config['out']),Path(config['candidate_run'])
    if (out/'summary-private.json').exists():raise ValueError('Inspect completed attribution instead of restarting')
    if (out/'protocol-private.json').exists():raise ValueError('Inspect prior attribution failure before resubmission')
    run_config=json.loads((run/'config-private.json').read_text())
    if run_config['input_version']!=config['input_version']:raise ValueError('Candidate input format changed')
    baseline=Path(config['baseline_checkpoint']);candidate=run/'training-run/checkpoints/step-000072'
    baseline_id,candidate_id=bundle_identity(baseline),bundle_identity(candidate)
    release=identity()
    if (baseline_id['identity_sha256'] not in (release['source_identity_sha256'],release['identity_sha256'])
            or baseline.resolve()!=Path(run_config['checkpoint']).resolve()):
        raise ValueError('Baseline is not the exact released v8')
    fit=json.loads((run/'fit-result-private.json').read_text())
    if not fit['actual_updates_verified'] or fit['selection']['step']!=0 or fit['validation'][-1]['step']!=72:
        raise ValueError('Expected the inspected unqualified tenth72 candidate')
    records=read_jsonl(run/'validation.jsonl')
    a,d=saved_predictions(run,0),saved_predictions(run,72)
    if len(records)!=186 or {r['_meta']['id'] for r in records}!=set(a) or set(a)!=set(d):
        raise ValueError('Fixed186 validation identities changed')
    reference=json.loads((run/'reference-targets-private.json').read_text())
    if reference['identity_sha256']!=baseline_id['identity_sha256']:
        raise ValueError('Frozen reference differs from original baseline')
    cached_metrics={'A':summarize(records,a,reference['rows']),'D':summarize(records,d,reference['rows'])}
    for name,logged in (('A',fit['validation'][0]),('D',fit['validation'][-1])):
        current=cached_metrics[name]
        comparisons=[(current['original_expert_ce'],logged['replay_target_ce']),
            (current['fourth_escape'],logged['fourth_escape_validation']['candidate']),
            (current['third_pursuit_utility'],logged['third_pursuit_validation']['candidate'])]
        comparisons.extend((current['terminal_utility_by_stratum'][k],v) for k,v in logged['utility_by_stratum'].items())
        comparisons.extend((current['reference_kl'][group],logged[field]) for group,field in (
            ('original','replay_reference_kl'),('trailing','native_reference_kl'),('leading','leading_native_reference_kl')))
        if any(abs(x-y)>1e-12 for x,y in comparisons):
            raise ValueError('Cached arm metrics do not reproduce the original training-validation log: '+name)
    controls=sorted(records,key=lambda r:digest([config['control_hash_prefix'],r['_meta']['id']]))[:8]
    upstream=Path(run_config['upstream_source']);sys.path.insert(0,str(upstream))
    paths=sorted(set(ROOT.glob('jevmahjong/**/*.py'))|set(ROOT.glob('desktop/jev_sidecar/**/*.py'))
        |set((upstream/'jevany').glob('**/*.py'))|{Path(__file__).resolve()})
    protocol={'schema':config['schema'],'configuration':config,'baseline_identity':baseline_id,
        'candidate_identity':candidate_id,'validation_sha256':checksum(run/'validation.jsonl'),
        'reference_sha256':checksum(run/'reference-targets-private.json'),
        'saved_prediction_hashes':{str(run/'training-run/training_eval'/f'step-{step:06d}'/'development/rows.json'):
            checksum(run/'training-run/training_eval'/f'step-{step:06d}'/'development/rows.json') for step in (0,72)},
        'control_record_ids':[r['_meta']['id'] for r in controls],
        'code_sha256':{str(p.resolve()):checksum(p) for p in paths},
        'versions':{name:importlib.metadata.version(name) for name in ('torch','transformers','peft','riichienv','mahjong')},
        'optimizer_updates':0,'new_simulations':0,'independent_confirmation':False,'deployed':False}
    write(out/'protocol-private.json',protocol)
    heads={name:torch.load(path/'head.pt',map_location='cpu',weights_only=True) for name,path in
           (('original',baseline),('trained',candidate))}
    for key in ('base','base_revision','lora','head_dim','head_residual_dim','head_type','decision_mode','branch_mode','backbone_adapter'):
        if heads['original'].get(key)!=heads['trained'].get(key):raise ValueError('Head architecture differs: '+key)
    from jevmahjong.policy.routing import load_model
    from jevany.api import validate_distribution
    def load(path):
        client=load_model(path);client.runtime.model.eval().requires_grad_(False)
        if client.runtime.inference_options.prefix_cache_size!=0:raise ValueError('Head attribution requires disabled inference cache')
        return client
    def query(client,subset,arm):
        predictions={}
        for record in subset:
            rid=record['_meta']['id'];keys=list(record['questions']['action']['criteria'])
            result=client.system_one(**inference_request(record))
            values,_=validate_distribution(result['answers']['action']['probabilities'],keys)
            predictions[rid]=dict(zip(keys,values))
            if len(predictions)%25==0 or len(predictions)==len(subset):
                write(out/'status-private.json',{'status':'attribution_inference','arm':arm,
                    'records_complete':len(predictions),'records_total':len(subset),'optimizer_updates':0})
        return predictions
    def swap_head(client,name):
        model=client.runtime.model
        before={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model.lm,save_embedding_layers=False).items()}
        model.head.load_state_dict(heads[name]['head'],strict=True)
        for key,value in model.head.state_dict().items():
            if not torch.equal(value.detach().cpu(),heads[name]['head'][key].to(dtype=value.dtype)):
                raise ValueError('Loaded readout tensor differs from source')
        after=get_peft_model_state_dict(model.lm,save_embedding_layers=False)
        if set(before)!=set(after) or any(not torch.equal(v.detach().cpu(),before[k]) for k,v in after.items()):
            raise ValueError('Readout swap changed encoder LoRA')
    client=load(baseline)
    control_a=control_comparison(controls,query(client,controls,'A_control'),a,1e-4)
    write(out/'controls-private.json',{'A':control_a})
    if not control_a['passed']:raise ValueError('Original API/control predictions differ from saved0; attribution stopped')
    del client;gc.collect();torch.cuda.empty_cache()
    client=load(candidate)
    control_d=control_comparison(controls,query(client,controls,'D_control'),d,1e-4)
    write(out/'controls-private.json',{'A':control_a,'D':control_d})
    if not control_d['passed']:raise ValueError('Trained API/control predictions differ from saved72; attribution stopped')
    swap_head(client,'original');c=query(client,records,'C')
    write(out/'C-predictions-private.json',c)
    swap_head(client,'trained')
    del client;gc.collect();torch.cuda.empty_cache()
    client=load(baseline);swap_head(client,'trained');b=query(client,records,'B')
    write(out/'B-predictions-private.json',b);swap_head(client,'original')
    del client;gc.collect();torch.cuda.empty_cache()
    for path,expected in protocol['code_sha256'].items():
        if checksum(path)!=expected:raise ValueError('Source changed during attribution: '+path)
    if (bundle_identity(baseline)!=baseline_id or bundle_identity(candidate)!=candidate_id
            or checksum(run/'validation.jsonl')!=protocol['validation_sha256']
            or checksum(run/'reference-targets-private.json')!=protocol['reference_sha256']):
        raise ValueError('Read-only source identity changed during attribution')
    for path,expected in protocol['saved_prediction_hashes'].items():
        if checksum(path)!=expected:raise ValueError('Cached control predictions changed')
    summary={'status':'encoder_readout_attribution_complete','controls':{'A':control_a,'D':control_d},
        'arms':{name:summarize(records,values,reference['rows']) for name,values in (('A',a),('B',b),('C',c),('D',d))},
        'new_model_queries':388,'optimizer_updates':0,'new_simulations':0,'validation_already_consumed':True,
        'independent_strength_evidence':False,'exported_or_deployed':False}
    write(out/'summary-private.json',summary);write(out/'status-private.json',{'status':summary['status']})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith(('lo-','login')):
        raise RuntimeError('Attribution must run inside a scheduled GPU job, never login')
    from tools.prepare_native_context_replay import allocation_metadata
    description=subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID'],'--oneliner'],text=True)
    allocation_metadata(description,socket.gethostname())
    print(json.dumps(audit(json.loads(args.config.read_text())),indent=2),flush=True)


if __name__=='__main__':main()
