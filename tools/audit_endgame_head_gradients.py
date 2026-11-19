"""TRAIN-only frozen-feature source-gradient and actual-batch composition audit."""
import argparse
from collections import Counter,defaultdict
import copy
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def validate_config(config):
    expected={'schema':'endgame-head-gradient-diagnostic-v1','training_records':905,
        'positive_loss_records':826,'reused_encoder_features':19,'new_encoder_queries':807,
        'parameter_steps':[0,32],'actual_order_steps':32,'effective_batch':16,
        'control_max_probability_difference':1e-7,'optimizer_updates':0,'new_simulations':0}
    if any(config.get(k)!=v for k,v in expected.items()) or any(config.get(k) is not False for k in (
            'validation_data_used','candidate_training_registered','export_registered',
            'independent_confirmation','raw_publication')):
        raise ValueError('Only the registered training-only zero-update gradient audit is enabled')


def cosine(a,b):
    import torch
    a,b=a.double(),b.double();denom=torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)
    return float(torch.dot(a,b)/denom) if denom>1e-20 else None


def gradient_directions(gradients,masses,parameters):
    import torch
    if set(gradients)!=set(masses) or abs(sum(masses.values())-1.)>1e-9:
        raise ValueError('Source gradient mass does not match the complete training objective')
    total=sum(masses[k]*v.double() for k,v in gradients.items())
    if not torch.isfinite(total).all():raise ValueError('Nonfinite combined source gradient')
    norm=torch.linalg.vector_norm(total);clipped=total*min(1.,1./(float(norm)+1e-6))
    adam=clipped/(clipped.abs()+1e-8)+.001*parameters.double()
    details={k:{'gradient_norm':float(torch.linalg.vector_norm(v.double())),
        'loss_mass':masses[k],'cosine_with_full_gradient':cosine(v,total),
        'sgd_first_order_loss_derivative_per_unit_lr':-float(torch.dot(v.double(),total)),
        'adamw_first_order_loss_derivative_per_unit_lr':-float(torch.dot(v.double(),adam))}
        for k,v in gradients.items()}
    pairs={a:{b:cosine(ga,gb) for b,gb in gradients.items()} for a,ga in gradients.items()}
    return {'sources':details,'pairwise_gradient_cosines':pairs,
        'full_gradient_norm':float(norm),'first_order_only':True},total


def audit_order(records,order,seen_ids,weights,*,steps=32,batch=16):
    if sorted(order)!=list(range(len(records))) or len(set(seen_ids))!=steps*batch:
        raise ValueError('Saved actual shuffle or exposure is incomplete')
    visited=[records[i] for i in order[:steps*batch]]
    if {r['_meta']['id'] for r in visited}!=set(seen_ids):
        raise ValueError('Saved shuffle does not reproduce the actual exposure ledger')
    blocks=[]
    for start in range(0,steps,8):
        rows=visited[start*batch:(start+8)*batch];counts=Counter(r['questions']['action']['src'] for r in rows)
        mass={k:counts[k]*weights[k]/len(rows) for k in counts}
        blocks.append({'first_update':start+1,'last_update':start+8,'counts':dict(counts),
            'weighted_loss_coefficient_sum_by_source':mass,
            'note':'coefficients are exposure weights, not measured loss values'})
    return blocks,visited


def source_gradients(head,records,features):
    import torch
    parameters=tuple(head.parameters());groups=defaultdict(list)
    for row in records:groups[row['questions']['action']['src']].append(row)
    gradients={};losses={}
    for source,rows in sorted(groups.items()):
        values=[]
        for row in rows:
            rid=row['_meta']['id'];value=features[rid];keys=value['keys'];q=row['questions']['action']
            logits=head(value['decide'],value['options']).float()
            target=torch.tensor([q['target'][k] for k in keys],device=logits.device,dtype=torch.float32)
            target=target/target.sum()
            values.append(-(target*logits.log_softmax(-1)).sum())
        loss=torch.stack(values).mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite source loss')
        gradients[source]=torch.cat([g.detach().reshape(-1).double() for g in torch.autograd.grad(loss,parameters)])
        losses[source]=float(loss.detach())
    return gradients,losses


def audit(config):
    validate_config(config)
    from jevmahjong.endgame.collect import digest,write
    from jevmahjong.endgame.prepare import read_jsonl,checksum
    from jevmahjong.endgame.reference import inference_request
    from jevmahjong.endgame.futures import bundle_identity
    from desktop.jev_sidecar.unified_policy import identity
    from tools.audit_endgame_parameter_attribution import control_comparison
    from tools.probe_endgame_head_capacity import encoded_public_state_audit,validated_head_distribution
    out,run,capture,previous=map(Path,(config['out'],config['research_data_run'],
        config['capacity_capture_run'],config['baseline_prediction_run']))
    if any((out/name).exists() for name in ('protocol-private.json','summary-private.json')):
        raise ValueError('Inspect the existing audit instead of silently restarting')
    completed=json.loads((run/'postfit-summary-private.json').read_text())
    if (completed['status']!='fourteenth_fit_selection_and_training_learning_audit_complete'
            or completed['last_step']!=32 or completed['selected_step']!=0
            or not completed['saved_lora_identical_to_formal_v8']):
        raise ValueError('Expected the completed unqualified frozen-encoder fit32')
    run_config=json.loads((run/'config-private.json').read_text())
    proof=json.loads((run/'preparation-private.json').read_text());weights=proof['loss_source_weights']
    all_rows=read_jsonl(run/'train.jsonl');rows=[r for r in all_rows if weights[r['questions']['action']['src']]>0]
    if len(all_rows)!=905 or len(rows)!=826 or len({r['_meta']['id'] for r in rows})!=826:
        raise ValueError('Complete positive-loss training set differs')
    parent=Path(run_config['checkpoint']);parent_id=bundle_identity(parent);released=identity()
    if parent_id['identity_sha256'] not in (released['identity_sha256'],released['source_identity_sha256']):
        raise ValueError('Source must be the formal v8')
    import torch
    resume_protocol=json.loads((capture/'resume-protocol-private.json').read_text())
    if (resume_protocol['parent_identity']!=parent_id or resume_protocol['versions']!={
            name:importlib.metadata.version(name) for name in resume_protocol['versions']}):
        raise ValueError('Cached encoder identity or library versions changed')
    # The only changed old dependency is the offline fit implementation from
    # round14; encoding/inference dependencies must still exactly match.
    for path,expected in resume_protocol['source_sha256'].items():
        if Path(path).resolve()==ROOT/'jevmahjong/endgame/fit.py':continue
        if checksum(path)!=expected:raise ValueError('Capture dependency changed: '+path)
    saved=torch.load(capture/'frozen-features-private.pt',map_location='cpu',weights_only=True)
    reuse=saved['features'];strict={r['_meta']['id']:r for r in rows if r['questions']['action']['src']=='endgame_terminal_fourth_improve'}
    if set(reuse)!=set(strict) or len(reuse)!=19 or saved['parent_identity']!=parent_id:
        raise ValueError('Cached19 feature identities differ')
    old_reference=json.loads((Path('runs/endgame-placement-v13')/'reference-targets-private.json').read_text())
    reference=json.loads((run/'reference-targets-private.json').read_text())
    if reference!=old_reference or reference['identity_sha256']!=parent_id['identity_sha256']:
        raise ValueError('Original target/capture inputs or reference changed')
    for rid,row in strict.items():
        if reference['rows'][rid]['request_sha256']!=digest(inference_request(row)):
            raise ValueError('Reused feature request changed')
    baseline=json.loads((previous/'baseline_rich_train-predictions-private.json').read_text())
    capacity_control=json.loads((capture/'end_to_end-control-private.json').read_text())
    if (not capacity_control['passed'] or capacity_control['records']!=19
            or capacity_control['max_absolute_probability_difference']!=0.):
        raise ValueError('Original cached19 end-to-end control failed')
    upstream=Path(run_config['upstream_source']);sys.path.insert(0,str(upstream))
    paths=set(ROOT.glob('jevmahjong/**/*.py'))|set(ROOT.glob('desktop/jev_sidecar/**/*.py'))
    paths.update((upstream/'jevany').glob('**/*.py'))
    paths.update(ROOT/name for name in ('tools/audit_endgame_head_gradients.py',
        'tools/probe_endgame_head_capacity.py','tools/audit_endgame_parameter_attribution.py',
        'configs/training/endgame-head-gradient-diagnostic-v1.json'))
    paths.update(run/name for name in ('train.jsonl','config-private.json','preparation-private.json',
        'reference-targets-private.json','postfit-summary-private.json','training-exposure-private.json',
        'training-run/trainer_state.pt','training-run/checkpoints/step-000032/head.pt'))
    paths.update(capture/name for name in ('frozen-features-private.pt','resume-protocol-private.json',
        'end_to_end-control-private.json','token-audits-private.json'))
    paths.add(previous/'baseline_rich_train-predictions-private.json')
    protocol={'configuration':config,'parent_identity':parent_id,
        'source_sha256':{str(p.resolve()):checksum(p) for p in sorted(paths)},
        'record_ids':[r['_meta']['id'] for r in rows],
        'versions':{name:importlib.metadata.version(name) for name in ('torch','transformers','peft','riichienv','mahjong')},
        'optimizer_updates':0,'validation_data_used':False}
    write(out/'protocol-private.json',protocol)
    from jevmahjong.policy.routing import load_model
    from jevany.api import validate_distribution
    from jevany.model import tokenizer_of,PointerHead
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    client=load_model(parent);client.runtime.model.eval().requires_grad_(False)
    if client.runtime.inference_options.prefix_cache_size!=0:raise ValueError('Inference prefix cache must be disabled')
    original_head=client.runtime.model.head
    if original_head.residual is not None or original_head.temperature!=1.:
        raise ValueError('Expected the existing plain temperature1 pointer head')
    features=copy.deepcopy(reuse);actual={rid:baseline[rid] for rid in reuse}
    token_audits=json.loads((capture/'token-audits-private.json').read_text())
    captured=[];encodings=[];original_admit=client.runtime._admit
    def traced_admit(record):
        result=original_admit(record);enc=result[1];count=enc['seg'].count(0)
        text=tokenizer_of(client.runtime.tok).decode(enc['ids'][:count],skip_special_tokens=False)
        forbidden=('expected_terminal_utilities','terminal_escape4_probabilities','strict_original_policy_improvement')
        if enc.get('state_truncated') or any(k in record['state'] or k in text for k in forbidden):
            raise ValueError('State truncated or contains target metadata')
        check={'state_truncated':False,'target_metadata_absent':True,'total_tokens':len(enc['ids'])}
        if 'endgame_context' in record['state']:check=encoded_public_state_audit(record['state'],text,enc)
        encodings.append(check);return result
    def capture_features(module,args):
        if len(args)!=2 or args[0].ndim!=1 or args[1].ndim!=2 or torch.is_autocast_enabled('cuda'):
            raise ValueError('Unexpected real FP32 pointer-head input')
        with torch.inference_mode(False):
            captured.append({'decide':args[0].detach().to('cpu',torch.float32).clone(),
                'options':args[1].detach().to('cpu',torch.float32).clone(),
                'autocast_enabled':False,'autocast_dtype':'bfloat16'})
    client.runtime._admit=traced_admit;handle=original_head.register_forward_pre_hook(capture_features)
    queried=0
    try:
        for row in rows:
            rid=row['_meta']['id'];keys=list(row['questions']['action']['criteria'])
            if rid in features:
                if features[rid]['keys']!=keys:raise ValueError('Cached option ordering differs')
                continue
            captured.clear();encodings.clear();answer=client.system_one(**inference_request(row))
            probabilities,_=validate_distribution(answer['answers']['action']['probabilities'],keys)
            if len(captured)!=1 or len(encodings)!=1 or captured[0]['options'].shape[0]!=len(keys):
                raise ValueError('Expected one encoding and head call for each training row')
            features[rid]=dict(captured[0],keys=keys);actual[rid]=dict(zip(keys,probabilities));token_audits[rid]=encodings[0]
            queried+=1
            if queried%50==0:
                value={'status':'capturing_training_features','new_queries':queried,'new_query_budget':807,'reused':19,'optimizer_updates':0}
                write(out/'status-private.json',value);print(json.dumps(value),flush=True)
    finally:handle.remove();client.runtime._admit=original_admit
    if queried!=807 or len(features)!=826:raise ValueError('Registered capture budget changed')
    controls=[r for r in rows if r['_meta']['id'] in baseline]
    repeat=control_comparison(controls,actual,baseline,1e-7)
    write(out/'end-to-end-control-private.json',repeat)
    if repeat['records']!=55 or not repeat['passed']:raise ValueError('Formal55 baseline repeat differs')
    saved_features={'features':features,'actual_probabilities':actual,'parent_identity':parent_id,
        'request_sha256':{r['_meta']['id']:digest(inference_request(r)) for r in rows},
        'versions':protocol['versions'],'diagnostic_only':True}
    torch.save(saved_features,out/'frozen-features-private.pt');write(out/'token-audits-private.json',token_audits)
    del client,original_head,original_admit;gc.collect();torch.cuda.empty_cache()
    metadata=torch.load(parent/'head.pt',map_location='cpu',weights_only=True);dp,d=metadata['head']['q.weight'].shape
    head=PointerHead(d,dp).to(device='cuda',dtype=torch.float32).requires_grad_(True)
    head.load_state_dict(metadata['head'],strict=True)
    if sum(p.numel() for p in head.parameters())!=524800:
        raise ValueError('Only the original524800-parameter readout may be differentiated')
    for value in features.values():
        value['decide']=value['decide'].to('cuda').clone();value['options']=value['options'].to('cuda').clone()
    replay={}
    with torch.no_grad():
        for row in rows:
            rid=row['_meta']['id'];value=features[rid]
            replay[rid]=validated_head_distribution(head(value['decide'],value['options']).softmax(-1).cpu().tolist(),value['keys'],validate_distribution)
    replay_control=control_comparison(rows,replay,actual,1e-7)
    write(out/'head-replay-control-private.json',replay_control)
    if not replay_control['passed']:raise ValueError('Frozen-feature replay is not real v8 inference')
    source_counts=Counter(r['questions']['action']['src'] for r in rows)
    masses={source:count*weights[source]/905 for source,count in source_counts.items()}
    trainer_state=torch.load(run/'training-run/trainer_state.pt',map_location='cpu',weights_only=True)
    if (trainer_state['optimizer_step']!=32 or trainer_state['epoch']!=0
            or trainer_state['next_microbatch']!=512 or trainer_state['world_size']!=1):
        raise ValueError('Saved actual training order checkpoint changed')
    exposure=json.loads((run/'training-exposure-private.json').read_text())
    blocks,visited=audit_order(all_rows,trainer_state['order'].tolist(),exposure['seen_record_ids'],weights)
    audits={};private_gradients={};batch_audits=[]
    for step in (0,32):
        state=metadata['head'] if step==0 else torch.load(run/'training-run/checkpoints/step-000032/head.pt',map_location='cpu',weights_only=True)['head']
        head.load_state_dict(state,strict=True);head.train()
        before={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        gradients,losses=source_gradients(head,rows,features)
        parameters=torch.cat([p.detach().reshape(-1).double() for p in head.parameters()])
        stats,total=gradient_directions(gradients,masses,parameters);stats['mean_ce_by_source']=losses
        audits[str(step)]=stats;private_gradients[str(step)]={k:v.cpu() for k,v in gradients.items()}
        if step==0:
            strict_gradient=gradients['endgame_terminal_fourth_improve']
            for index in range(32):
                batch_rows=[r for r in visited[index*16:(index+1)*16] if weights[r['questions']['action']['src']]>0]
                gs,_=source_gradients(head,batch_rows,features)
                counts=Counter(r['questions']['action']['src'] for r in batch_rows)
                batch_gradient=sum(counts[k]*weights[k]/16*value for k,value in gs.items())
                batch_audits.append({'update':index+1,'cosine_with_full_gradient_at_v8':cosine(batch_gradient,total),
                    'strict_loss_sgd_derivative_at_v8':-float(torch.dot(strict_gradient,batch_gradient)),
                    'counts':dict(counts)})
        if any(not torch.equal(v.detach().cpu(),before[k]) for k,v in head.state_dict().items()):
            raise ValueError('Diagnostic gradient calculation changed a head parameter')
    for path,expected in protocol['source_sha256'].items():
        if checksum(path)!=expected:raise ValueError('Read-only source changed during audit')
    if bundle_identity(parent)!=parent_id:raise ValueError('Formal v8 weights changed')
    torch.save(private_gradients,out/'source-gradients-private.pt');write(out/'batch-gradients-private.json',batch_audits)
    write(out/'source-gradient-summary-private.json',audits);write(out/'actual-order-summary-private.json',blocks)
    summary={'status':'training_head_gradient_audit_complete','training_records':905,'positive_loss_records':826,
        'encoder_queries_reused':19,'new_encoder_queries':807,'head_replay_control':replay_control,
        'baseline_repeat_control':repeat,'source_counts':dict(source_counts),'source_loss_masses':masses,
        'parameter_step_audits':audits,'actual_order_blocks':blocks,
        'at_v8_batches_opposing_strict_sgd_loss_descent':sum(r['strict_loss_sgd_derivative_at_v8']>0 for r in batch_audits),
        'optimizer_updates':0,'all_source_and_formal_weights_unchanged':True,'validation_data_used':False,
        'new_simulations':0,'independent_confirmation':False,'exported_or_deployed':False,'raw_publication':False,
        'feature_cache_sha256':checksum(out/'frozen-features-private.pt')}
    write(out/'summary-private.json',summary);write(out/'status-private.json',{'status':summary['status']})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith(('lo-','login')):
        raise RuntimeError('Gradient diagnostic requires a scheduled GPU allocation')
    from tools.prepare_native_context_replay import allocation_metadata
    description=subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID'],'--oneliner'],text=True)
    print(json.dumps({'node':socket.gethostname(),'allocation':allocation_metadata(description,socket.gethostname())}),flush=True)
    print(json.dumps(audit(json.loads(args.config.read_text())),indent=2),flush=True)


if __name__=='__main__':main()
