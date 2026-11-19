"""Prepare audited frozen features and fit the original head with source descent constraints."""
import argparse
from collections import Counter
import copy
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def validate_config(config,*,fit=False):
    expected={'schema':'endgame-full-batch-constrained-existing-head-v1','train_records':905,
        'positive_loss_records':826,'validation_records':186,'reused_training_features':826,
        'new_validation_encoder_queries':186,'parent':'released_v8_step2088',
        'trainable_scope':'existing_head_only','trainable_parameters':524800,
        'input_version':'endgame-trailing-public-v1','data_targets_and_loss_masses':'unchanged_from_round14',
        'training_objective':'top2_primary_v1','terminal_utility':[1.1,1.,0.,-.05],
        'gradient_method':'full_batch_projected_source_descent','projection_normalized_margin':.01,
        'projection_max_cycles':2048,'projection_tolerance':1e-8,'relative_head_l2_step':.005,
        'max_line_search_halvings':12,'source_ce_increase_tolerance':1e-7,'max_optimizer_steps':8,
        'validation_every_steps':1,'early_stop_patience':2,'selection_min_delta':.0001,
        'guard_protocol':'unchanged_from_round14','exact_probability_control_tolerance':1e-7}
    if any(config.get(k)!=v for k,v in expected.items()) or config.get('fit_registered') is not fit or any(
            config.get(k) is not False for k in ('independent_confirmation','export_registered','raw_publication')):
        raise ValueError('Only the registered v8 full-source constrained-head protocol is enabled')


def verify_sources(protocol):
    from jevmahjong.endgame.prepare import checksum
    for path,expected in protocol['source_sha256'].items():
        if checksum(path)!=expected:raise ValueError('Registered read-only dependency changed: '+path)
    versions={name:importlib.metadata.version(name) for name in protocol['versions']}
    if versions!=protocol['versions']:raise ValueError('Feature/inference library versions changed')


def load_assets(config):
    from jevmahjong.endgame.prepare import checksum,read_jsonl
    from jevmahjong.endgame.futures import bundle_identity
    from desktop.jev_sidecar.unified_policy import identity
    run=Path(config['source_data_run']);cache=Path(config['training_feature_cache_run'])
    source_config=json.loads((run/'config-private.json').read_text())
    if (source_config['input_version']!=config['input_version'] or source_config['training_objective']!=config['training_objective']
            or source_config['terminal_utility']!=config['terminal_utility']):
        raise ValueError('Source input, objective or terminal utility changed')
    proof=json.loads((run/'preparation-private.json').read_text())
    for item in proof['files'].values():
        if checksum(item['path'])!=item['sha256']:raise ValueError('Source data changed')
    train=read_jsonl(run/'train.jsonl');validation=read_jsonl(run/'validation.jsonl')
    rows=[r for r in train if proof['loss_source_weights'][r['questions']['action']['src']]>0]
    if len(train)!=905 or len(rows)!=826 or len(validation)!=186:
        raise ValueError('Complete training and validation membership changed')
    if {r['_meta']['id'] for r in train}&{r['_meta']['id'] for r in validation}:
        raise ValueError('Training and validation IDs overlap')
    parent=Path(source_config['checkpoint']);parent_id=bundle_identity(parent);released=identity()
    if parent_id['identity_sha256'] not in (released['identity_sha256'],released['source_identity_sha256']):
        raise ValueError('Continuation must start from formal v8')
    summary=json.loads((cache/'summary-private.json').read_text())
    if (summary['status']!='training_head_gradient_audit_complete' or summary['optimizer_updates']!=0
            or summary['head_replay_control']['records']!=826 or not summary['head_replay_control']['passed']
            or summary['head_replay_control']['max_absolute_probability_difference']!=0.
            or checksum(cache/'frozen-features-private.pt')!=summary['feature_cache_sha256']):
        raise ValueError('Full826 feature provenance or exact replay check failed')
    verify_sources(json.loads((cache/'protocol-private.json').read_text()))
    return run,cache,source_config,proof,rows,validation,parent,parent_id


def head_predictions(head,records,features):
    import torch
    from jevany.api import validate_distribution
    from tools.probe_endgame_head_capacity import validated_head_distribution
    result={};head.eval()
    with torch.no_grad():
        for row in records:
            rid=row['_meta']['id'];v=features[rid]
            result[rid]=validated_head_distribution(head(v['decide'],v['options']).float().softmax(-1).cpu().tolist(),v['keys'],validate_distribution)
    return result


def prepare(config):
    validate_config(config)
    from jevmahjong.endgame.collect import digest,write
    from jevmahjong.endgame.prepare import checksum
    from jevmahjong.endgame.reference import inference_request
    from jevmahjong.endgame.constrained_head import guarded_validation
    from tools.audit_endgame_parameter_attribution import saved_predictions,control_comparison
    from tools.audit_endgame_target_learning import validation_metrics,verify_logged_metrics
    run,cache,source_config,proof,train,validation,parent,parent_id=load_assets(config)
    out=Path(config['out'])
    if any((out/name).exists() for name in ('protocol-private.json','ready-private.json','training-run')):
        raise ValueError('Inspect existing preparation or training; do not restart silently')
    import torch
    saved=torch.load(cache/'frozen-features-private.pt',map_location='cpu',weights_only=True)
    if (saved['parent_identity']!=parent_id or set(saved['features'])!={r['_meta']['id'] for r in train}
            or any(saved['request_sha256'][r['_meta']['id']]!=digest(inference_request(r)) for r in train)):
        raise ValueError('Training feature identities or actual requests changed')
    reference=json.loads((run/'reference-targets-private.json').read_text())['rows']
    baseline=saved_predictions(run,0);initial=validation_metrics(validation,baseline,reference)
    old_fit=json.loads((run/'fit-result-private.json').read_text());guard_checks=[]
    for logged in old_fit['validation']:
        current=validation_metrics(validation,saved_predictions(run,logged['step']),reference)
        verify_logged_metrics(current,logged)
        flags=guarded_validation(logged['step'],current,initial,source_config,
            exposure_met=logged['training_exposure']['minimum_exposure_met'])
        compare=('replay_guard_passed','placement_strata_guard_passed','reference_guard_passed',
            'native_reference_guard_passed','leading_native_reference_guard_passed',
            'fourth_escape_guard_passed','native_fourth_escape_guard_passed','third_pursuit_guard_passed',
            'exposure_guard_passed','win_stratum_guard_passed','all_guards_passed')
        if any(flags[k]!=logged[k] for k in compare):raise ValueError('New trainer changed a historical guard decision')
        guard_checks.append({'step':logged['step'],'all_metrics_and_guard_flags_identical':True})
    upstream=Path(source_config['upstream_source']);sys.path.insert(0,str(upstream))
    paths=set(ROOT.glob('jevmahjong/**/*.py'))|set(ROOT.glob('desktop/jev_sidecar/**/*.py'))
    paths.update((upstream/'jevany').glob('**/*.py'))
    paths.update(ROOT/name for name in ('tools/train_endgame_constrained_head.py','tools/audit_endgame_head_gradients.py',
        'tools/audit_endgame_target_learning.py','tools/audit_endgame_parameter_attribution.py',
        'tools/probe_endgame_head_capacity.py','configs/training/endgame-placement-v15.json'))
    paths.update(run/name for name in ('train.jsonl','validation.jsonl','config-private.json','reference-targets-private.json',
        'preparation-private.json','fit-result-private.json'))
    paths.update(cache/name for name in ('frozen-features-private.pt','summary-private.json','protocol-private.json'))
    paths.update(run/'training-run/training_eval'/f"step-{v['step']:06d}"/'development/rows.json' for v in old_fit['validation'])
    protocol={'configuration':config,'parent_identity':parent_id,'source_sha256':{str(p.resolve()):checksum(p) for p in sorted(paths)},
        'versions':saved['versions'],'training_ids':[r['_meta']['id'] for r in train],
        'validation_ids':[r['_meta']['id'] for r in validation],'historical_metric_and_guard_checks':guard_checks}
    write(out/'protocol-private.json',protocol)
    from jevmahjong.policy.routing import load_model
    from jevany.api import validate_distribution
    client=load_model(parent);client.runtime.model.eval().requires_grad_(False)
    if client.runtime.inference_options.prefix_cache_size!=0:raise ValueError('Inference prefix cache must be disabled')
    captured=[]
    def capture_features(module,args):
        if len(args)!=2 or args[0].ndim!=1 or args[1].ndim!=2 or torch.is_autocast_enabled('cuda'):
            raise ValueError('Unexpected actual FP32 readout')
        with torch.inference_mode(False):
            captured.append({'decide':args[0].detach().to('cpu',torch.float32).clone(),
                'options':args[1].detach().to('cpu',torch.float32).clone()})
    handle=client.runtime.model.head.register_forward_pre_hook(capture_features);features={};actual={}
    try:
        for row in validation:
            captured.clear();rid=row['_meta']['id'];keys=list(row['questions']['action']['criteria'])
            answer=client.system_one(**inference_request(row))
            values,_=validate_distribution(answer['answers']['action']['probabilities'],keys)
            if len(captured)!=1 or captured[0]['options'].shape[0]!=len(keys):
                raise ValueError('Expected exactly one head call per validation request')
            features[rid]=dict(captured[0],keys=keys);actual[rid]=dict(zip(keys,values))
    finally:handle.remove()
    repeat=control_comparison(validation,actual,baseline,1e-7)
    write(out/'validation-end-to-end-control-private.json',repeat)
    if not repeat['passed']:raise ValueError('New validation encoding differs from saved v8 baseline')
    del client;gc.collect();torch.cuda.empty_cache()
    from jevany.model import PointerHead
    meta=torch.load(parent/'head.pt',map_location='cpu',weights_only=True);dp,d=meta['head']['q.weight'].shape
    head=PointerHead(d,dp).to(device='cuda',dtype=torch.float32);head.load_state_dict(meta['head'],strict=True)
    for value in features.values():
        value['decide']=value['decide'].to('cuda');value['options']=value['options'].to('cuda')
    replay=control_comparison(validation,head_predictions(head,validation,features),actual,1e-7)
    write(out/'validation-head-replay-control-private.json',replay)
    if not replay['passed']:raise ValueError('Validation cached readout differs from actual model')
    for value in features.values():value['decide']=value['decide'].cpu();value['options']=value['options'].cpu()
    torch.save({'features':features,'actual_probabilities':actual,'parent_identity':parent_id,
        'request_sha256':{r['_meta']['id']:digest(inference_request(r)) for r in validation},
        'versions':protocol['versions']},out/'validation-features-private.pt')
    verify_sources(protocol)
    ready={'status':'constrained_head_data_ready_fit_disabled','parent_identity':parent_id,
        'train':905,'positive_train':826,'validation':186,'new_encoder_queries':186,'reused_training_features':826,
        'train_source_weights_unchanged':True,'historical_guard_checks':guard_checks,
        'initial_validation':initial,'configuration':config,'validation_cache_sha256':checksum(out/'validation-features-private.pt'),
        'optimizer_updates':0,'fit_registered':False}
    write(out/'ready-private.json',ready);return ready


def fit(config):
    validate_config(config,fit=True)
    from jevmahjong.endgame.collect import write
    from jevmahjong.endgame.prepare import checksum
    from jevmahjong.endgame.futures import bundle_identity
    from jevmahjong.endgame.constrained_head import (projected_descent,constrained_update,guarded_validation,
                                                   updated_head_metadata)
    from tools.audit_endgame_head_gradients import source_gradients
    from tools.audit_endgame_target_learning import validation_metrics,training_metrics
    run,cache,source_config,proof,train,validation,parent,parent_id=load_assets(config)
    out=Path(config['out']);ready=json.loads((out/'ready-private.json').read_text())
    registered=dict(config,fit_registered=False)
    if (ready['configuration']!=registered or ready['status']!='constrained_head_data_ready_fit_disabled'
            or checksum(out/'validation-features-private.pt')!=ready['validation_cache_sha256']
            or (out/'training-run').exists()):raise ValueError('Unverified preparation or existing training')
    protocol=json.loads((out/'protocol-private.json').read_text());verify_sources(protocol)
    import torch
    sys.path.insert(0,source_config['upstream_source'])
    from jevany.model import PointerHead
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    meta=torch.load(parent/'head.pt',map_location='cpu',weights_only=True);dp,d=meta['head']['q.weight'].shape
    head=PointerHead(d,dp).to(device='cuda',dtype=torch.float32).requires_grad_(True)
    head.load_state_dict(meta['head'],strict=True)
    if sum(p.numel() for p in head.parameters())!=524800:raise ValueError('Original readout architecture changed')
    training_saved=torch.load(cache/'frozen-features-private.pt',map_location='cpu',weights_only=True)
    validation_saved=torch.load(out/'validation-features-private.pt',map_location='cpu',weights_only=True)
    features=dict(training_saved['features'],**validation_saved['features'])
    for value in features.values():
        value['decide']=value['decide'].to('cuda').clone();value['options']=value['options'].to('cuda').clone()
    reference=json.loads((run/'reference-targets-private.json').read_text())['rows']
    counts=Counter(r['questions']['action']['src'] for r in train)
    masses={source:count*proof['loss_source_weights'][source]/905 for source,count in counts.items()}
    ordinary=[r for r in train if r['questions']['action']['src'].startswith('endgame_terminal_fourth')]
    if len(ordinary)!=55:raise ValueError('Fixed55 ordinary-fourth training diagnosis changed')
    root=out/'training-run';curve=[];selected=0;selected_utility=None;stale=0;stop_reason='max_updates'
    def save(step):
        path=root/'checkpoints'/f'step-{step:06d}'
        shutil.copytree(parent,path,ignore=shutil.ignore_patterns('trainer_state*'))
        metadata=updated_head_metadata(meta,{k:v.detach().cpu().clone() for k,v in head.state_dict().items()},step)
        torch.save(metadata,path/'head.pt')
    def evaluate(step,update=None,projection=None):
        predictions=head_predictions(head,validation,features)
        current=validation_metrics(validation,predictions,reference)
        flags=guarded_validation(step,current,ready['initial_validation'],source_config,exposure_met=step>0)
        training=training_metrics(ordinary,head_predictions(head,ordinary,features),reference)
        item={'step':step,'metrics':current,'guards':flags,'training55':training,
            'full_positive_exposure':826 if step else 0,'positive_source_counts':dict(counts),
            'training_update':update,'projection':projection}
        curve.append(item);save(step)
        write(root/'predictions'/f'step-{step:06d}-private.json',predictions)
        write(out/'progress-private.json',item);write(out/'curve-private.json',curve)
        print(json.dumps(item),flush=True);return item
    first=evaluate(0);initial_feasible=first['guards']['raw_constraints_met']
    if initial_feasible:selected_utility=first['metrics']['equally_stratified_utility']
    for step in range(1,9):
        head.train();gradients,losses=source_gradients(head,train,features)
        try:
            direction,projection=projected_descent(gradients,masses)
        except ValueError as error:
            stop_reason='projection_stopped: '+str(error);break
        update=constrained_update(head,train,features,direction,losses)
        if not update['accepted']:
            stop_reason=update['reason'];break
        current=evaluate(step,update,projection);utility=current['metrics']['equally_stratified_utility']
        improves=current['guards']['eligible_nonzero'] and (selected_utility is None or utility>selected_utility+.0001)
        if improves:selected=step;selected_utility=utility;stale=0
        else:stale+=1
        write(root/'selection.json',{'step':selected,'selected_checkpoint':f'checkpoints/step-{selected:06d}',
            'metric':'guarded_equally_stratified_terminal_utility','value':selected_utility})
        if stale>=2:stop_reason='two_evaluations_without_qualified_improvement';break
    if not (root/'selection.json').exists():
        write(root/'selection.json',{'step':0,'selected_checkpoint':'checkpoints/step-000000',
            'metric':'guarded_equally_stratified_terminal_utility','value':selected_utility})
    verify_sources(protocol)
    if bundle_identity(parent)!=parent_id:raise ValueError('Formal v8 files changed')
    result={'status':'constrained_full_batch_head_fit_complete','last_step':curve[-1]['step'],
        'selected_step':selected,'eligible_for_consumed_development':selected>0,'stop_reason':stop_reason,
        'source_and_parent_unchanged':True,'training_encoder_queries':0,'external_networks':0,
        'trainable_parameters':524800,'curve':curve,'deployed':False,'raw_publication':False}
    write(out/'fit-result-private.json',result);write(out/'status-private.json',{'status':result['status'],
        'last_step':result['last_step'],'selected_step':selected,'stop_reason':stop_reason});return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','fit'])
    parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith(('lo-','login')):
        raise RuntimeError('Constrained-head preparation/training requires a scheduled GPU allocation')
    from tools.prepare_native_context_replay import allocation_metadata
    description=subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID'],'--oneliner'],text=True)
    print(json.dumps({'node':socket.gethostname(),'allocation':allocation_metadata(description,socket.gethostname())}),flush=True)
    function=prepare if args.command=='prepare' else fit
    print(json.dumps(function(json.loads(args.config.read_text())),indent=2),flush=True)


if __name__=='__main__':main()
