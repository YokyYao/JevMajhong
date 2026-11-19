"""Cache-reusing v8 continued fit with explicit fourth-place top-two risk preference."""
import argparse
from collections import Counter
import copy
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
    expected={'schema':'endgame-fourth-top2-risk-fit-v1','out':'runs/endgame-fourth-top2-chase-v1',
        'train_records':905,'positive_loss_records':826,'validation_records':186,
        'fourth_training_positions':149,'fourth_validation_positions':51,
        'input_version':'endgame-trailing-public-v1','training_objective':'top2_primary_v1',
        'terminal_utility':[1.1,1.,0.,-.05],'fourth_terminal_utility':[1.,1.,0.,0.],
        'fourth_escape_constraint':'report_only','fourth_native_kl_constraint':'report_only',
        'fourth_ordinary_scope':'all_confirmed_late_fourth_legal_actions',
        'win_continue_scope':'preserve_existing_public_gate','ordinary_fourth_total_mass':.20,
        'ordinary_fourth_strict_max_mass':.15,'max_per_record_loss_weight':8.,
        'parent':'released_v8_step2088','trainable_scope':'existing_head_only','trainable_parameters':524800,
        'gradient_method':'full_batch_projected_source_descent','projection_normalized_margin':.01,
        'projection_max_cycles':2048,'projection_tolerance':1e-8,'relative_head_l2_step':.005,
        'max_line_search_halvings':12,'source_ce_increase_tolerance':1e-7,'max_optimizer_steps':8,
        'validation_every_steps':1,'early_stop_patience':2,'soft_progress_min_delta':1e-6,
        'selection_min_hard_top2_gain':1e-12,'guard_protocol':'fourth_top2_risk_v1','third_native_kl_floor':.05,
        'new_reference_queries':0,'new_simulations':0}
    fixed={'replay_reference':.35,'replay_reference_leading':.15,'replay_fixed_win':0.,
        'endgame_terminal_win_third':.05,'endgame_terminal_win_fourth_chase':.15,
        'endgame_terminal_third':.05,'endgame_terminal_hold':.05}
    if (any(config.get(k)!=v for k,v in expected.items()) or config.get('fit_registered') is not fit
            or config.get('fixed_loss_source_mass')!=fixed or any(config.get(k) is not False for k in (
                'loo_stability_filter','independent_confirmation','export_registered','raw_publication'))):
        raise ValueError('Only the registered fourth-place risk-taking continued fit is enabled')


def verify(protocol):
    from jevmahjong.endgame.prepare import checksum
    for path,expected in protocol['source_sha256'].items():
        if checksum(path)!=expected:raise ValueError('Read-only source changed: '+path)


def read_caches(config,parent_id,train,validation):
    import torch
    from jevmahjong.endgame.collect import digest
    from jevmahjong.endgame.prepare import checksum
    from jevmahjong.endgame.reference import inference_request
    training_root=Path(config['training_feature_cache_run']);validation_root=Path(config['validation_feature_cache_run'])
    train_cache=torch.load(training_root/'frozen-features-private.pt',map_location='cpu',weights_only=True)
    ready=json.loads((validation_root/'ready-private.json').read_text())
    if checksum(validation_root/'validation-features-private.pt')!=ready['validation_cache_sha256']:
        raise ValueError('Validated186 feature file changed')
    val_cache=torch.load(validation_root/'validation-features-private.pt',map_location='cpu',weights_only=True)
    for cache,rows in ((train_cache,train),(val_cache,validation)):
        if cache['parent_identity']!=parent_id or set(cache['features'])!={r['_meta']['id'] for r in rows}:
            raise ValueError('Cached feature source identity or complete membership changed')
        for row in rows:
            rid=row['_meta']['id']
            if (cache['request_sha256'][rid]!=digest(inference_request(row))
                    or cache['features'][rid]['keys']!=list(row['questions']['action']['criteria'])):
                raise ValueError('Actual model input or ordered options changed')
    return train_cache,val_cache


def prepare(config):
    validate_config(config)
    from tools.train_endgame_constrained_head import load_assets,head_predictions
    from jevmahjong.endgame.prepare import read_jsonl,checksum,jsonl
    from jevmahjong.endgame.collect import digest,write
    from jevmahjong.endgame.reference import inference_request
    from jevmahjong.endgame.public_policy import action_keys
    from jevmahjong.endgame.native_terminal import is_native_terminal
    from jevmahjong.endgame.fourth_chase import fourth_chase_action_keys,top2_chase_target
    from jevmahjong.endgame.fourth_chase_training import fourth_probabilities,source_weights,chase_validation
    from tools.audit_endgame_parameter_attribution import control_comparison
    out=Path(config['out'])
    if any((out/name).exists() for name in ('prepare-protocol-private.json','ready-private.json','training-run')):
        raise ValueError('Inspect existing output rather than silently restarting')
    run,cache,source_config,old_proof,positive,validation,parent,parent_id=load_assets(config)
    audit_root=Path(config['risk_audit_run']);risk=json.loads((audit_root/'summary-private.json').read_text())
    if risk['status']!='complete_training_fourth_risk_audit' or not risk['old_targets_and_ledgers_reproduced']:
        raise ValueError('Fixed149-menu risk comparison is incomplete')
    verify(json.loads((audit_root/'protocol-private.json').read_text()))
    old_splits={'train':read_jsonl(run/'train.jsonl'),'validation':validation}
    splits=copy.deepcopy(old_splits);reference=json.loads((run/'reference-targets-private.json').read_text())
    paths=set(ROOT.glob('jevmahjong/**/*.py'))|set(ROOT.glob('desktop/jev_sidecar/**/*.py'))
    paths.update(ROOT/name for name in ('tools/train_endgame_fourth_chase.py','tools/train_endgame_constrained_head.py',
        'tools/audit_endgame_head_gradients.py','tools/audit_endgame_target_learning.py','tools/audit_endgame_parameter_attribution.py',
        'configs/training/endgame-fourth-top2-chase-v1.json','tests/core/test_endgame_fourth_chase_training.py'))
    paths.update(run/name for name in ('train.jsonl','validation.jsonl','config-private.json','reference-targets-private.json','preparation-private.json'))
    paths.update(cache/name for name in ('frozen-features-private.pt','summary-private.json','protocol-private.json'))
    validation_cache=Path(config['validation_feature_cache_run'])
    paths.update(validation_cache/name for name in ('validation-features-private.pt','ready-private.json'))
    paths.update(audit_root/name for name in ('summary-private.json','protocol-private.json'))
    changed=Counter();strict=Counter()
    for split,rows in splits.items():
        for row in rows:
            meta,q=row['_meta'],row['questions']['action'];rid=meta['id']
            active=(q['src'] in ('endgame_terminal_fourth','endgame_terminal_fourth_improve','endgame_terminal_win_fourth')
                if split=='train' else q['src']=='endgame_terminal' and meta.get('source_initial_placement')==4)
            if not active:continue
            root=Path(source_config['native_terminal_label_root']) if is_native_terminal(meta) else run
            path=root/'labels'/(rid.split(':',1)[1]+'.json');paths.add(path)
            if checksum(path)!=meta['terminal_label_provenance']['file_sha256']:raise ValueError('Raw terminal label changed')
            label=json.loads(path.read_text());keys=list(q['criteria'])
            top2,escape,utility=fourth_probabilities(label,keys,label['source']['seat'],
                meta['terminal_label_provenance']['worlds_per_action'],parent_id['identity_sha256'],split)
            if any(abs(escape[k]-meta['terminal_escape4_probabilities'][k])>1e-12
                    or abs(utility[k]-meta['expected_terminal_utilities'][k])>1e-12 for k in keys):
                raise ValueError('Source legacy utility/escape does not reproduce')
            meta['fourth_chase_top2_probabilities']=top2;meta['fourth_chase_policy_version']='endgame-fourth-chase-v1'
            if split=='train':
                prior=reference['rows'][rid]['probabilities'];target=top2_chase_target(prior,top2,escape,fourth_chase_action_keys(row))
                baseline=max(action_keys(row),key=prior.get);chosen=max(target,key=target.get)
                improves=top2[chosen]>top2[baseline]+1e-12;meta['strict_fourth_top2_improvement']=improves
                kind='win' if q['src']=='endgame_terminal_win_fourth' else 'ordinary';strict[kind]+=improves
                q['src']='endgame_terminal_win_fourth_chase' if kind=='win' else 'endgame_terminal_fourth_chase_'+('improve' if improves else 'keep')
                q.update(target=target,label=chosen);meta['fourth_chase_target_basis']='max_final_top2_then_escape_exact_ties'
            changed[split]+=1
    if dict(changed)!={'train':149,'validation':51} or strict!={'ordinary':16,'win':32}:
        raise ValueError('Registered149/51 target scope or audited16/32 strict partition changed')
    for split,rows in splits.items():
        for row,old in zip(rows,old_splits[split]):
            if row['_meta']['id']!=old['_meta']['id'] or inference_request(row)!=inference_request(old):
                raise ValueError('Model requests or split order changed')
            rid=row['_meta']['id']
            if reference['rows'][rid]['request_sha256']!=digest(inference_request(row)):
                raise ValueError('Frozen reference request differs')
            if 'fourth_chase_top2_probabilities' not in row['_meta'] and row!=old:
                raise ValueError('Nonfourth training/validation record changed')
    recipe=source_weights(splits['train'],config)
    protocol={'configuration':config,'parent_identity':parent_id,
        'source_sha256':{str(p.resolve()):checksum(p) for p in sorted(paths)}}
    write(out/'prepare-protocol-private.json',protocol)
    for split,rows in splits.items():jsonl(out/(split+'.jsonl'),rows)
    shutil.copy2(run/'reference-targets-private.json',out/'reference-targets-private.json')
    if (out/'reference-targets-private.json').read_bytes()!=(run/'reference-targets-private.json').read_bytes():
        raise ValueError('Original1091 references changed')
    new_positive=[r for r in splits['train'] if recipe['weights'][r['questions']['action']['src']]>0]
    if {r['_meta']['id'] for r in new_positive}!={r['_meta']['id'] for r in positive}:raise ValueError('Positive826 exposure changed')
    train_cache,val_cache=read_caches(config,parent_id,new_positive,splits['validation'])
    import torch
    sys.path.insert(0,source_config['upstream_source'])
    from jevany.model import PointerHead
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    meta=torch.load(parent/'head.pt',map_location='cpu',weights_only=True);dp,d=meta['head']['q.weight'].shape
    head=PointerHead(d,dp).to(device='cuda',dtype=torch.float32);head.load_state_dict(meta['head'],strict=True)
    controls={}
    for name,saved,rows in (('train826',train_cache,new_positive),('validation186',val_cache,splits['validation'])):
        features=saved['features']
        for v in features.values():v['decide']=v['decide'].to('cuda');v['options']=v['options'].to('cuda')
        actual=head_predictions(head,rows,features)
        controls[name]=control_comparison(rows,actual,saved['actual_probabilities'],1e-7)
        if not controls[name]['passed']:raise ValueError('Reused cache is not actual v8 inference')
    initial=chase_validation(splits['validation'],val_cache['actual_probabilities'],reference['rows'])
    verify(protocol)
    ready={'status':'fourth_top2_chase_ready_fit_disabled','configuration':config,'parent_identity':parent_id,
        'records':{'train':905,'positive_train':826,'validation':186},'changed_fourth_records':dict(changed),
        'strict_fourth_counts':dict(strict),'loss_recipe':recipe,'model_requests_unchanged':1091,
        'new_encoder_queries':0,'new_reference_queries':0,'new_simulations':0,'cache_controls':controls,
        'initial_validation':initial,'source_config':source_config,
        'data_sha256':{str(out/(name+'.jsonl')):checksum(out/(name+'.jsonl')) for name in ('train','validation')},
        'reference_sha256':checksum(out/'reference-targets-private.json'),'fit_registered':False}
    write(out/'ready-private.json',ready);return ready


def fit(config):
    validate_config(config,fit=True)
    from jevmahjong.endgame.collect import write
    from jevmahjong.endgame.prepare import read_jsonl,checksum
    from jevmahjong.endgame.futures import bundle_identity
    from jevmahjong.endgame.constrained_head import projected_descent,constrained_update,updated_head_metadata
    from jevmahjong.endgame.fourth_chase_training import chase_validation,chase_guards,chase_training_metrics,progress_state
    from tools.audit_endgame_head_gradients import source_gradients
    from tools.train_endgame_constrained_head import head_predictions
    out=Path(config['out']);ready=json.loads((out/'ready-private.json').read_text())
    if ready['configuration']!=dict(config,fit_registered=False) or (out/'training-run').exists():
        raise ValueError('Unverified preparation or existing fit')
    protocol=json.loads((out/'prepare-protocol-private.json').read_text());verify(protocol)
    for path,expected in ready['data_sha256'].items():
        if checksum(path)!=expected:raise ValueError('Prepared dataset changed')
    if checksum(out/'reference-targets-private.json')!=ready['reference_sha256']:raise ValueError('Reference changed')
    recipe=ready['loss_recipe'];train=[r for r in read_jsonl(out/'train.jsonl') if recipe['weights'][r['questions']['action']['src']]>0]
    validation=read_jsonl(out/'validation.jsonl');fourth=[r for r in train if 'fourth_chase_top2_probabilities' in r['_meta']]
    if len(train)!=826 or len(fourth)!=149:raise ValueError('Full826/149 source coverage changed')
    import torch
    source_config=ready['source_config'];sys.path.insert(0,source_config['upstream_source'])
    from jevany.model import PointerHead
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parent=Path(source_config['checkpoint']);meta=torch.load(parent/'head.pt',map_location='cpu',weights_only=True)
    dp,d=meta['head']['q.weight'].shape;head=PointerHead(d,dp).to(device='cuda',dtype=torch.float32).requires_grad_(True)
    head.load_state_dict(meta['head'],strict=True)
    if sum(p.numel() for p in head.parameters())!=524800:raise ValueError('Existing head architecture changed')
    train_cache,val_cache=read_caches(config,ready['parent_identity'],train,validation)
    features=dict(train_cache['features'],**val_cache['features'])
    for v in features.values():v['decide']=v['decide'].to('cuda').clone();v['options']=v['options'].to('cuda').clone()
    reference=json.loads((out/'reference-targets-private.json').read_text())['rows']
    masses={k:v for k,v in recipe['mass'].items() if v>0};root=out/'training-run';curve=[];progress=None;stop_reason='max8_updates'
    def evaluate(step,update=None,projection=None):
        nonlocal progress
        predictions=head_predictions(head,validation,features);metrics=chase_validation(validation,predictions,reference)
        guards=chase_guards(step,metrics,ready['initial_validation'],source_config,exposure_met=step>0)
        progress=progress_state(step,metrics,guards,progress,ready['initial_validation'])
        point={'step':step,'metrics':metrics,'guards':guards,'progress':progress,
            'training149':chase_training_metrics(fourth,head_predictions(head,fourth,features),reference),
            'full_positive_exposure':826 if step else 0,'training_update':update,'projection':projection}
        curve.append(point);path=root/'checkpoints'/f'step-{step:06d}'
        shutil.copytree(parent,path,ignore=shutil.ignore_patterns('trainer_state*'))
        saved=updated_head_metadata(meta,{k:v.detach().cpu().clone() for k,v in head.state_dict().items()},step)
        saved.update(training_backend='full_batch_fourth_top2_chase',policy_version='endgame-fourth-chase-v1')
        torch.save(saved,path/'head.pt')
        write(root/'predictions'/f'step-{step:06d}-private.json',predictions)
        write(root/'selection.json',{'step':progress['selected_step'],
            'selected_checkpoint':f"checkpoints/step-{progress['selected_step']:06d}",
            'metric':'fourth_final_top2_probability_with_third_leading_guards'})
        write(out/'progress-private.json',point);write(out/'curve-private.json',curve)
        print(json.dumps(point),flush=True);return point
    evaluate(0)
    for step in range(1,9):
        head.train();gradients,losses=source_gradients(head,train,features)
        try:direction,projection=projected_descent(gradients,masses)
        except ValueError as error:stop_reason='projection_stopped: '+str(error);break
        update=constrained_update(head,train,features,direction,losses)
        if not update['accepted']:stop_reason=update['reason'];break
        point=evaluate(step,update,projection)
        if progress['stop']:stop_reason='two_stale_or_unsafe_validations';break
    verify(protocol)
    if bundle_identity(parent)!=ready['parent_identity']:raise ValueError('Formal v8 files changed')
    result={'status':'fourth_top2_chase_fit_complete','last_step':curve[-1]['step'],
        'selected_step':progress['selected_step'],'eligible_for_development':progress['selected_step']>0,
        'stop_reason':stop_reason,'curve':curve,'training_encoder_queries':0,'external_networks':0,
        'fourth_escape_is_not_a_veto':True,'deployed':False,'raw_publication':False}
    write(out/'fit-result-private.json',result);write(out/'status-private.json',{k:v for k,v in result.items() if k!='curve'})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','fit'])
    parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith(('lo-','login')):
        raise RuntimeError('Fourth chase preparation and training require a scheduled GPU allocation')
    from tools.prepare_native_context_replay import allocation_metadata
    description=subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID'],'--oneliner'],text=True)
    print(json.dumps({'node':socket.gethostname(),'allocation':allocation_metadata(description,socket.gethostname())}),flush=True)
    print(json.dumps((prepare if args.command=='prepare' else fit)(json.loads(args.config.read_text())),indent=2),flush=True)


if __name__=='__main__':main()
