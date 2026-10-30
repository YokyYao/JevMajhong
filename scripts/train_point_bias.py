"""Fit and audit paired net-point values through contextual option embeddings."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from diagonal_value import cross_validate,ridge_fit
from point_bias import folded_bias,value_features
from head_rl import fingerprint,write
from scripts.audit_score_confirmation import require
from scripts.train_diagonal_value import BASE_FILES,ORIGINAL_HEAD,load_contrasts,pointer,policy_change
from scripts.prepare_preserved_incumbent import validate_frozen_runtime
from scripts.audit_diagonal_value import audit_development


def load_bias_data(config,metadata,head):
    _,y,seeds,states,data=load_contrasts(config,metadata,head)
    features=[];observed=[];device=next(head.parameters()).device
    with torch.no_grad():
        for source in data['sources']:
            path=ROOT/source['source']/'rollout-stage-01.pt'
            require(fingerprint(path)==source['rollout_sha256'],'Validated cache changed')
            rows=torch.load(path,map_location='cpu',weights_only=False)
            for row in rows:
                a,t=row['eligible'][row['action']],row['eligible'][row['teacher_action']]
                if a==t:continue
                values=value_features(head,row['decide'].to(device),row['options'].to(device))
                features.append((values[a]-values[t]).double().cpu());observed.append(row['episode_key'][0])
            require(fingerprint(path)==source['rollout_sha256'],'Cache changed while reading')
    require(observed==seeds and len(features)==len(y),'Contextual value rows differ from verified paired targets')
    return torch.stack(features),y,seeds,states,data


def frozen(protocol):
    require(hashlib.sha256(json.dumps(protocol['config'],sort_keys=True).encode()).hexdigest()==protocol['sha256'],
            'Frozen configuration changed')
    for name,sha in protocol['code_sha256'].items():require(fingerprint(ROOT/name)==sha,f'Frozen code changed:{name}')


def fit(config_path):
    config=json.loads(config_path.read_text());out=ROOT/config['out'];original=ROOT/config['reference']
    require(not out.exists(),'Preserve existing complete or partial value projection')
    prior=ROOT/config['previous_run'];prior_protocol=json.loads((prior/'protocol.json').read_text())
    checked=json.loads((prior/'fit-audit.json').read_text())
    require(checked['audit_status']=='passed' and checked['source_protocol_sha256']==prior_protocol['sha256'],
            'Previous fit must be audited before changing its representation')
    frozen(prior_protocol)
    comparison=json.loads((prior/'native-development/preserved-anchor/comparison-result.json').read_text())
    require(comparison['main_selection_changed'] is False,'Missing preserved-policy diagnostic')
    comparison_protocol=json.loads((prior/'native-development/preserved-anchor/protocol.json').read_text())
    require(comparison_protocol['source_protocol_sha256']==prior_protocol['sha256'],'Wrong preserved development source')
    for name,sha in comparison_protocol['code_sha256'].items():require(fingerprint(ROOT/name)==sha,'Preserved diagnostic code changed')
    require(fingerprint(original/'head.pt')==ORIGINAL_HEAD,'Keep the original SFT2000 prior')
    proposed=[set(range(config[p+'_seed_start'],config[p+'_seed_start']+config[p+'_seeds']))
              for p in ['development','confirmation']]
    for path in (ROOT/'runs').glob('*/protocol.json'):
        old=json.loads(path.read_text()).get('config',{})
        for start,length in [('training_seed_start','training_seeds_per_stage'),('development_seed_start','development_seeds'),
                             ('confirmation_seed_start','confirmation_seeds')]:
            if start in old and length in old:
                count=old.get('max_stages',1) if start=='training_seed_start' else 1
                previous=set(range(old[start],old[start]+old[length]*count))
                require(not any(walls&previous for walls in proposed),f'New study wall range already declared:{path}')
    out.mkdir(parents=True)
    code=['point_bias.py','diagonal_value.py','scripts/train_point_bias.py','scripts/train_diagonal_value.py',
          'head_rl.py','arena.py','semantic.py','seed_control.py','scripts/prepare_preserved_incumbent.py',
          'scripts/train_head_rl.py','scripts/confirm_head_rl.py','scripts/evaluate_diagonal_value.py',
          'scripts/audit_diagonal_value.py','scripts/audit_score_confirmation.py',
          'vendor/JevAny/jevany/model.py','vendor/JevAny/jevany/checkpoint.py']
    protocol={'config':config,'sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
              'code_tree_sha':subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip(),
              'code_sha256':{name:fingerprint(ROOT/name) for name in code},'reference_head_sha256':ORIGINAL_HEAD,
              'reference_adapter_sha256':fingerprint(original/'adapter_model.safetensors'),
              'feature_map':'scaled contextual key embedding; query-independent paired value difference',
              'trainable_degrees_of_freedom':256,'parameter_export':'additive query bias only'}
    write(out/'protocol.json',protocol);write(out/'status.json',{'status':'extracting_contextual_value_features'})
    metadata=torch.load(original/'head.pt',map_location='cpu',weights_only=False);head=pointer(metadata,'cuda')
    x,y,seeds,states,data=load_bias_data(config,metadata,head)
    write(out/'data-audit.json',{'audit_status':'passed',**data})
    cv=cross_validate(x,y,seeds,config['ridge_alphas'],config['cross_validation_folds']);write(out/'cross-validation.json',cv)
    if not cv['improves_over_zero']:
        result={'status':'stop_no_predictive_signal','candidates':[],'data':data,'cross_validation':cv}
        write(out/'training-result.json',result);write(out/'status.json',{'status':result['status']});return result
    coefficient=ridge_fit(x,y,cv['best_alpha'])
    torch.save({'coefficient':coefficient,'x':x,'y':y,'seeds':seeds},out/'private-regression.pt')
    candidates=[];seen=set()
    for index,requested in enumerate(config['calibration_strengths'],1):
        strength=requested
        for _ in range(16):
            candidate=folded_bias(head,coefficient,strength);change=policy_change(candidate,states)
            if change['mean_guarded_mixture_kl']<=config['max_reference_kl']:break
            strength*=.5
        else:continue
        digest=hashlib.sha256(candidate.q.bias.detach().cpu().numpy().tobytes()).hexdigest()
        if digest in seen or not change['cached_greedy_flips']:continue
        seen.add(digest);checkpoint=out/'checkpoints'/f'calibration-{index:02d}';checkpoint.mkdir(parents=True)
        for name in BASE_FILES:shutil.copy2(original/name,checkpoint/name)
        payload=dict(metadata);payload['head']={k:v.detach().cpu().clone() for k,v in candidate.state_dict().items()}
        payload.update(training_method='ridge paired-point contextual-option query-bias projection',parent=str(original),
                       learned_degrees_of_freedom=len(coefficient),regression_alpha=cv['best_alpha'],
                       calibration_strength=strength,protocol_sha256=protocol['sha256'])
        torch.save(payload,checkpoint/'head.pt')
        complete={'head_sha256':fingerprint(checkpoint/'head.pt'),'protocol_sha256':protocol['sha256'],
                  'requested_strength':requested,'actual_strength':strength,'learned_degrees_of_freedom':len(coefficient),**change}
        write(checkpoint/'complete.json',complete);candidates.append({'checkpoint':str(checkpoint.relative_to(ROOT)),**complete})
    for index,source in enumerate(data['sources'],1):
        shutil.copy2(ROOT/source['source']/'training-games-stage-01.json',out/f'training-games-stage-cache-{index:02d}.json')
    result={'status':'fit_complete' if candidates else 'stop_no_observed_policy_change','candidates':candidates,
            'data':data,'cross_validation':cv,'original_head_unchanged':fingerprint(original/'head.pt')==ORIGINAL_HEAD}
    write(out/'training-result.json',result);write(out/'status.json',{'status':result['status']});return result


def audit_fit(source):
    protocol=json.loads((source/'protocol.json').read_text());config=protocol['config'];frozen(protocol)
    original=ROOT/config['reference'];require(fingerprint(original/'head.pt')==ORIGINAL_HEAD,'Original head changed')
    metadata=torch.load(original/'head.pt',map_location='cpu',weights_only=False);head=pointer(metadata,'cpu')
    x,y,seeds,states,data=load_bias_data(config,metadata,head)
    recorded=json.loads((source/'training-result.json').read_text());require(data==recorded['data'],'Training provenance changed')
    private=torch.load(source/'private-regression.pt',map_location='cpu',weights_only=False)
    require(torch.equal(private['y'],y) and private['seeds']==seeds,'Point targets or grouping changed')
    require(torch.allclose(private['x'],x,atol=1e-4,rtol=1e-4),'Option-value features do not reproduce')
    cv=cross_validate(private['x'],y,seeds,config['ridge_alphas'],config['cross_validation_folds'])
    require(cv==recorded['cross_validation'],'Grouped value regression changed')
    coefficient=ridge_fit(private['x'],y,cv['best_alpha']);require(torch.equal(coefficient,private['coefficient']),'Value coefficients changed')
    for item in recorded['candidates']:
        checkpoint=ROOT/item['checkpoint'];require(fingerprint(checkpoint/'head.pt')==item['head_sha256'],'Projected head changed')
        actual=torch.load(checkpoint/'head.pt',map_location='cpu',weights_only=False);validate_frozen_runtime(actual,metadata)
        expected=folded_bias(head,coefficient,item['actual_strength']).state_dict()
        for name,tensor in expected.items():
            require(torch.allclose(tensor,actual['head'][name],atol=1e-7,rtol=1e-6),f'Incorrect bias-only native export:{name}')
        for name in BASE_FILES:require(fingerprint(original/name)==fingerprint(checkpoint/name),'Frozen input changed')
        require(item['mean_guarded_mixture_kl']<=config['max_reference_kl'] and item['cached_greedy_flips']>0,'Trust or observed-choice check failed')
    return {'audit_status':'passed','source_protocol_sha256':protocol['sha256'],'data':data,
            'candidate_heads':{item['checkpoint']:item['head_sha256'] for item in recorded['candidates']},
            'private_regression_sha256':fingerprint(source/'private-regression.pt'),
            'maximum_cpu_cuda_feature_difference':float((private['x']-x).abs().max()),
            'scope':'verified paired targets, contextual-key value fit and exact native query-bias export; no performance claim'}


def prepare(source,out):
    require(not out.exists(),'Preserve existing complete or partial point-value confirmation')
    protocol=json.loads((source/'protocol.json').read_text());frozen(protocol)
    evidence=json.loads((source/'fit-audit.json').read_text())
    require(evidence['audit_status']=='passed' and evidence['source_protocol_sha256']==protocol['sha256'],
            'Missing matching completed value-fit audit')
    require(fingerprint(source/'private-regression.pt')==evidence['private_regression_sha256'],'Audited fit changed')
    for record in evidence['data']['sources']:
        for filename,key in [('rollout-stage-01.pt','rollout_sha256'),('training-games-stage-01.json','games_sha256'),
                             ('reference-controls-stage-01.json','controls_sha256')]:
            require(fingerprint(ROOT/record['source']/filename)==record[key],'Audited source data changed')
    for name,sha in evidence['candidate_heads'].items():require(fingerprint(ROOT/name/'head.pt')==sha,'Audited head changed')
    development=audit_development(source);chosen=development['selected'];require(chosen is not None,'No positive native value candidate')
    config={**protocol['config'],'out':str(out)}
    reference=ROOT/config['reference'];checkpoint=ROOT/chosen['checkpoint']
    require(fingerprint(reference/'head.pt')==protocol['reference_head_sha256']==ORIGINAL_HEAD,'Original benchmark changed')
    require(fingerprint(reference/'adapter_model.safetensors')==protocol['reference_adapter_sha256'],'Original adapter changed')
    validate_frozen_runtime(torch.load(checkpoint/'head.pt',map_location='cpu',weights_only=False),
                            torch.load(reference/'head.pt',map_location='cpu',weights_only=False))
    for name in BASE_FILES:require(fingerprint(reference/name)==fingerprint(checkpoint/name),'Candidate frozen input changed')
    require(config['target_net_points_gain']==0 and config['target_mean_net_points']==0
            and config['confirmation_min_ci95_lower']==0 and config['confirmation_seeds']>=128,
            'Keep the declared positive-evidence criterion')
    files=sorted(source.glob('training-games-stage-cache-*.json'))
    training={r['seed'] for path in files for r in json.loads(path.read_text())}
    dev=set(range(config['development_seed_start'],config['development_seed_start']+config['development_seeds']))
    final=set(range(config['confirmation_seed_start'],config['confirmation_seed_start']+config['confirmation_seeds']))
    require(training and not training&dev and not training&final and not dev&final,'Missing or overlapping wall provenance')
    for path in (ROOT/'runs').glob('*/protocol.json'):
        if path.resolve()==(source/'protocol.json').resolve():continue
        old=json.loads(path.read_text()).get('config',{})
        for start,length in [('training_seed_start','training_seeds_per_stage'),('development_seed_start','development_seeds'),
                             ('confirmation_seed_start','confirmation_seeds')]:
            if start in old and length in old:
                count=old.get('max_stages',1) if start=='training_seed_start' else 1
                require(not final&set(range(old[start],old[start]+old[length]*count)),f'Final walls already declared:{path}')
    native=json.loads((source/'native-development/protocol.json').read_text())
    frozen_final={**protocol,'config':config,'sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
                  'code_sha256':native['code_sha256'],'source_run':str(source),'source_protocol_sha256':protocol['sha256'],
                  'study_type':'paired_point_query_bias_native_confirmation'}
    selection={'checkpoint':chosen['checkpoint'],'candidate_head_sha256':chosen['head_sha256'],
               'config':config,'source_run':str(source),'development_variant':chosen['variant'],
               'selection_rule':'highest complete positive development gain; earliest predefined variant on tie'}
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent,prefix=out.name+'.prepare-') as temporary:
        staged=Path(temporary)/'study';staged.mkdir()
        for name,value in [('protocol.json',frozen_final),('confirmation-selection.json',selection),
                           ('source-fit-audit.json',evidence),('source-development-audit.json',development),
                           ('status.json',{'status':'awaiting_confirmation'})]:write(staged/name,value)
        for path in files:shutil.copy2(path,staged/path.name)
        shutil.copy2(source/'protocol.json',staged/'source-training-protocol.json')
        staged.rename(out)
    return selection


def advance(source):
    ledger=source/'continuation-jobs.json';require(not ledger.exists(),'Existing submission ledger; inspect handles, do not duplicate')
    development=audit_development(source);write(source/'native-development/development-audit.json',development)
    if development['selected'] is None:
        result={'status':'stop_no_positive_native_candidate','goal_complete':False};write(ledger,result);return result
    selection=prepare(source,Path(str(source)+'-final'));final=Path(selection['config']['out'])
    parent=os.environ['SLURM_JOB_ID'];node=os.environ['SLURMD_NODENAME']
    def submit(arguments):
        actual=subprocess.check_output(['sbatch','--parsable',*arguments],cwd=ROOT,text=True).strip().split(';',1)[0]
        require(actual.isdecimal(),'Unexpected scheduler handle');return actual
    result={'status':'submitting_confirmation','run':str(final),'checkpoint':selection['checkpoint'],
            'candidate_head_sha256':selection['candidate_head_sha256'],'goal_complete':False}
    confirm=submit(['--partition=ws-ia',f'--nodelist={node}',f'--dependency=afterok:{parent}',
                    '--kill-on-invalid-dep=yes','--time=02:00:00','head_rl_confirm.slurm',str(final)])
    result['confirmation_job_id']=confirm;write(ledger,result)
    command=shlex.join(['.venv/bin/python','scripts/audit_score_confirmation.py','--run',str(final),
                       '--expected-head-sha256',selection['candidate_head_sha256'],'--out',str(final/'confirmation-audit.json')])
    check=submit(['--partition=ws-ia',f'--nodelist={node}',f'--dependency=afterok:{confirm}','--kill-on-invalid-dep=yes',
                  '--cpus-per-task=8','--mem=8G','--time=00:15:00','--job-name=mahjong-point-bias-audit',
                  '--output=logs/point-bias-final-audit-%j.log','--wrap',command])
    result['confirmation_audit_job_id']=check;write(ledger,result)
    smoke=submit(['--partition=ws-ia',f'--nodelist={node}',f'--dependency=afterok:{confirm}','--kill-on-invalid-dep=yes',
                  'recommend_state_smoke.slurm',selection['checkpoint'],str(final/'recommendation-interface-smoke.json')])
    result.update(status='confirmation_and_checks_submitted',recommendation_smoke_job_id=smoke);write(ledger,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    fitting=sub.add_parser('fit');fitting.add_argument('--config',type=Path,required=True)
    auditing=sub.add_parser('audit');auditing.add_argument('--run',type=Path,required=True);auditing.add_argument('--out',type=Path,required=True)
    preparing=sub.add_parser('prepare');preparing.add_argument('--source',type=Path,required=True);preparing.add_argument('--out',type=Path,required=True)
    advancing=sub.add_parser('advance');advancing.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='fit':result=fit(args.config)
    elif args.mode=='audit':
        require(not args.out.exists(),'Preserve existing projection audit');result=audit_fit(args.run);write(args.out,result)
    elif args.mode=='prepare':result=prepare(args.source,args.out)
    else:result=advance(args.run)
    print(json.dumps(result),flush=True)
