"""Fit paired net-point values on frozen caches and export native Jev heads."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from jevany.model import PointerHead
from diagonal_value import cross_validate,folded_head,pointer_products,ridge_fit
from head_rl import deployment_greedy_index,fingerprint,mixed_probabilities,reference_advantages,write
from scripts.prepare_preserved_incumbent import validate_frozen_runtime

BASE_FILES=['adapter_config.json','adapter_model.safetensors','tokenizer.json','tokenizer_config.json','chat_template.jinja']
ORIGINAL_HEAD='3c586e1bf6e86892c29a1f76bbab4fda4f714b17a31f87c6fe871157748a7d5b'


def pointer(meta,device):
    if meta['head_residual_dim']!=0 or meta['temperature']!=1.:raise ValueError('Plain temperature1 head required')
    dp,d=meta['head']['q.weight'].shape
    h=PointerHead(d,dp).to(device).eval().requires_grad_(False);h.load_state_dict(meta['head']);return h


@torch.no_grad()
def load_contrasts(config,base_meta,base_head):
    matrices,targets,seeds,states,provenance=[],[],[],[],[]
    seen_features=set();seen_episodes=set();all_streams=set();all_count=0;nonzero=0
    original=ROOT/config['reference'];device=next(base_head.parameters()).device
    for source_name in config['training_sources']:
        source=ROOT/source_name;protocol=json.loads((source/'protocol.json').read_text());cfg=protocol['config']
        for n,h in protocol['code_sha256'].items():
            if fingerprint(ROOT/n)!=h:raise ValueError(f'Source code changed:{n}')
        teacher_path=ROOT/cfg['reference']
        if fingerprint(teacher_path/'head.pt')!=protocol['reference_head_sha256']:raise ValueError('Source teacher changed')
        teacher_meta=torch.load(teacher_path/'head.pt',map_location='cpu',weights_only=False)
        validate_frozen_runtime(teacher_meta,base_meta)
        for name in BASE_FILES:
            if fingerprint(teacher_path/name)!=fingerprint(original/name):raise ValueError('Cached backbone inputs differ')
        teacher=pointer(teacher_meta,device)
        games_path=source/'training-games-stage-01.json';controls_path=source/'reference-controls-stage-01.json';features_path=source/'rollout-stage-01.pt'
        feature_sha=fingerprint(features_path)
        if feature_sha in seen_features:raise ValueError('Duplicate cache would double-count training data')
        seen_features.add(feature_sha)
        games=json.loads(games_path.read_text());controls=json.loads(controls_path.read_text())
        rows=torch.load(features_path,map_location='cpu',weights_only=False)
        cached_advantages=[row['advantage'] for row in rows]
        reference_advantages(controls,games,rows,cfg['training_replicas'],cfg['advantage_scale'])
        if any(abs(previous-row['advantage'])>1e-9 for previous,row in zip(cached_advantages,rows)):
            raise ValueError('Saved paired advantages differ from reconstructed returns')
        episodes={(g['seed'],g['seat'],g['replica']):g for g in games};fixed={(g['seed'],g['seat']):g for g in controls}
        for row in rows:
            key=tuple(row['episode_key'])
            if key in seen_episodes:raise ValueError('Duplicated episode across cache sources')
            seen_episodes.add(key);all_streams.add(key[0]);all_count+=1
            decide,options=row['decide'].to(device),row['options'].to(device)
            teacher_logits=teacher(decide,options);p=mixed_probabilities(teacher_logits,row['eligible'],cfg['exploration_uniform'])
            if not torch.allclose(p.cpu(),row['reference'],atol=2e-5,rtol=2e-4):raise ValueError('Cached behavior mismatch')
            if abs(float(p[row['action']].log())-row['old_logp'])>2e-4:raise ValueError('Behavior likelihood mismatch')
            if deployment_greedy_index(teacher_logits,row['eligible'])!=row['teacher_action']:raise ValueError('Frozen control action mismatch')
            value=(episodes[key]['score']-fixed[key[:2]]['score'])/1000.
            if abs(value/10-row['advantage'])>1e-9:raise ValueError('Paired point target mismatch')
            a,t=row['eligible'][row['action']],row['eligible'][row['teacher_action']]
            logits=base_head(decide,options)
            states.append({'decide':row['decide'],'options':row['options'],'eligible':row['eligible'],
                           'original_logits':logits.cpu()})
            if a==t:
                if value!=0:raise ValueError('Unchanged action has a nonzero target')
                continue
            products=pointer_products(base_head,decide,options)
            matrices.append((products[a]-products[t]).double().cpu());targets.append(value);seeds.append(key[0])
            nonzero+=value!=0
        provenance.append({'source':source_name,'teacher_head_sha256':protocol['reference_head_sha256'],
                           'rollout_sha256':feature_sha,'games_sha256':fingerprint(games_path),
                           'controls_sha256':fingerprint(controls_path),'states':len(rows)})
    if not matrices:raise ValueError('No changed-action contrasts')
    for prefix,count in [('development',config['development_seeds']),('confirmation',config['confirmation_seeds'])]:
        start=config[prefix+'_seed_start']
        if all_streams&set(range(start,start+count)):raise ValueError('Training and evaluation overlap')
    return torch.stack(matrices),torch.tensor(targets,dtype=torch.float64),seeds,states,{
        'raw_states':all_count,'changed_action_contrasts':len(matrices),'nonzero_point_targets':nonzero,
        'independent_wall_streams':len(all_streams),'sources':provenance,'new_simulated_training_matches':0}


@torch.no_grad()
def policy_change(head,states):
    device=next(head.parameters()).device;kl=0.;flips=0
    for row in states:
        logits=head(row['decide'].to(device),row['options'].to(device));eligible=row['eligible']
        p=mixed_probabilities(logits,eligible,.15);q=mixed_probabilities(row['original_logits'].to(device),eligible,.15)
        kl+=float((p*(p.log()-q.log())).sum())
        flips+=deployment_greedy_index(logits,eligible)!=deployment_greedy_index(row['original_logits'],eligible)
    return {'mean_guarded_mixture_kl':kl/len(states),'cached_greedy_flips':flips,'states':len(states)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    config=json.loads(args.config.read_text());out=ROOT/config['out'];original=ROOT/config['reference']
    if out.exists():raise ValueError('Preserve completed or partial experiments')
    if fingerprint(original/'head.pt')!=ORIGINAL_HEAD:raise ValueError('Training prior must be originalSFT2000')
    out.mkdir(parents=True)
    code=['diagonal_value.py','scripts/train_diagonal_value.py','head_rl.py','arena.py','semantic.py','seed_control.py',
          'scripts/train_head_rl.py','scripts/confirm_head_rl.py','scripts/prepare_preserved_incumbent.py',
          'vendor/JevAny/jevany/model.py']
    protocol={'config':config,'sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
              'code_tree_sha':subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip(),
              'code_sha256':{n:fingerprint(ROOT/n) for n in code},'reference_head_sha256':ORIGINAL_HEAD,
              'reference_adapter_sha256':fingerprint(original/'adapter_model.safetensors')}
    write(out/'protocol.json',protocol);write(out/'status.json',{'status':'extracting_frozen_pointer_contrasts'})
    meta=torch.load(original/'head.pt',map_location='cpu',weights_only=False);head=pointer(meta,'cuda')
    x,y,seeds,states,data=load_contrasts(config,meta,head)
    write(out/'data-audit.json',{'audit_status':'passed',**data})
    report=cross_validate(x,y,seeds,config['ridge_alphas'],config['cross_validation_folds'])
    write(out/'cross-validation.json',report)
    if not report['improves_over_zero']:
        write(out/'training-result.json',{'status':'stop_no_predictive_signal','candidates':[],'cross_validation':report,'data':data})
        write(out/'status.json',{'status':'stop_no_predictive_signal'});print(json.dumps({'status':'stop_no_predictive_signal',**report}),flush=True);return
    coefficient=ridge_fit(x,y,report['best_alpha']);torch.save({'coefficient':coefficient,'x':x,'y':y,'seeds':seeds},out/'private-regression.pt')
    candidates=[];seen_weights=set()
    for index,requested in enumerate(config['calibration_strengths'],1):
        strength=min(requested,config['max_relative_q_change']/max(1e-12,float(coefficient.abs().max())))
        for _ in range(16):
            candidate=folded_head(head,coefficient,strength);change=policy_change(candidate,states)
            if change['mean_guarded_mixture_kl']<=config['max_reference_kl']:break
            strength*=.5
        else:continue
        weights_sha=hashlib.sha256(candidate.q.weight.detach().cpu().numpy().tobytes()+candidate.q.bias.detach().cpu().numpy().tobytes()).hexdigest()
        if weights_sha in seen_weights or not change['cached_greedy_flips']:continue
        seen_weights.add(weights_sha);checkpoint=out/'checkpoints'/f'calibration-{index:02d}';checkpoint.mkdir(parents=True)
        for name in BASE_FILES:shutil.copy2(original/name,checkpoint/name)
        payload=dict(meta);payload['head']={k:v.detach().cpu().clone() for k,v in candidate.state_dict().items()}
        payload.update(training_method='ridge paired-point diagonal pointer calibration',parent=str(original),
                       learned_degrees_of_freedom=len(coefficient),regression_alpha=report['best_alpha'],
                       calibration_strength=strength,protocol_sha256=protocol['sha256'])
        torch.save(payload,checkpoint/'head.pt')
        complete={'head_sha256':fingerprint(checkpoint/'head.pt'),'protocol_sha256':protocol['sha256'],
                  'requested_strength':requested,'actual_strength':strength,'learned_degrees_of_freedom':len(coefficient),**change}
        write(checkpoint/'complete.json',complete)
        candidates.append({'checkpoint':str(checkpoint.relative_to(ROOT)),**complete})
    for i,source in enumerate(data['sources'],1):
        shutil.copy2(ROOT/source['source']/'training-games-stage-01.json',out/f'training-games-stage-cache-{i:02d}.json')
    result={'status':'fit_complete' if candidates else 'stop_no_observed_policy_change','candidates':candidates,
            'cross_validation':report,'data':data,'original_head_unchanged':fingerprint(original/'head.pt')==ORIGINAL_HEAD}
    write(out/'training-result.json',result);write(out/'status.json',{'status':result['status']})
    print(json.dumps({'status':result['status'],'data':data,'cross_validation_best_mse':report['best_mse'],
                      'zero_predictor_mse':report['zero_predictor_mse'],'candidates':candidates}),flush=True)


if __name__=='__main__':main()
