"""Collect fresh paired point labels under a preserved, completely frozen teacher."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from head_rl import fingerprint,write
from single_decision_rl import SingleDecisionPolicy
from scripts.train_single_decision import intervention_advantages
from scripts.train_diagonal_value import BASE_FILES


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text());out=ROOT/config['out']
    if out.exists():raise ValueError('Preserve partial or complete datasets; never duplicate collection')
    teacher=ROOT/config['reference'];head_sha=fingerprint(teacher/'head.pt')
    if head_sha!=config['frozen_teacher_head_sha256']:raise ValueError('Preserved teacher changed')
    proposed=set(range(config['training_seed_start'],config['training_seed_start']+config['training_seeds_per_stage']))
    for path in (ROOT/'runs').glob('*/protocol.json'):
        previous=json.loads(path.read_text()).get('config',{})
        for start,length in [('training_seed_start','training_seeds_per_stage'),('development_seed_start','development_seeds'),
                             ('confirmation_seed_start','confirmation_seeds')]:
            if start in previous and length in previous:
                count=previous.get('max_stages',1) if start=='training_seed_start' else 1
                if proposed&set(range(previous[start],previous[start]+previous[length]*count)):
                    raise ValueError(f'Training wall range already declared:{path}')
    out.mkdir(parents=True)
    code=['head_rl.py','arena.py','semantic.py','seed_control.py','single_decision_rl.py',
          'scripts/train_single_decision.py','scripts/collect_diagonal_data.py','scripts/train_head_rl.py',
          'vendor/JevAny/jevany/model.py','vendor/JevAny/jevany/checkpoint.py']
    protocol={'config':config,'sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
              'code_tree_sha':subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip(),
              'code_sha256':{name:fingerprint(ROOT/name) for name in code},'reference_head_sha256':head_sha,
              'reference_adapter_sha256':fingerprint(teacher/'adapter_model.safetensors'),
              'training_data_mode':'fresh one-decision Q_frozen_teacher pairs; no actor updates during collection'}
    write(out/'protocol.json',protocol)
    initial=out/'checkpoints'/'update-000000';initial.mkdir(parents=True)
    for name in ['head.pt',*BASE_FILES]:shutil.copy2(teacher/name,initial/name)
    random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    deadline=time.monotonic()+60*config['collection_wall_minutes']
    write(out/'status.json',{'status':'loading_frozen_teacher','actor_updates':0})
    policy=SingleDecisionPolicy(config);policy.head.requires_grad_(False)
    games,controls,rows=[],[],[]
    for seed in sorted(proposed):
        for seat in range(4):
            if time.monotonic()>=deadline:raise TimeoutError('Frozen collection deadline reached')
            control,_=policy.game(seed,seat,training=False,frozen=True,old_opponent=bool(seed%2),replica=-1)
            controls.append(control)
            for replica in range(config['training_replicas']):
                if time.monotonic()>=deadline:raise TimeoutError('Frozen collection deadline reached')
                game,trace=policy.game(seed,seat,training=True,old_opponent=bool(seed%2),replica=replica)
                games.append(game);rows.extend(trace)
                print(json.dumps({'phase':'fresh_paired_data','controls':len(controls),'proposals':len(games),
                                  'states':len(rows),'point_delta':game['terminal_point_delta']}),flush=True)
        write(out/'status.json',{'status':'collecting_frozen_pairs','completed_streams':seed-config['training_seed_start']+1,
                                 'controls':len(controls),'proposals':len(games),'states':len(rows),'actor_updates':0})
        if (seed-config['training_seed_start']+1)%32==0:
            write(out/'partial-games.json',games);write(out/'partial-controls.json',controls)
            torch.save(rows,out/'partial-rollout.pt')
    diagnostics=intervention_advantages(controls,games,rows,config['training_replicas'],config['advantage_scale'])
    if fingerprint(teacher/'head.pt')!=head_sha:raise ValueError('Frozen teacher modified during collection')
    if policy.checks<1:raise ValueError('Native feature parity was not exercised')
    write(out/'training-games-stage-01.json',games);write(out/'reference-controls-stage-01.json',controls)
    torch.save(rows,out/'rollout-stage-01.pt');write(out/'rloo-groups-stage-01.json',diagnostics)
    result={'status':'ready_for_fit' if diagnostics['informative_groups']>=config['minimum_informative_groups'] else 'insufficient_signal',
            'controls':len(controls),'proposals':len(games),'states':len(rows),'actor_updates':0,
            'native_parity_checks':policy.checks,'teacher_unchanged':True,'diagnostics':diagnostics}
    write(out/'collection-result.json',result);write(out/'status.json',result)
    if result['status']!='ready_for_fit':raise ValueError('Insufficient informative groups; no unplanned resampling')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
