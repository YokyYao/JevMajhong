"""Freeze a fresh single-factor RI-loss control before any optimizer updates."""
import argparse
import copy
import json
import os
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read,sha
from tools.curriculum_riichi_control import METHOD,require_recipe,require_registered


def register(config,base_path,reference_path):
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS')
        or socket.gethostname().startswith(('lo-','login'))):raise RuntimeError('Allocated scheduler GPU required')
    c=read(config);control_path=(ROOT/c['control_registration']).resolve();control=read(control_path)
    require_recipe(c,control)
    if any(sha(p)!=h for p,h in control['inputs'].items()):raise ValueError('Original control inputs changed')
    from tools.register_curriculum_basic_repair import register as original_register
    original_register(config,base_path,reference_path)
    path=ROOT/c['plan_root']/'registered.json';reg=read(path)
    # This new run has not started. Preserve the original generic registration
    # before explicitly freezing the changed loss implementation and config.
    original_path=path.with_name('generic-registration-private.json')
    with original_path.open('x') as f:json.dump(reg,f,indent=2);f.write('\n')
    cfg=copy.deepcopy(reg['training_config']);cfg.update(loss_method=METHOD,control_registration=str(control_path))
    cfg_path=path.parent/'training-config.json';cfg_path.write_text(json.dumps(cfg,indent=2)+'\n')
    reg.update(status='stage1_riichi_class_control_registered_not_started',training_config=cfg)
    paths=[Path(__file__),config,control_path,original_path,cfg_path,
        ROOT/'tools/train_curriculum_riichi_control.py',ROOT/'tools/audit_curriculum_riichi_control.py',
        ROOT/'tools/curriculum_riichi_control.py',ROOT/'tools/curriculum_riichi_class_loss.py',
        ROOT/'tests/core/test_curriculum_riichi_control.py',ROOT/'tests/core/test_curriculum_riichi_class_loss.py',
        ROOT/'docs/experiments/plans/curriculum-riichi-class-loss-control-20261122.md',
        ROOT/'docs/experiments/plans/curriculum-riichi-control-execution-20261123.md']
    control_fit=ROOT/control['curriculum']['out'];checkpoint=control_fit/'training-run/checkpoints/step-000000'
    for name in ('adapter_model.safetensors','head.pt'):paths.append(checkpoint/name)
    paths.extend([control_fit/'training-run/training_eval/step-000000/development/rows.json',
        control_fit/'evaluations/step-000000/protocol.json',control_fit/'evaluations/step-000000/summary.json',
        control_fit/'evaluations/step-000000/matches-private.jsonl'])
    reg['inputs'].update({str(p.resolve()):sha(p) for p in paths})
    require_registered(reg)
    path.write_text(json.dumps(reg,indent=2)+'\n')
    print(json.dumps(dict(status=reg['status'],registered_final_step=608,source_weights_unchanged=True,
        actual_training_started=False,conditional_final_failure_launch_required=True)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','base-path','reference-path'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();register(a.config.resolve(),a.base_path.resolve(),a.reference_path.resolve())
