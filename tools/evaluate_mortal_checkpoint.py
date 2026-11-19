"""Complete East development comparison of one saved fresh-base checkpoint.

Reuse audited reference games; never reload that reference policy or update the
learner. This may diagnose an early checkpoint, outside the fit's registered
post-exposure selection rule. It is not independent confirmation or a release.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--step',type=int,required=True)
    p.add_argument('--out',type=Path,required=True,help='New directory inside the configured run')
    a=p.parse_args()
    if a.step<=0:raise ValueError('Require a saved nonzero optimizer step')
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    os.environ.setdefault('TORCHDYNAMO_DISABLE','1')
    if not os.environ.get('SLURM_JOB_GPUS') and os.environ.get('SLURM_STEP_GPUS'):
        os.environ['SLURM_JOB_GPUS']=os.environ['SLURM_STEP_GPUS']
    from tools.build_mortal_teacher import require_gpu,verify,sha
    require_gpu()
    import torch
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    c=json.loads(a.config.read_text());sys.path.insert(0,c['upstream_source'])
    os.environ['JEV_BASE_MODEL_DIR']=c['base_model']
    run=(ROOT/c['out']).resolve();out=(ROOT/a.out).resolve()
    if not out.is_relative_to(run) or out==run or out.exists():
        raise ValueError('Output must be a new subdirectory within this run')
    from jevmahjong.endgame.collect import write
    from jevmahjong.endgame.futures import bundle_identity
    from jevmahjong.policy.routing import load_model
    from jevmahjong.endgame.fourth_policy import FourthChaseAgent
    from jevmahjong.teachers.mortal import MortalTeacher
    from tools.train_mortal_base import cached_parent
    from tools.train_mortal_teacher import play_match
    from tools.audit_mortal_teacher_run import audit_games
    training=json.loads((run/'training-protocol-private.json').read_text());verify(training)
    runtime_files={}
    for name,key in (('cached-environment-audit-private.json','native_files'),
                     ('input-isolation-audit-private.json','source_files')):
        runtime_files.update(json.loads((run/name).read_text())[key])
    verify({'files':runtime_files})
    rows,reference_id,seeds=cached_parent(c)
    path=run/'training-run/checkpoints'/f'step-{a.step:06d}'
    meta=torch.load(path/'head.pt',map_location='cpu',weights_only=True)
    if not(meta.get('base_only_initialization') and meta['policy_version']=='mortal-base-v1'
           and meta['optimizer_step']==a.step):raise ValueError('Require this fresh-base run checkpoint')
    identity=bundle_identity(path);out.mkdir(parents=True,exist_ok=False)
    source={str(Path(__file__).resolve()):sha(__file__)}
    write(out/'protocol-private.json',dict(status='frozen_before_candidate_games',step=a.step,
        seeds=seeds,profiles=['calling','mortal'],controlled_seats=4,
        candidate_identity=identity,reference_identity=reference_id,
        cached_reference_sha256=c['reference_games_sha256'],candidate_games=64,reference_games_reused=64,
        stage='posthoc_development_not_independent_confirmation',
        registered_fit_selection_unchanged=True,no_optimizer_updates=True,
        runtime_files=runtime_files,evaluation_source=source,raw_publication=False))
    teacher=MortalTeacher(c['teacher_root']);client=load_model(path)
    client.runtime.model.eval().requires_grad_(False);agent=FourthChaseAgent(client)
    started=time.monotonic();raw=out/'games-private.jsonl'
    with raw.open('w') as f:
        for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
        f.flush()
        for seed in seeds:
            for profile in ('calling','mortal'):
                for seat in range(4):
                    row=dict(arm='candidate',**play_match(c,agent,teacher,seed,seat,profile))
                    rows.append(row);f.write(json.dumps(row,separators=(',',':'))+'\n');f.flush()
                    state=dict(status='running',step=a.step,candidate_games_completed=len(rows)-64,
                        candidate_games_total=64,reference_games_reused=64,
                        seconds=time.monotonic()-started,job_id=os.environ['SLURM_JOB_ID'])
                    write(out/'status-private.json',state)
                    if len(rows)%8==0:print(json.dumps(state),flush=True)
    del agent,client;gc.collect();torch.cuda.empty_cache()
    if bundle_identity(path)!=identity:raise ValueError('Evaluation modified the candidate')
    verify(training);verify({'files':runtime_files});verify({'files':source})
    result=audit_games(rows,seeds)
    result.update(status='saved_checkpoint_complete_east_development_finished',step=a.step,
        candidate_games=64,reference_games_reused=64,no_reference_policy_inference=True,
        candidate_identity=identity,source_files=source,runtime_files=runtime_files,
        games_sha256=sha(raw),no_optimizer_updates=True,raw_publication=False)
    write(out/'result-private.json',result)
    write(out/'status-private.json',dict(status='complete',step=a.step,candidate_games_completed=64,
        reference_games_reused=64,release_qualified=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_files','runtime_files','candidate_identity')}),flush=True)


if __name__=='__main__':main()
