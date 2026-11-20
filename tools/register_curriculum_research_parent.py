"""Freeze a prospective research-only parent amendment before final games."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import datetime

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def register(policy,out):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-','login'))
    value=read(policy);assert value['schema']=='curriculum-research-parent-policy-v2'
    reference=ROOT/'runs/curriculum-optimization-v2/reference-riichi-gate-private.json';riichi=read(reference)
    assert riichi['counts']==value['riichi']['reference_counts'] and riichi['windows']==107
    assert riichi['checkpoint_sha256']==value['riichi']['reference_identity']
    baseline=ROOT/'runs/curriculum-reproduction-v1/stage1-fit/evaluations/step-001162/summary.json';old=read(baseline)
    limits=value['complete_match_development']
    assert old['checkpoint_sha256']==limits['baseline_identity']
    for name,limit in [('top2_rate','min_top2_rate'),('net_points','min_net_points'),('dealin_hand_rate','max_dealin_hand_rate')]:
        assert old[name]==limits[limit]
    paths=[policy.resolve(),reference,baseline,Path(__file__).resolve()];observations=[]
    for name in ('curriculum-optimization-v2','curriculum-context-v3'):
        path=ROOT/'runs'/name/'fit-plan/registered.json';reg=read(path);c=reg['curriculum'];fit=ROOT/c['out']
        assert c['run_id'] in value['applies_to_run_ids'] and c['candidate_rule']==value['candidate_rule']
        assert reg['registered_final_step']==584 and c['curve_seed_start']==limits['seed_start']
        assert c['opponent_profiles']==limits['profiles'] and c['stage_end_seed_groups']==limits['seed_groups']
        assert not (fit/'evaluations/step-000584').exists() and not (fit/'completion.json').exists()
        raw=(fit/'metrics/train_metrics.jsonl').read_text();lines=raw.splitlines()
        if raw and not raw.endswith('\n'):lines=lines[:-1]
        step=json.loads(lines[-1])['local_step'];assert step<584
        observations.append(dict(run_id=c['run_id'],observed_updates=step,registered_final=584,final584_games_not_started=True))
        paths.append(path)
    result=dict(status='research_parent_amendment_bound_before_final_games',node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'],
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),policy=str(policy.resolve()),
        bound_files={str(p.resolve()):sha(p) for p in paths},observations=observations,
        original_stage1_gate_and_audits_unchanged=True,parent_admitted=False,child_fit_started=False,
        development_inspection_amendment_not_original_preregistration=True,full_goal_not_achieved=True)
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();register(a.policy,a.out)
