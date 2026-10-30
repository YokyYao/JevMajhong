"""Advance only a fully audited positive native variant to the fixed final study."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from head_rl import write
from scripts.audit_diagonal_value import audit_development
from scripts.prepare_diagonal_confirmation import prepare


def submit(arguments):
    job=subprocess.check_output(['sbatch','--parsable',*arguments],cwd=ROOT,text=True).strip()
    if not job.isdecimal():raise ValueError(f'Unexpected scheduler handle:{job}')
    return job


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    source=args.run;final=Path(str(source)+'-final')
    ledger=source/'continuation-jobs.json'
    if ledger.exists():raise ValueError('Existing continuation submissions; inspect their handles, do not duplicate')
    development=audit_development(source)
    write(source/'native-development'/'development-audit.json',development)
    if development['selected'] is None:
        write(ledger,{'status':'stop_no_positive_native_candidate','development_audit':'passed','goal_complete':False})
        print('No positive native variant. Stop this recipe; the broader training improvement goal remains active.',flush=True)
        return
    dependency=os.environ['SLURM_JOB_ID'];node=os.environ['SLURMD_NODENAME']
    selection=prepare(source,final)
    jobs={'status':'submitting_confirmation','run':str(final),'checkpoint':selection['checkpoint'],
          'candidate_head_sha256':selection['candidate_head_sha256'],'goal_complete':False}
    # Persist each successful handle before the next submission. An observer or
    # later failure must never cause the expensive confirmation to be duplicated.
    confirm=submit(['--partition=ws-ia',f'--nodelist={node}',f'--dependency=afterok:{dependency}',
                    '--kill-on-invalid-dep=yes','--time=02:00:00','head_rl_confirm.slurm',str(final)])
    jobs['confirmation_job_id']=confirm;write(ledger,jobs)
    audit_command=shlex.join(['.venv/bin/python','scripts/audit_score_confirmation.py','--run',str(final),
                             '--expected-head-sha256',selection['candidate_head_sha256'],
                             '--out',str(final/'confirmation-audit.json')])
    audit=submit(['--partition=ws-ia',f'--nodelist={node}',f'--dependency=afterok:{confirm}',
                  '--kill-on-invalid-dep=yes','--cpus-per-task=8','--mem=8G','--time=00:15:00',
                  '--job-name=mahjong-diagonal-confirm-audit','--output=logs/diagonal-confirm-audit-%j.log',
                  '--wrap',audit_command])
    jobs['confirmation_audit_job_id']=audit;write(ledger,jobs)
    smoke=submit(['--partition=ws-ia',f'--nodelist={node}',f'--dependency=afterok:{confirm}',
                  '--kill-on-invalid-dep=yes','recommend_state_smoke.slurm',selection['checkpoint'],
                  str(final/'recommendation-interface-smoke.json')])
    jobs.update(status='confirmation_and_checks_submitted',recommendation_smoke_job_id=smoke);write(ledger,jobs)
    print(json.dumps(jobs),flush=True)


if __name__=='__main__':main()
