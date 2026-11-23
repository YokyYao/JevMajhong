"""Require actual successful GPU confirmation terminal before the final audit."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read
from tools.curriculum_calls_post_reach_data_v1 import gpu_only,write
from tools.prepare_curriculum_placement_launch_v1 import terminal


def finish(registered,run_job):
    gpu_only();reg=read(registered);directory=Path(reg['out']).resolve()
    receipt=read(ROOT/'logs'/f'curriculum-confirmation-run-exit-{run_job}.json')
    terminal(run_job,receipt,ROOT/'cluster/curriculum-confirmation-run-v1.slurm',
        directory/'runner-terminal-scheduler-private.json')
    opened=read(directory/'opened-private.json')
    if str(opened['job_id'])!=str(run_job):raise ValueError('Actual outcomes came from a different confirmation job')
    from tools.audit_curriculum_confirmation_v1 import audit
    audit(registered,directory/'independent-evidence-private.json')
    write(directory/'audit-execution-private.json',dict(status='actual_confirmation_runner_terminal_and_evidence_audited',
        actual_confirmation_job=run_job,automatic_release=False,task_completion_authorized=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registered',type=Path,required=True)
    p.add_argument('--run-job',type=int,required=True);a=p.parse_args();finish(a.registered.resolve(),a.run_job)
