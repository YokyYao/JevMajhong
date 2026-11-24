"""Queue one finite recovery after the real original shard terminates."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools import curriculum_targeted_win_resumed_pipeline_v3 as pipeline
from tools import curriculum_targeted_win_resume_v3 as recovery
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.capture_curriculum_paired_terminal_v2 import resource, scheduler, fields


def queue(shard, original_job):
    allocation = resource(); pipeline.verify_entry()
    if type(shard) is not int or shard not in range(4): raise ValueError('Fixed original shard required')
    target = recovery.RUN/f'shard-{shard:02d}'/'attempt-001'
    receipt = recovery.RUN/f'shard-{shard:02d}-attempt-001-queue-private.json'
    if receipt.exists() or target.exists(): raise ValueError('Never duplicate an attempt or queued recovery')
    observed = scheduler(original_job); values = fields(observed.stdout)
    expected_entry = ROOT/'cluster/curriculum-targeted-win-labels-v1.slurm'
    if (observed.returncode or values.get('JobId') != str(original_job)
            or values.get('ArrayJobId') != '319759' or values.get('ArrayTaskId') != str(shard)
            or values.get('JobState') != 'RUNNING' or values.get('Command') != str(expected_entry)):
        raise ValueError('A real live original fixed array shard is required before queuing')
    prior = recovery.previous.RUN/'shards'/f'shard-{shard:02d}'
    launch = read(prior/'registered-private.json')
    if str(launch['job_id']) != str(original_job): raise ValueError('Actual shard writer differs')
    args = ['sbatch', '--hold', '--parsable', '--chdir='+str(ROOT),
        '--dependency=afterany:'+str(original_job), '--kill-on-invalid-dep=yes',
        str(recovery.ENTRY), '--shard', str(shard), '--attempt', '1', '--prior', str(prior)]
    created = None; released = False
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30, check=True)
        created = result.stdout.strip().split(';')[0]
        if not created.isdigit(): raise ValueError('Actual numeric recovery job ID required')
        held = scheduler(created); state = fields(held.stdout)
        tres = dict(t.split('=', 1) for t in state.get('ReqTRES', '').split(',') if '=' in t)
        if (held.returncode or state.get('JobState') != 'PENDING'
                or state.get('Command') != str(recovery.ENTRY)
                or 'afterany:'+str(original_job)+'(' not in state.get('Dependency', '')
                or int(tres.get('gres/gpu', '0')) != 1 or state.get('NumCPUs') != '4'
                or state.get('TimeLimit') != '12:00:00' or state.get('KillOInInvalidDependent') != 'Yes'):
            raise ValueError('Actual held recovery resources, timeout and dependency must match')
        write(receipt, dict(status='held_complete_world_recovery_queued_after_actual_original_terminal',
            job_id=created, shard=shard, original_job=str(original_job), args=args,
            original_live_observation=observed.stdout, held_scheduler_observation=held.stdout,
            recovery_binding_sha256=sha(recovery.BINDING), pipeline_binding_sha256=sha(pipeline.BINDING),
            launcher_sha256=sha(__file__), submission_allocation=allocation,
            current_writer_not_modified=True, no_polling=True, actual_prior_terminal_required_at_execution=True,
            automatic_fit=False, student_updates=0))
        subprocess.run(['scontrol', 'release', created], capture_output=True, text=True, timeout=20, check=True)
        released = True
        current = scheduler(created)
        write(receipt.with_name(receipt.stem+'-release.json'), dict(job_id=created,
            scheduler_returncode=current.returncode, raw=current.stdout, stderr=current.stderr))
        print(json.dumps(dict(status='complete_world_recovery_queued_not_started', job_id=created,
            shard=shard, afterany_original_job=str(original_job), student_updates=0)), flush=True)
    finally:
        if created and created.isdigit() and not released:
            subprocess.run(['scancel', created], capture_output=True, text=True, timeout=20, check=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--shard', type=int, required=True)
    p.add_argument('--original-job', type=int, required=True); a = p.parse_args()
    queue(a.shard, a.original_job)
