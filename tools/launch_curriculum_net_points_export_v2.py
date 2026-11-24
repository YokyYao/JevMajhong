"""Queue an audited post-fit net-point export without changing the fit recipe."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource, scheduler, fields
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_transfer import read, sha

RUN = ROOT/'runs/curriculum-progression-parent-retention-v1'
CONTROL = ROOT/'runs/curriculum-progression-riichi-paired-v1'
PREFLIGHT = CONTROL/'net-points-export-v2-preflight-v1'
ENTRY = ROOT/'cluster/curriculum-net-points-export-v2.slurm'
BINDING = RUN/'net-points-export-v2-binding-private.json'
QUEUE = RUN/'net-points-export-v2-queue-private.json'
PUBLIC = ('tools/plot_curriculum_net_points_v2.py', 'tools/launch_curriculum_net_points_export_v2.py',
          'cluster/curriculum-net-points-export-v2.slurm',
          'docs/experiments/plans/curriculum-net-points-export-20261124.md')


def launch():
    allocation = resource()
    if BINDING.exists() or QUEUE.exists(): raise ValueError('Never submit duplicate post-fit exports')
    from tools.curriculum_progression_parent_runtime_v1 import require_registration
    registered = RUN/'fit-plan/registered.json'; reg = read(registered); require_registration(reg)
    actual_queue = read(CONTROL/'parent-retention-queue-v1-private.json')
    fit_job, audit_job = actual_queue['fit_job'], actual_queue['audit_job']
    fit, audit = scheduler(fit_job), scheduler(audit_job)
    fv, av = fields(fit.stdout), fields(audit.stdout)
    if (fit.returncode or audit.returncode or fv.get('JobState') != 'RUNNING'
            or fv.get('Command') != str(ROOT/'cluster/curriculum-progression-parent-retention-v1.slurm')
            or av.get('Command') != fv.get('Command') or av.get('JobState') != 'PENDING'
            or f'afterok:{fit_job}(' not in av.get('Dependency', '')
            or read(RUN/'launch-binding-private.json')['registered_sha256'] != sha(registered)):
        raise ValueError('Bind export during the actual immutable fit with its original pending audit')
    pf = read(PREFLIGHT/'aggregate-provenance.json')
    bound = read(PREFLIGHT/'completion-binding-private.json')['files']
    if (pf['status'] != 'actual_completed_fixed_grid_net_points_plotted_not_confirmation'
            or pf['curve_local_steps'] != [0, 600, 1200, 1800, 2400]
            or pf['curve_matches_per_point'] != 16 or pf['final_comparison_matches_per_policy'] != 64
            or pf['new_games_model_forwards_or_optimizer_updates'] != 0
            or any(sha(p) != h for p, h in bound.items())):
        raise ValueError('Actual complete historical-grid export preflight required')
    subprocess.run(['bash', '-n', str(ENTRY)], check=True, timeout=10)
    files = {str((ROOT/p).resolve()): sha(ROOT/p) for p in PUBLIC}
    for name in ('tools/curriculum_progression_runtime_v1.py',
                 'tools/curriculum_progression_parent_runtime_v1.py',
                 'tools/curriculum_progression_objective_v1.py',
                 'tools/curriculum_progression_parent_loss_v1.py',
                 'tools/plot_curriculum_fixed_grid.py', 'tools/plot_curriculum.py',
                 'jevmahjong/training/metric_journal.py',
                 'tools/capture_curriculum_paired_terminal_v2.py',
                 'tools/curriculum_transfer.py', 'tools/curriculum_calls_post_reach_data_v1.py'):
        files[str(ROOT/name)] = sha(ROOT/name)
    for p in (registered, PREFLIGHT/'aggregate-provenance.json', PREFLIGHT/'completion-binding-private.json'):
        files[str(p)] = sha(p)
    write(BINDING, dict(status='actual_parent_retention_post_audit_net_export_bound',
        fit_job=fit_job, audit_job=audit_job, observed_live_fit=fit.stdout, observed_pending_audit=audit.stdout,
        files=files, allocation=allocation, active_training_recipe_modified=False,
        automatic_publication_or_release=False, model_forwards_or_updates=0))
    submitted = subprocess.run(['sbatch', '--parsable', '--hold', '--dependency=afterok:'+str(audit_job),
        '--kill-on-invalid-dep=yes', str(ENTRY)], cwd=ROOT, capture_output=True, text=True, timeout=30)
    job = submitted.stdout.strip().split(';')[0]
    if submitted.returncode or not job.isdigit():
        write(RUN/'net-points-export-v2-submit-error-private.json', dict(stdout=submitted.stdout,
            stderr=submitted.stderr, returncode=submitted.returncode))
        raise ValueError('No actual scheduler identity returned')
    try:
        actual = scheduler(job); values = fields(actual.stdout)
        if (actual.returncode or values.get('Command') != str(ENTRY) or values.get('Reason') != 'JobHeldUser'
                or f'afterok:{audit_job}(' not in values.get('Dependency', '')):
            raise ValueError('Actual held export job or dependency differs')
        write(QUEUE, dict(status='actual_post_audit_net_export_job_bound_while_held',
            job_id=job, audit_job=audit_job, binding_sha256=sha(BINDING), observed_scheduler=actual.stdout))
        subprocess.run(['scontrol', 'release', job], check=True, capture_output=True, text=True, timeout=20)
    except BaseException:
        subprocess.run(['scancel', job], check=False, capture_output=True, text=True, timeout=20)
        raise
    print(json.dumps(dict(status='actual_post_audit_net_export_submitted', job_id=job,
        after_successful_audit=audit_job, new_games_or_updates=0)), flush=True)


def render():
    resource(); binding = read(BINDING); queue = read(QUEUE)
    import os
    if (str(queue['job_id']) != os.environ['SLURM_JOB_ID']
            or queue['binding_sha256'] != sha(BINDING)
            or any(sha(p) != h for p, h in binding['files'].items())):
        raise ValueError('Original registered post-audit export job and unchanged implementation required')
    from tools.plot_curriculum_net_points_v2 import render as draw
    draw(RUN/'fit-plan/registered.json', RUN/'net-points-final2400-v2')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--phase', choices=('launch', 'render'), required=True)
    a = p.parse_args()
    if a.phase == 'launch': launch()
    else: render()
