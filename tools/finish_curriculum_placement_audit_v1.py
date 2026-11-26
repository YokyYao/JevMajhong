"""Preserve the actual fit terminal, then independently audit placement evidence."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only,write
from tools.prepare_curriculum_placement_launch_v1 import terminal


def finish(registered,fit_job):
    gpu_only();run=registered.parent.parent;launch=read(run/'launch-binding-private.json')
    if launch['registered_sha256']!=sha(registered):raise ValueError('Original launch registration differs')
    if any(sha(p)!=h for p,h in launch['files'].items()):raise ValueError('Launch inputs changed')
    receipt_path=ROOT/'logs'/f'curriculum-placement-course-exit-{fit_job}.json';receipt=read(receipt_path)
    if receipt['phase']!='fit':raise ValueError('Actual successful full FIT receipt required')
    terminal(fit_job,receipt,ROOT/'cluster/curriculum-placement-course-v1.slurm',run/'fit-terminal-scheduler-private.json')
    from tools.audit_curriculum_placement_course_v1 import audit
    evidence_path=run/'final-evidence-private.json';audit(registered,evidence_path)
    if any(sha(p)!=h for p,h in launch['files'].items()):raise ValueError('Launch changed during audit')
    evidence=read(evidence_path)
    write(run/'audit-execution-private.json',dict(status='actual_full_placement_evidence_independently_audited',
        actual_fit_job=fit_job,original_launch_binding_unchanged=True,
        development_grade_passed=evidence['development_assessment']['passed'],strategy_grade_overridden=False,
        independent_v8_confirmation_performed=False,automatic_release=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registered',type=Path,required=True)
    p.add_argument('--fit-job',type=int,required=True);a=p.parse_args();finish(a.registered.resolve(),a.fit_job)
