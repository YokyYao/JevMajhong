"""Preserve real final3B terminal evidence before placement data is ready."""
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_transfer import read, sha
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit


def accounting_proof(raw, receipt, job, script, root=ROOT):
    rows = [line.split('|') for line in raw.splitlines() if line.strip()]
    matched = [row for row in rows if row[0] == str(job)]
    if len(matched) != 1 or len(matched[0]) != 8:
        raise ValueError('One exact complete accounting row required')
    jid, state, exit_code, node, tres, start, end, command = matched[0]
    values = dict(v.split('=', 1) for v in tres.split(',') if '=' in v)
    words = shlex.split(command)
    submitted = (Path(words[-1]) if words else Path('/missing'))
    submitted = submitted if submitted.is_absolute() else root/submitted
    if (str(receipt.get('job_id')) != jid or receipt.get('exit_code') != 0
            or state != 'COMPLETED' or exit_code != '0:0'
            or not node or node.startswith(('lo-', 'login'))
            or values.get('gres/gpu', '0').isdigit() is False or int(values.get('gres/gpu', '0')) < 1
            or any(v in ('', 'Unknown', 'N/A', '(null)') for v in (start, end))
            or submitted.resolve() != script.resolve()):
        raise ValueError('Actual successful GPU accounting with original submitted script required')
    return dict(source='sacct_original_row', raw=raw, receipt=receipt,
        actual_submitted_script_checked=True, synthesized_scontrol_record=False)


def capture():
    gpu_only(); config_path = ROOT/'configs/training/curriculum-yaku-terminal-capture-v1.json'; c = read(config_path)
    if c['status'] != 'planned_terminal_capture_not_child_parent_or_fit_admission' or c['automatic_fit']:
        raise ValueError('Terminal collection only; no child admission')
    run = ROOT/c['parent_run']; registered = run/'fit-plan/registered.json'; reg = read(registered)
    fit = ROOT/reg['curriculum']['out']; completion_path = fit/'completion.json'
    evidence_path = run/'final-evidence-private.json'; execution_path = run/'audit-execution-private.json'
    done = read(completion_path); evidence = read(evidence_path); execution = read(execution_path)
    if (reg['registered_final_step'] != 2500 or done['optimizer_steps'] != 2500
            or done['status'] != 'yaku2500_fit_and_real_curves_complete'
            or evidence['status'] != 'yaku2500_final_evidence_audit_passed'
            or evidence['saved_final_reloaded_menus'] != 24516
            or not evidence['source_and_data_hashes_unchanged']
            or execution['status'] != 'actual_yaku2500_full_evidence_independently_audited'
            or execution['actual_fit_job'] != c['fit_job'] or not execution['original_launch_binding_unchanged']):
        raise ValueError('Actual complete final3B audit required; strategy grade remains separate')
    out = ROOT/c['out']; out.mkdir(parents=True, exist_ok=False)
    observed = {}; proofs = {}; files = {}
    script = ROOT/'cluster/curriculum-yaku-course-v3.slurm'
    for role in ('fit', 'audit'):
        job = c[role+'_job']; receipt_path = ROOT/'logs'/f'curriculum-yaku-course-exit-{job}.json'
        receipt = read(receipt_path)
        if receipt['phase'] != role:
            raise ValueError('Wrong actual parent phase receipt')
        archive = run/'fit-terminal-scheduler-private.json'
        if role == 'fit':
            captured = read(archive)
            if captured['exit'] != receipt:
                raise ValueError('Archived fit exit differs')
            raw = captured['raw']; scheduler_exit(receipt, raw, script)
            proofs[role] = dict(source='original_independent_audit_scontrol_archive', raw=raw, receipt=receipt)
            files[str(archive.resolve())] = sha(archive)
        else:
            result = subprocess.run(['scontrol', '-o', 'show', 'job', str(job)],
                capture_output=True, text=True, timeout=20)
            observed[role] = dict(scontrol_stdout=result.stdout, scontrol_stderr=result.stderr,
                                  scontrol_returncode=result.returncode)
            write(out/'scheduler-observation-private.json', observed)
            if result.returncode == 0:
                scheduler_exit(receipt, result.stdout, script)
                proofs[role] = dict(source='actual_terminal_scontrol', raw=result.stdout, receipt=receipt)
            else:
                accounting = subprocess.run(['sacct', '-X', '-n', '-P', '-j', str(job),
                    '--format=JobIDRaw,State,ExitCode,NodeList,AllocTRES,Start,End,SubmitLine'],
                    capture_output=True, text=True, timeout=20)
                write(out/'accounting-observation-private.json', dict(stdout=accounting.stdout,
                    stderr=accounting.stderr, returncode=accounting.returncode))
                if accounting.returncode != 0:
                    raise RuntimeError('Real terminal record absent and accounting unavailable; no parent admission')
                proofs[role] = accounting_proof(accounting.stdout, receipt, job, script)
        files[str(receipt_path.resolve())] = sha(receipt_path)
    for path in (Path(__file__), config_path, script, registered, completion_path, evidence_path, execution_path,
                 ROOT/'cluster/curriculum-yaku-terminal-capture-v1.slurm',
                 ROOT/'tests/core/test_curriculum_yaku_terminal_capture_v1.py',
                 ROOT/'docs/experiments/plans/curriculum-yaku-terminal-capture-20261123.md',
                 ROOT/'tools/seal_curriculum_basic_repair_v2.py'):
        files[str(path.resolve())] = sha(path)
    result = dict(status='actual_final3B_fit_and_audit_terminal_evidence_preserved_not_child_admitted',
        selected_step=2500, parent_registration=str(registered.resolve()),
        final_identity=evidence['final_identity'], final_policy_identity=evidence['final_policy_identity'],
        original_strategy_grade=evidence['stage3B_advancement'], scheduler_proofs=proofs,
        parent_fit_job=c['fit_job'], parent_audit_job=c['audit_job'],
        strategy_grade_overridden=False, child_parent_sealed=False, student_updates=0,
        automatic_release=False, raw_private=True, node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'])
    write(out/'terminal-evidence-private.json', result)
    write(out/'completion-binding-private.json', dict(files=files))
    print(json.dumps({k: v for k, v in result.items() if k not in
        ('scheduler_proofs', 'parent_registration')}), flush=True)


if __name__ == '__main__': capture()
