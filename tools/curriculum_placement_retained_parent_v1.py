"""Consume a future retained3B placement seal; no seal producer or job launch."""
from contextlib import contextmanager
from pathlib import Path

from tools import curriculum_placement_parent_v1 as common
from tools.curriculum_transfer import read, sha
from tools.curriculum_yaku_parent_runtime_v1 import require_registration as require_parent_registration
from tools.curriculum_yaku_parent_loss_v1 import METHOD as PARENT_LOSS
from tools.curriculum_placement_parent_loss_v2 import METHOD as CHILD_LOSS
from tools.capture_curriculum_paired_terminal_v2 import terminal_record
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT/'cluster/curriculum-yaku-parent-v1.slurm'


def require_parent_evidence(parent, done, evidence, execution, cap):
    require_parent_registration(parent)
    if (done['status'] != 'yaku2500_fit_and_real_curves_complete' or done['loss_method'] != PARENT_LOSS
            or done['optimizer_steps'] != 2500 or done['actual_records_seen'] != 182494
            or evidence['status'] != 'yaku2500_final_evidence_audit_passed'
            or evidence['loss_method'] != PARENT_LOSS or evidence['optimizer_steps'] != 2500
            or evidence['actual_records_seen'] != 182494
            or evidence['stage3B_advancement']['passed'] is not True
            or evidence['saved_final_reloaded_menus'] != 24516
            or not evidence['all376_optimizer_states_match'] or not evidence['source_and_data_hashes_unchanged']
            or evidence['final_reserved_grid_unused'] is not True
            or execution['status'] != 'actual_yaku2500_full_evidence_independently_audited'
            or execution['loss_method'] != PARENT_LOSS or not execution['original_launch_binding_unchanged']
            or cap['status'] != 'actual_retained_yaku_terminals_preserved_not_stage4_admitted'
            or cap['strategy_grade_overridden'] or cap['stage4_parent_sealed'] or cap['student_updates'] != 0
            or cap['original_strategy_grade'] != evidence['stage3B_advancement']
            or set(cap['scheduler_proofs']) != {'prepare', 'fit', 'audit'}
            or any(not p['successful'] for p in cap['scheduler_proofs'].values())
            or str(execution['actual_fit_job']) != str(cap['scheduler_proofs']['fit']['job_id'])):
        raise ValueError('Actual qualified retained3B2500, full audit and genuine successful phases required')


def verify_parent(cfg):
    sealed = read(cfg['parent_initialization']['manifest'])
    if (cfg['loss_method'] != CHILD_LOSS or sealed.get('parent_loss_method') != PARENT_LOSS
            or sealed.get('child_loss_method') != CHILD_LOSS
            or sealed.get('exact_retained_3B_replay_objective') is not True):
        raise ValueError('A future actual seal must explicitly bind the retained parent and child loss')
    registration = Path(sealed['parent_registration']); parent = read(registration)
    run = registration.parent.parent; fit = ROOT/parent['curriculum']['out']
    cap_path = run/'yaku-parent-terminal-v1/terminal-evidence-private.json'
    done_path = fit/'completion.json'; evidence_path = run/'final-evidence-private.json'
    execution_path = run/'audit-execution-private.json'
    cap = read(cap_path); execution = read(execution_path); evidence = read(evidence_path)
    require_parent_evidence(parent, read(done_path), evidence, execution, cap)
    if (sealed['model']['identity_sha256'] != evidence['final_identity']
            or sealed['parent_policy_identity'] != evidence['final_policy_identity']
            or Path(sealed['model']['directory']).resolve() != (fit/'training-run/checkpoints/step-002500').resolve()):
        raise ValueError('Only the actual audited final2500 parameters and policy may continue')
    paths = [registration, done_path, evidence_path, execution_path, cap_path,
        cap_path.parent/'completion-binding-private.json', run.parent/'curriculum-progression-parent-retention-v1/yaku-parent-queue-v1-private.json']
    for role, proof in cap['scheduler_proofs'].items():
        receipt = proof['exit_receipt']; actual = terminal_record(proof['raw'], proof['job_id'], ENTRY)
        if not actual['successful'] or receipt['phase'] != role: raise ValueError('Actual retained-parent phase differs')
        scheduler_exit(receipt, proof['raw'], ENTRY)
        path = ROOT/'logs'/f"curriculum-yaku-parent-exit-{proof['job_id']}.json"
        if read(path) != receipt: raise ValueError('Original retained-parent exit receipt changed')
        paths.append(path)
    queue = read(paths[6]); capture_job = queue['capture_job']
    if any(str(queue[role+'_job']) != str(cap['scheduler_proofs'][role]['job_id'])
           for role in ('prepare', 'fit', 'audit')):
        raise ValueError('Original held job identities differ from the actual phase proofs')
    capture_path = run/'capture-terminal-for-placement-private.json'; captured = read(capture_path)
    receipt_path = ROOT/'logs'/f'curriculum-yaku-parent-exit-{capture_job}.json'; receipt = read(receipt_path)
    if captured['exit'] != receipt or receipt['phase'] != 'capture': raise ValueError('Actual capture terminal required')
    scheduler_exit(receipt, captured['raw'], ENTRY)
    paths.extend((capture_path, receipt_path))
    for path in paths:
        if sealed['bound_files'].get(str(path.resolve())) != sha(path):
            raise ValueError('Actual retained-parent qualification/terminal evidence escaped the seal')
    return common.verify_parent(cfg)


@contextmanager
def parent_loader(initialization):
    original = initialization.verified_parent; initialization.verified_parent = verify_parent
    try: yield
    finally: initialization.verified_parent = original
