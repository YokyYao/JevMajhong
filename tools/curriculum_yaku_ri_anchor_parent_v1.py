"""Admit only the actual originally qualified augmented3A final2596."""
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import curriculum_ri_anchor_runtime_v1 as parent_runtime
from tools.curriculum_yaku_balanced_parent_v1 import gpu_only
from tools.curriculum_transfer import read, sha, canonical_protocol
from tools.curriculum_parent import rows, require_replay
from tools.curriculum_progression_parent_v1 import extend
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.capture_curriculum_paired_terminal_v2 import fields, terminal_record
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit
from tools.curriculum_ri_parent_anchor_weighted_v1 import METHOD as PARENT_LOSS
from tools.curriculum_progression_parent_loss_v1 import METHOD as PARENT_VALIDATION_LOSS
from tools.curriculum_yaku_ri_anchor_loss_v1 import METHOD as CHILD_LOSS

METHOD = 'audited_qualified_ri_anchor2596_same_policy_augmented25_yaku_v1'
STATUS = 'actual_qualified_ri_anchor2596_final_sealed_for_augmented_yaku_course'
PARENT_RUN = parent_runtime.RUN
PARENT_ENTRY = parent_runtime.ENTRY
CAPTURE_ARCHIVE = 'yaku-ri-anchor-parent-capture-terminal-private.json'


def require_qualified_parent(reg, done, evidence, execution, queue):
    parent_runtime.require_registration(reg)
    if (done['status'] != 'ri_anchor2596_fit_and_real_curves_complete'
            or done['optimizer_steps'] != 2596 or done['actual_records_seen'] != 189508
            or done['unique_train_rows'] != 92570 or done['loss_method'] != PARENT_LOSS
            or done['curve_steps'] != parent_runtime.POINTS
            or done['automatic_stage4_started'] is not False or done['automatic_release'] is not False
            or reg['validation_loss_method'] != PARENT_VALIDATION_LOSS
            or evidence['status'] != 'ri_anchor2596_final_evidence_audit_passed'
            or evidence['optimizer_steps'] != 2596 or evidence['actual_records_seen'] != 189508
            or evidence['stage3_advancement']['passed'] is not True
            or evidence['basic_retention']['passed'] is not True
            or evidence['retention_quality_cohort'] != 'original23688'
            or evidence['saved_final_reloaded_menus'] != 24998 or evidence['checked_complete_games'] != 320
            or evidence['all376_optimizer_states_match'] is not True
            or evidence['source_and_data_hashes_unchanged'] is not True
            or evidence['source_split_replay_and_exposure_exact'] is not True
            or evidence['final_reserved_grid_unused'] is not True
            or evidence['automatic_stage4_started'] is not False or evidence['automatic_release'] is not False
            or execution['status'] != 'actual_ri_anchor2596_full_evidence_independently_audited'
            or execution['automatic_stage3B'] is not False or execution['automatic_release'] is not False
            or str(execution['actual_fit_job']) != str(queue['fit_job'])):
        raise ValueError('Actual full2596 audit and unmodified original strategy/retention qualification required')


def actual_terminals(run, queue, evidence, files):
    if run.resolve() != PARENT_RUN.resolve():
        raise ValueError('Only the exact augmented2596 run may provide parent terminals')
    for name, expected in queue['files'].items():
        if sha(name) != expected:
            raise ValueError('Original before-fit parent binding changed')
        files[name] = expected
    observed = queue['observed']
    if (queue['no_held_or_released_before_activation_success'] is not True
            or any(fields(observed[role]).get('Command') != str(PARENT_ENTRY)
                or fields(observed[role]).get('JobState') != 'PENDING'
                or fields(observed[role]).get('JobId') != str(queue[role+'_job'])
                for role in ('fit', 'audit', 'capture'))
            or 'afterok:'+str(queue['activation_job'])+'(' not in fields(observed['fit']).get('Dependency', '')
            or 'afterok:'+str(queue['fit_job'])+'(' not in fields(observed['audit']).get('Dependency', '')
            or any('afterany:'+str(queue[role+'_job'])+'(' not in fields(observed['capture']).get('Dependency', '')
                for role in ('fit', 'audit'))):
        raise ValueError('Exact actual parent activation/fit/audit/capture dependency graph required')
    capture = run/'terminal-v1'; value = read(capture/'terminal-evidence-private.json')
    if (value['status'] != 'ri_anchor2596_actual_terminals_preserved_not_promoted'
            or value['strategy_grade'] != evidence['stage3_advancement']
            or value['automatic_stage3B'] is not False or value['automatic_release'] is not False):
        raise ValueError('Actual original parent terminal capture and unchanged grade required')
    extend(files, capture/'completion-binding-private.json')
    exits = {}
    for role in ('fit', 'audit'):
        proof = value['scheduler_proofs'][role]
        path = ROOT/'logs'/f"curriculum-ri-anchor-v1-exit-{queue[role+'_job']}.json"
        receipt = read(path)
        if (proof['job_id'] != str(queue[role+'_job']) or proof['successful'] is not True
                or proof['exit_receipt'] != receipt or receipt.get('phase') != role):
            raise ValueError('Actual original parent role exit differs')
        exits[role] = scheduler_exit(receipt, proof['raw'], PARENT_ENTRY)
        files[str(path)] = sha(path)
    job = str(queue['capture_job']); receipt_path = ROOT/'logs'/f'curriculum-ri-anchor-v1-exit-{job}.json'
    receipt = read(receipt_path)
    if receipt.get('phase') != 'capture':
        raise ValueError('Actual parent capture phase receipt required')
    archive = run/CAPTURE_ARCHIVE
    if archive.exists():
        cached = read(archive)
        if cached['exit'] != receipt:
            raise ValueError('Original capture archive differs')
        raw = cached['raw']
    else:
        actual = subprocess.run(['scontrol', '-o', 'show', 'job', job],
            capture_output=True, text=True, timeout=20)
        if actual.returncode:
            raise ValueError('Actual parent capture terminal unavailable; never invent it')
        raw = actual.stdout
        terminal_record(raw, job, PARENT_ENTRY)
        scheduler_exit(receipt, raw, PARENT_ENTRY)
        write(archive, dict(raw=raw, exit=receipt))
    exits['capture'] = scheduler_exit(receipt, raw, PARENT_ENTRY)
    for path in (archive, receipt_path, run/'fit-queue-private.json'):
        files[str(path)] = sha(path)
    return exits


def seal(config, out):
    gpu_only(); config = Path(config).resolve(); out = Path(out).resolve(); c = read(config)
    if (c['parent_selection'] != 'actual_registered_final2596_only'
            or c['admission_mode'] != 'original_augmented_stage3A_grade_pass_required'
            or (ROOT/c['parent_registration']).resolve() != parent_runtime.REGISTERED
            or c['loss_method'] != CHILD_LOSS or c['final_optimizer_steps'] != 2698):
        raise ValueError('Only an explicit qualified2596 to augmented2698 recipe may request a seal')
    registered = ROOT/c['parent_registration']; reg = read(registered); pc = reg['curriculum']
    fit = ROOT/pc['out']; run = registered.parent.parent
    evidence_path = ROOT/c['parent_evidence']; evidence = read(evidence_path)
    done = read(fit/'completion.json'); execution_path = run/'audit-execution-private.json'
    execution = read(execution_path); queue_path = ROOT/c['parent_jobs_from']; queue = read(queue_path)
    if queue_path.resolve() != run/'fit-queue-private.json' or evidence_path.resolve() != run/'final-evidence-private.json':
        raise ValueError('Exact original parent evidence and queue locations required')
    require_qualified_parent(reg, done, evidence, execution, queue)
    parent_runtime.require_launch(registered)
    files = dict(reg['inputs']); extend(files, run/'launch-binding-private.json')
    exits = actual_terminals(run, queue, evidence, files)
    data_path = ROOT/c['data_ready']; data = read(data_path)
    if (data['status'] != 'complete_augmented25_yaku_corpus_and_RI_coverage_plans_frozen_not_fit_admitted'
            or data['files']['train']['records'] != 96222 or data['files']['validation']['records'] != 25826
            or data['optimizer_steps_if_later_registered'] != 2698 or data['exposures_if_later_registered'] != 196954
            or not data['original_parent_and_route_line_bytes_preserved']
            or data['complete_cross_split_ID_group_visible_input_overlap'] != 0):
        raise ValueError('The exact complete augmented25 corpus and two epoch plans required')
    extend(files, data_path.parent/'completion-binding-private.json')
    numeric = ROOT/c['objective_preflight']; numerical = read(numeric)
    if (numerical['status'] != 'before_bound_actual_yaku25_full_numerical_and_epoch_repeat_passed_not_parent_or_fit_admitted'
            or numerical['unique_TRAIN_rows_checked'] != 96222 or numerical['actual_physical_epoch_mappings'] != 196954
            or numerical['all_actual_imports_bound_before_and_unchanged_after'] is not True
            or numerical['student_updates'] != 0 or numerical['focused_tests'] != 7):
        raise ValueError('Actual before-bound full25 numerical/epoch evidence required')
    extend(files, numeric.parent/'completion-binding-private.json')
    sys.path[:0] = [reg['source'], reg['training_config']['upstream_source']]
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.training.initialization import lineage_training_files, check_groups
    if (package_digest(Path(reg['source'])/'jevmahjong') != reg['source_sha256']
            or package_digest(Path(reg['training_config']['upstream_source'])/'jevany') != reg['training_config']['upstream_source_sha256']):
        raise ValueError('Actual pinned parent runtime changed')
    args = tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    for split, key in (('train', 'data'), ('validation', 'eval_data')):
        spec = data['files'][split]; path = ROOT/spec['path']
        if sha(path) != spec['sha256']:
            raise ValueError('Actual complete child source changed')
        require_replay(rows(args[key]), rows(path)); files[str(path)] = sha(path)
    protocol = canonical_protocol(read(fit/'protocol.json'))
    if protocol['config'] != reg['training_config']:
        raise ValueError('Actual parent training protocol differs')
    lineage = lineage_training_files(protocol)
    groups = check_groups(args['data'], ROOT/data['files']['train']['path'],
        ROOT/data['files']['validation']['path'], lineage[1:])
    model = bundle(fit/'training-run/checkpoints/step-002596')
    contract = read(fit/'policy-contracts-private.json')['2596']
    cache = fit/'evaluations/step-002596'; summary = read(cache/'summary.json')
    if (model['identity_sha256'] != evidence['final_identity']
            or contract['model_identity'] != model['identity_sha256']
            or contract['policy_identity'] != evidence['final_policy_identity']
            or len(rows(cache/'matches-private.jsonl')) != 64
            or summary['checkpoint_sha256'] != model['identity_sha256']
            or summary['policy_identity'] != contract['policy_identity']):
        raise ValueError('Actual audited final parameters, same policy and full64 cache required')
    paths = [Path(__file__), config, registered, evidence_path, execution_path, queue_path, numeric, data_path,
        run/'fit-terminal-scheduler-private.json', fit/'completion.json', fit/'protocol.json',
        fit/'policy-contracts-private.json', fit/'actual-replay-mapping-private.json',
        fit/'training-run/training_eval/step-002596/development/rows.json']
    paths += [fit/'metrics'/name for name in ('train_metrics.jsonl', 'validation_metrics.jsonl', 'evaluation_metrics.jsonl')]
    paths += [cache/name for name in ('protocol.json', 'summary.json', 'matches-private.jsonl')]
    paths += [Path(model['directory'])/name for name in model['files']]
    for path in paths:
        files[str(path)] = sha(path)
    if any(sha(path) != expected for path, expected in files.items()):
        raise ValueError('An actual parent/source prerequisite changed')
    result = dict(status=STATUS, method=METHOD, selected_step=2596, model=model,
        parent_loss_method=PARENT_LOSS, parent_validation_loss_method=PARENT_VALIDATION_LOSS,
        child_loss_method=CHILD_LOSS, policy=contract['policy'], policy_identity=contract['policy_identity'],
        original_parent_grade=evidence['stage3_advancement'], parent_registration=str(registered),
        parent_evidence=str(evidence_path), parent_training_file=args['data'], parent_validation_file=args['eval_data'],
        parent_evaluation_cache=str(cache), original_guarded_evaluation_cache=pc['original_guarded_evaluation_cache'],
        original_guarded_identity=pc['parent_identity'], child_data_ready=str(data_path),
        base=pc['base'], base_revision=pc['base_revision'], bound_files=files, group_proof=groups,
        scheduler_exits=exits, raw_grade_overridden=False, optimizer_resumed=False,
        selection_from_minimum_CE=False, child_fit_started=False)
    out.mkdir(parents=True, exist_ok=False); write(out/'parent-seal-private.json', result)


def verify_parent(cfg):
    spec = cfg['parent_initialization']; path = Path(spec['manifest']).resolve(); sealed = read(path)
    if (cfg.get('initialization') != 'continued' or cfg.get('loss_method') != CHILD_LOSS
            or spec['method'] != METHOD or sha(path) != spec['sha256']
            or sealed['method'] != METHOD or sealed['status'] != STATUS
            or spec['selected_step'] != 2596 or sealed['selected_step'] != 2596
            or sealed['parent_loss_method'] != PARENT_LOSS
            or sealed['parent_validation_loss_method'] != PARENT_VALIDATION_LOSS
            or sealed['child_loss_method'] != CHILD_LOSS or sealed['original_parent_grade']['passed'] is not True
            or sealed['raw_grade_overridden'] or sealed['optimizer_resumed'] or sealed['selection_from_minimum_CE']
            or sealed['model']['identity_sha256'] != spec['identity_sha256']
            or sealed['policy_identity'] != spec['policy_identity_sha256']
            or cfg['base'] != sealed['base'] or cfg['base_revision'] != sealed['base_revision']):
        raise ValueError('Exact originally qualified augmented2596 final ancestry required')
    for name, expected in sealed['bound_files'].items():
        if sha(name) != expected:
            raise ValueError('Actual sealed source changed')
    reg = read(sealed['parent_registration']); fit = ROOT/reg['curriculum']['out']
    parent_runtime.require_registration(reg); parent_runtime.require_launch(Path(sealed['parent_registration']))
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.unified_contract import bundle
    if (Path(spec['source']).resolve() != Path(reg['source']).resolve()
            or spec['source_sha256'] != reg['source_sha256']
            or package_digest(Path(reg['source'])/'jevmahjong') != reg['source_sha256']
            or bundle(sealed['model']['directory']) != sealed['model']
            or Path(spec['run']).resolve() != fit.resolve() or Path(cfg['out']).resolve() == fit.resolve()
            or spec['child_data_ready'] != sealed['child_data_ready']):
        raise ValueError('Actual source, parent bundle or independent output differs')
    args = tomllib.loads(Path(cfg['trainer_config']).read_text()); data = read(sealed['child_data_ready'])
    if args.get('resume') or Path(args['init_from']).resolve() != Path(sealed['model']['directory']).resolve():
        raise ValueError('Exact inherited adapter/head with a fresh optimizer required')
    for split, key in (('train', 'data'), ('validation', 'eval_data')):
        if Path(args[key]).resolve() != (ROOT/data['files'][split]['path']).resolve():
            raise ValueError('Child dataset differs from the explicit seal')
    protocol = canonical_protocol(read(fit/'protocol.json'))
    if protocol['config'] != reg['training_config']:
        raise ValueError('Actual parent protocol differs')
    receipt = dict(sealed['bound_files']); receipt[str(path)] = spec['sha256']
    return dict(model=sealed['model'], receipt=receipt, training_protocol=protocol,
        training_config=reg['training_config'], selected_step=2596)


@contextmanager
def parent_loader(initialization):
    previous = initialization.verified_parent; initialization.verified_parent = verify_parent
    try:
        yield
    finally:
        initialization.verified_parent = previous
