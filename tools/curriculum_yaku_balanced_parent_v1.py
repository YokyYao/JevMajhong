"""Seal only a genuinely qualified balanced-replay2472 final for future 3B.

Preserve the original grade, exact parent weights, unchanged policy and real
fit/audit/capture evidence. This module never starts a child fit.
"""
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import tomllib
import re

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha, canonical_protocol
from tools.curriculum_parent import rows, require_replay
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.capture_curriculum_paired_terminal_v2 import resource, fields
from tools.curriculum_progression_parent_v1 import extend
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit
from tools.curriculum_ri_balanced_replay_runtime_v1 import require_registration as require_parent_registration, require_launch
from tools.curriculum_ri_inverse_repeat_loss_v1 import METHOD as PARENT_LOSS
from tools.curriculum_progression_parent_loss_v1 import METHOD as CANONICAL_PARENT_LOSS
from tools.curriculum_yaku_parent_loss_v1 import METHOD as CHILD_LOSS

METHOD = 'audited_balanced_replay2472_same_policy_yaku_continuation_v1'
STATUS = 'actual_balanced_replay2472_final_sealed_for_yaku_course'


def gpu_only():
    actual = resource()  # Actual RUNNING job / node / allocated GPU; no env spoofing.
    limit = fields(actual).get('TimeLimit', '')
    match = re.fullmatch(r'(?:(\d+)-)?(\d+):(\d{2}):(\d{2})', limit)
    if match is None or not any(int(v or 0) for v in match.groups()):
        raise RuntimeError('Actual finite scheduler GPU time limit required')
    return actual


def require_qualified_parent(parent, done, evidence, execution, config):
    require_parent_registration(parent)
    if (done['status'] != 'balanced_replay2472_fit_and_real_curves_complete'
            or done['optimizer_steps'] != 2472 or done['actual_records_seen'] != 180456
            or done['loss_method'] != PARENT_LOSS
            or parent['validation_loss_method'] != CANONICAL_PARENT_LOSS
            or evidence['status'] != 'balanced_replay2472_final_evidence_audit_passed'
            or evidence['optimizer_steps'] != 2472 or evidence['actual_records_seen'] != 180456
            or evidence['stage3_advancement']['passed'] is not True
            or evidence['basic_retention']['passed'] is not True
            or evidence['saved_final_reloaded_menus'] != 23688
            or evidence['checked_complete_games'] != 320
            or evidence['all376_optimizer_states_match'] is not True
            or evidence['source_and_data_hashes_unchanged'] is not True
            or execution['status'] != 'actual_balanced_replay2472_full_evidence_independently_audited'
            or execution['automatic_release'] is not False
            or execution['automatic_stage3B'] is not False
            or str(execution['actual_fit_job']) != str(config['parent_fit_job'])):
        raise ValueError('Actual balanced-replay final2472 and original successful strategy grade required')


def actual_terminals(c, run, evidence, files):
    queue_path = ROOT/c['parent_jobs_from']; queue = read(queue_path)
    expected_queue = ROOT/'runs/curriculum-ri-balanced-replay-fit-v1/fit-queue-private.json'
    held = dict(t.split('=', 1) for t in queue['actual_observations']['fit'].split() if '=' in t)
    entry = ROOT/'cluster/curriculum-ri-balanced-replay-v1.slurm'
    if (queue_path.resolve() != expected_queue.resolve()
            or queue['status'] != 'actual_failed_control_standby_jobs_bound_while_held'
            or held.get('JobState') != 'PENDING' or held.get('Reason') != 'JobHeldUser'
            or held.get('Command') != str(entry)
            or held.get('JobId') != str(c['parent_fit_job'])
            or str(queue['fit_job']) != str(c['parent_fit_job'])
            or str(queue['audit_job']) != str(c['parent_audit_job'])
            or str(queue['capture_job']) != str(c['parent_terminal_capture_job'])):
        raise ValueError('Actual fit/audit/capture IDs bound before releasing the parent fit required')
    for p, h in queue['files'].items():
        if sha(p) != h: raise ValueError('Before-fit terminal binding changed: '+p)
        files[p] = h
    cap = run/'terminal-v1'; result = read(cap/'terminal-evidence-private.json')
    if (result['status'] != 'balanced_replay_actual_terminals_preserved_not_promoted'
            or result['strategy_grade'] != evidence['stage3_advancement']
            or result['automatic_stage3B'] is not False or result['automatic_release'] is not False):
        raise ValueError('Actual unmodified terminal capture with original grade required')
    extend(files, cap/'completion-binding-private.json')
    files[str(queue_path.resolve())] = sha(queue_path)
    files[str((cap/'completion-binding-private.json').resolve())] = sha(cap/'completion-binding-private.json')
    exits = {}
    for role in ('fit', 'audit'):
        proof = result['scheduler_proofs'][role]
        path = ROOT/'logs'/f"curriculum-ri-balanced-replay-exit-{c['parent_'+role+'_job']}.json"
        receipt = read(path)
        if (receipt != proof['exit_receipt'] or receipt.get('phase') != role
                or proof['job_id'] != str(c['parent_'+role+'_job']) or not proof['successful']):
            raise ValueError('Actual original parent phase exit differs')
        exits[role] = scheduler_exit(receipt, proof['raw'], entry)
        files[str(path.resolve())] = sha(path)
    job = c['parent_terminal_capture_job']
    receipt_path = ROOT/'logs'/f'curriculum-ri-balanced-replay-exit-{job}.json'
    receipt = read(receipt_path)
    if receipt.get('phase') != 'capture': raise ValueError('Exact actual capture phase exit required')
    archive = run/'capture-terminal-for-yaku-private.json'
    capture_entry = entry
    if archive.exists():
        captured = read(archive)
        if captured['exit'] != receipt: raise ValueError('Actual capture task exit differs')
        raw = captured['raw']
    else:
        observed = subprocess.run(['scontrol', '-o', 'show', 'job', str(job)], capture_output=True,
                                  text=True, timeout=20)
        prior = cap/'capture-job-observed-completed-private.txt'
        if observed.returncode and prior.exists():
            raw = prior.read_text()
            files[str(prior.resolve())] = sha(prior)
        elif observed.returncode:
            raise ValueError('Actual capture terminal unavailable; never synthesize scheduler evidence')
        else:
            raw = observed.stdout
        scheduler_exit(receipt, raw, capture_entry)
        write(archive, dict(raw=raw, exit=receipt))
    exits['capture'] = scheduler_exit(receipt, raw, capture_entry)
    for path in (archive, receipt_path): files[str(path.resolve())] = sha(path)
    return exits


def seal(config, out):
    gpu_only(); c = read(config)
    if (c['parent_selection'] != 'actual_registered_final2472_only'
            or c['admission_mode'] != 'original_stage3A_grade_pass_required'):
        raise ValueError('The actual qualified final 3A is required; no peak selection')
    registered = ROOT/c['parent_registration']; parent = read(registered); pc = parent['curriculum']
    fit = ROOT/pc['out']; run = registered.parent.parent
    evidence_path = ROOT/c['parent_evidence']; evidence = read(evidence_path); done = read(fit/'completion.json')
    execution = read(run/'audit-execution-private.json')
    require_qualified_parent(parent, done, evidence, execution, c)
    require_launch(registered)
    sys.path.insert(0, parent['source']); sys.path.insert(0, parent['training_config']['upstream_source'])
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.training.initialization import lineage_training_files, check_groups
    if package_digest(Path(parent['source'])/'jevmahjong') != parent['source_sha256']:
        raise ValueError('Parent pinned runtime changed')
    files = dict(parent['inputs'])
    binding = read(run/'launch-binding-private.json')
    files.update(binding['files'])
    exits = actual_terminals(c, run, evidence, files)
    data_path = ROOT/c['data_ready']; data = read(data_path)
    if (data['status'] != 'exact_replay_and_normalized_targeted_yaku_corpus_prepared_not_fit_admitted'
            or not data['original_replay_objects_labels_and_partitions_unchanged']
            or data['cross_split_family_and_public_input_overlap'] != 0):
        raise ValueError('Exact audited replay/yaku data required')
    recipe = ROOT/c['objective_preflight']
    extend(files, data_path.parent/'completion-binding-private.json')
    extend(files, recipe.parent/'completion-binding-private.json')
    objective = read(recipe)
    validation_path = ROOT/c['validation_adapter_preflight']; validation_proof = read(validation_path)
    if (objective['method'] != CHILD_LOSS or objective['actual_TRAIN_records'] != 91247
            or objective['maximum_actual_objective_or_CE_difference'] != 0
            or objective['maximum_representative_gradient_difference'] != 0
            or validation_proof['method'] != CHILD_LOSS or validation_proof['validation_menus'] != 24516
            or not validation_proof['canonical_CE_and_every_classification_metric_unchanged']
            or not validation_proof['retained_nine_component_profile_and_signed_accounting_verified']):
        raise ValueError('Actual retained-objective TRAIN and full validation contracts required')
    extend(files, validation_path.parent/'completion-binding-private.json')
    files[str(validation_path.resolve())] = sha(validation_path)
    epoch_recipe = ROOT/c['epoch_recipe']
    extend(files, epoch_recipe.parent/'completion-binding-private.json')
    files[str(epoch_recipe.resolve())] = sha(epoch_recipe)
    args = tomllib.loads(Path(parent['training_config']['trainer_config']).read_text())
    for split, key, size in (('train', 'data', 91247), ('validation', 'eval_data', 24516)):
        path = ROOT/data['files'][split]['path']
        if sha(path) != data['files'][split]['sha256'] or data['files'][split]['records'] != size:
            raise ValueError('Child data changed')
        require_replay(rows(args[key]), rows(path)); files[str(path.resolve())] = sha(path)
    protocol = canonical_protocol(read(fit/'protocol.json'))
    lineage = lineage_training_files(protocol)
    groups = check_groups(args['data'], ROOT/data['files']['train']['path'],
        ROOT/data['files']['validation']['path'], lineage[1:])
    model = bundle(fit/'training-run/checkpoints/step-002472')
    contract = read(fit/'policy-contracts-private.json')['2472']
    if (model['identity_sha256'] != evidence['final_identity']
            or contract['policy_identity'] != evidence['final_policy_identity']
            or contract['model_identity'] != model['identity_sha256']):
        raise ValueError('Exact audited final parameters and unchanged policy required')
    cache = fit/'evaluations/step-002472'; summary = read(cache/'summary.json')
    if (len(rows(cache/'matches-private.jsonl')) != 64
            or summary['checkpoint_sha256'] != model['identity_sha256']
            or summary['policy_identity'] != contract['policy_identity']):
        raise ValueError('Reuse actual full final64 parent cache')
    out.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), config, registered, evidence_path, data_path, recipe,
        run/'launch-binding-private.json', run/'fit-terminal-scheduler-private.json', run/'audit-execution-private.json',
        fit/'completion.json', fit/'protocol.json', fit/'policy-contracts-private.json']
    paths += [fit/'metrics'/name for name in ('train_metrics.jsonl', 'validation_metrics.jsonl', 'evaluation_metrics.jsonl')]
    paths += [cache/name for name in ('protocol.json', 'summary.json', 'matches-private.jsonl')]
    paths += [Path(model['directory'])/name for name in model['files']]
    paths += [fit/'training-run/training_eval/step-002472/development/rows.json',
        run/'launch-gate-private.json',run/'fit-queue-private.json',fit/'actual-replay-mapping-private.json']
    for path in paths: files[str(path.resolve())] = sha(path)
    if any(sha(p) != h for p, h in files.items()):
        raise ValueError('An immutable parent/data prerequisite changed')
    result = dict(status=STATUS, method=METHOD, selected_step=2472, model=model,
        parent_loss_method=PARENT_LOSS, child_loss_method=CHILD_LOSS,
        parent_canonical_validation_loss_method=CANONICAL_PARENT_LOSS,
        inverse_repeat_per_original_empirical_mass_preserved=True,
        original_epoch_recipe_used_only_for_weights_and_orders=True,
        policy=contract['policy'], policy_identity=contract['policy_identity'],
        original_parent_grade=evidence['stage3_advancement'], parent_registration=str(registered.resolve()),
        parent_evidence=str(evidence_path.resolve()), parent_training_file=args['data'], parent_validation_file=args['eval_data'],
        parent_evaluation_cache=str(cache.resolve()), original_guarded_evaluation_cache=pc['original_guarded_evaluation_cache'],
        original_guarded_identity=pc['parent_identity'], child_data_ready=str(data_path.resolve()),
        base=pc['base'], base_revision=pc['base_revision'], bound_files=files, group_proof=groups,
        scheduler_exits=exits, raw_grade_overridden=False, optimizer_resumed=False,
        selection_from_minimum_CE=False, child_fit_started=False)
    write(out/'parent-seal-private.json', result)


def verify_parent(cfg):
    spec = cfg['parent_initialization']; path = Path(spec['manifest']).resolve(); sealed = read(path)
    if (cfg.get('initialization') != 'continued' or cfg.get('loss_method') != CHILD_LOSS or spec['method'] != METHOD
            or sha(path) != spec['sha256'] or sealed['method'] != METHOD or sealed['status'] != STATUS
            or spec['selected_step'] != 2472 or sealed['selected_step'] != 2472
            or sealed['parent_loss_method'] != PARENT_LOSS or sealed['child_loss_method'] != CHILD_LOSS
            or sealed['parent_canonical_validation_loss_method'] != CANONICAL_PARENT_LOSS
            or sealed['inverse_repeat_per_original_empirical_mass_preserved'] is not True
            or sealed['original_parent_grade']['passed'] is not True or sealed['raw_grade_overridden']
            or sealed['optimizer_resumed'] or sealed['selection_from_minimum_CE']
            or sealed['model']['identity_sha256'] != spec['identity_sha256']
            or sealed['policy_identity'] != spec['policy_identity_sha256']
            or cfg['base'] != sealed['base'] or cfg['base_revision'] != sealed['base_revision']):
        raise ValueError('Exact original-qualified final 3A ancestry required')
    for p, h in sealed['bound_files'].items():
        if sha(p) != h: raise ValueError('Sealed prerequisite changed: '+p)
    parent = read(sealed['parent_registration']); fit = ROOT/parent['curriculum']['out']
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.unified_contract import bundle
    if (Path(spec['source']).resolve() != Path(parent['source']).resolve()
            or spec['source_sha256'] != parent['source_sha256']
            or package_digest(Path(parent['source'])/'jevmahjong') != parent['source_sha256']
            or bundle(sealed['model']['directory']) != sealed['model']
            or Path(cfg['out']).resolve() == fit.resolve() or Path(spec['run']).resolve() != fit.resolve()
            or spec['child_data_ready'] != sealed['child_data_ready']):
        raise ValueError('Pinned parent model/runtime or independent output differs')
    args = tomllib.loads(Path(cfg['trainer_config']).read_text()); data = read(sealed['child_data_ready'])
    if args.get('resume') or Path(args['init_from']).resolve() != Path(sealed['model']['directory']).resolve():
        raise ValueError('Same adapter/head with a new optimizer required')
    for split, key in (('train', 'data'), ('validation', 'eval_data')):
        if Path(args[key]).resolve() != (ROOT/data['files'][split]['path']).resolve():
            raise ValueError('Child dataset differs from seal')
    protocol = canonical_protocol(read(fit/'protocol.json'))
    if protocol['config'] != parent['training_config']:
        raise ValueError('Parent protocol differs')
    receipt = dict(sealed['bound_files']); receipt[str(path)] = spec['sha256']
    return dict(model=sealed['model'], receipt=receipt, training_protocol=protocol,
        training_config=parent['training_config'], selected_step=2472)


@contextmanager
def parent_loader(initialization):
    original = initialization.verified_parent; initialization.verified_parent = verify_parent
    try: yield
    finally: initialization.verified_parent = original


